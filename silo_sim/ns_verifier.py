"""Neuro-symbolic verifier (SILO-BENCH NS Verifier spec, layers 2-3).

Standalone: it sits in the message path (Network.msg_filter) and the round loop
(run_silo_case), and works with any topology — fixed, routed, or none.

Layer 1 (grammar-constrained decoding) is not built: add it if runs show agents
emitting malformed tool calls; parse_tool_calls already skips broken blocks.

Layer 2  NSVerifier.check — MessageFilter rules plus task invariants on every
         message. Violations are typed.
Layer 3  NSVerifier.end_round — a blocked sender gets a minimal error vector in
         its own context only (max MAX_RETRIES per edge); the epoch barrier checks
         submissions and reopens violators. Generic checks run for every task
         (premature submission on Level I/III, disagreement on unsegmented tasks);
         task barriers (sort_barrier for III-21) add task invariants on top.

The simulator is synchronous: every step-k message is verified inside
Network.send before step k+1 starts, so the spec's epoch barrier on messages
holds by construction.

The verifier may read every agent's input shard (never the expected output).
Feedback carries the violation type and, for violations between agents (boundary,
duplicate, missing value), the other agents' ids, never values: values would hand
agents input-derived data they have not communicated for; ids only say who to talk to.

Usage:
    result = run_silo_case(..., ns_verifier=True)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from silo_sim.agent import SimMessage
from silo_sim.msg_filter import MessageFilter, rule_no_duplicate

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Layer 2 — extracting data values
# ---------------------------------------------------------------------------

_NUM = r"-?\d+(?:\.\d+)?"
_LIST_RE = re.compile(rf"{_NUM}(?:\s*,\s*{_NUM}){{2,}}")  # run of 3+ comma-separated numbers, brackets optional


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def payload_values(content: Any, universe: set | None = None) -> list | None:
    """Data values a message or answer carries, or None if it carries none.

    Sources: a bare numeric list, the "values" field (always trusted as data), and
    runs of 3+ comma-separated numbers in free text or other dict fields. With *universe* (all input values),
    a text list counts as data only if at least half of it is input values, so range
    bounds / splitters like [0, 250, 500] pass through.
    ponytail: half-rule heuristic — a fully invented list in free text slips through;
    the "values" field has no such gap.
    """
    if isinstance(content, list):
        return content if content and all(_is_num(v) for v in content) else None
    vals: list = []
    text = content
    if isinstance(content, dict):
        field_vals = content.get("values")
        if isinstance(field_vals, list):
            vals += [v for v in field_vals if _is_num(v)]
        # agents pick their own keys ({"data": [...]}, {"sorted_data": [...]}): scan every other field
        text = json.dumps({k: v for k, v in content.items() if k != "values"}, default=str)
    if isinstance(text, str):
        for m in _LIST_RE.findall(text):
            lst = json.loads(f"[{m}]")
            if universe is None or 2 * sum(v in universe for v in lst) >= len(lst):
                vals += lst
    return vals or None


# ---------------------------------------------------------------------------
# Layer 3 — task barriers: (answers, verifier) -> [(agent_ids, violation_type)]
# ---------------------------------------------------------------------------

def _owners(x: float, parsed: dict[int, list]) -> list[int]:
    """Agents whose submitted range should contain x (the gap neighbours if nobody's does)."""
    inside = [i for i, vs in parsed.items() if min(vs) <= x <= max(vs)]
    if inside:
        return inside
    below = [i for i, vs in parsed.items() if max(vs) < x]
    above = [i for i, vs in parsed.items() if min(vs) > x]
    return sorted({max(below, key=lambda i: max(parsed[i]), default=None),
                   min(above, key=lambda i: min(parsed[i]), default=None)} - {None})


def sort_barrier(answers: dict[int, Any], v: NSVerifier) -> list[tuple[list[int], str]]:
    """III-21: local order, provenance, adjacent boundary, disjoint partitions, conservation."""
    out: list[tuple[list[int], str]] = []
    parsed: dict[int, list] = {}
    for i, a in answers.items():
        vals = payload_values(a)
        if vals is None:
            out.append(([i], "MALFORMED_ANSWER"))
            continue
        parsed[i] = vals
        if any(x > y for x, y in zip(vals, vals[1:])):
            out.append(([i], "LOCAL_ORDER"))
        if set(vals) - v.seen.get(i, set()):
            out.append(([i], "HALLUCINATED_VALUE"))

    for i in parsed:
        if i + 1 in parsed and max(parsed[i]) > min(parsed[i + 1]):
            out.append(([i, i + 1], "BOUNDARY_VIOLATION"))

    # Disjoint partitions: no input value placed more often than it exists.
    placed = Counter(x for vals in parsed.values() for x in vals)
    dupes = {tuple(sorted(i for i, vals in parsed.items() if x in vals))
             for x, c in placed.items() if x in v.inputs and c > v.inputs[x]}
    out += [(list(holders), "DUPLICATE_VALUE") for holders in sorted(dupes)]

    # Conservation: once everyone has submitted, every input value is placed.
    if len(parsed) == v.n_agents:
        missing = {tuple(_owners(x, parsed)) for x in v.inputs - placed}
        out += [(list(owners), "MISSING_VALUE") for owners in sorted(missing)]
    return out


# ponytail: explicit case lists. Provenance only holds where messages move raw input values;
# derived-value tasks (prefix sum, max) would false-positive. Add ids as invariants are written.
RELAY_CASES = {"III-21"}
BARRIERS: dict[str, Callable] = {"III-21": sort_barrier}


# ---------------------------------------------------------------------------
# Verifier (layers 2 + 3)
# ---------------------------------------------------------------------------

@dataclass
class NSVerifier(MessageFilter):
    """MessageFilter that types violations, feeds them back, and gates submissions.

    seen:     agent -> values it held or was sent; None disables provenance.
              ponytail: counted at send time, so a message later lost to loss_rate still counts.
    inputs:   multiset of every agent's input values (conservation / duplicate checks).
    barrier:  (answers, verifier) -> [(agent_ids, violation_type)], task-specific, run after each round.
    needs_all_data: Level I / III — every answer depends on all N shards (generic check).
    consensus:      not segmented — every agent must submit the same answer (generic check).
    """

    seen: dict[int, set] | None = None
    inputs: Counter = field(default_factory=Counter)
    n_agents: int = 0
    barrier: Callable | None = None
    needs_all_data: bool = False
    consensus: bool = False
    violations: list[dict] = field(default_factory=list)
    _notes: dict[int, list[str]] = field(default_factory=dict, repr=False)
    _retries: dict[tuple[int, int, str], int] = field(default_factory=dict, repr=False)
    _reopens: dict[int, int] = field(default_factory=dict, repr=False)
    _round: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._universe = set(self.inputs)
        # Resends are legitimate here: unread inboxes are cleared each turn and submitted agents
        # ignore theirs, so agents repeat data. Dropping repeats cut III-21 P from ~0.29 to 0.16.
        self.rules = [r for r in self.rules if r is not rule_no_duplicate]
        if self.seen is not None:
            self.rules = self.rules + [self._rule_provenance]

    @classmethod
    def for_case(cls, case: dict) -> NSVerifier:
        shards = {int(c["agent_id"]): json.loads(c["input_shard"]) if isinstance(c["input_shard"], str)
                  else c["input_shard"] for c in case["agent_configs"]}
        cid = case["case_id"]
        return cls(
            seen={i: set(s) for i, s in shards.items()} if cid in RELAY_CASES else None,
            inputs=Counter(x for s in shards.values() if isinstance(s, list) for x in s if _is_num(x)),
            n_agents=len(shards),
            barrier=BARRIERS.get(cid),
            needs_all_data=case.get("paradigm") in ("Paradigm I", "Paradigm III"),
            consensus=case["metadata"].get("is_segmented") is False,
        )

    def _generic(self, agents: dict, answers: dict[int, Any]) -> list[tuple[list[int], str]]:
        """Checks for every SILO-BENCH task, no ground truth needed."""
        out: list[tuple[list[int], str]] = []
        if self.needs_all_data:
            # `known` over-approximates what an agent saw, so this misses some premature
            # submissions but never flags a complete one.
            out += [([aid], "PREMATURE_SUBMISSION") for aid in answers
                    if getattr(agents[aid], "state", {}).get("known_at_submit", self.n_agents) < self.n_agents]
        if self.consensus and len(answers) > 1:
            keys = {aid: json.dumps(a, sort_keys=True, default=str) for aid, a in answers.items()}
            top, count = Counter(keys.values()).most_common(1)[0]
            # ponytail: majority is presumed right; a wrong majority reopens the right minority
            majority = top if 2 * count > len(keys) else None
            out += [([aid], "DISAGREEMENT") for aid, k in keys.items() if k != majority]
        return out

    def finish(self, agents: dict) -> int:
        """End of episode: reopened agents that never resubmitted get their rejected answer back."""
        return sum(agents[aid].restore_rejected() for aid in self._reopens)

    def _values(self, msg: SimMessage) -> list | None:
        return payload_values(msg.content, self._universe or None)

    def _rule_provenance(self, msg: SimMessage, _: list[SimMessage]) -> tuple[bool, str]:
        vals = self._values(msg)
        if vals is None:
            return True, ""
        return (False, "HALLUCINATED_VALUE") if set(vals) - self.seen.get(msg.sender_id, set()) else (True, "")

    def check(self, msg: SimMessage) -> tuple[bool, str]:
        ok, reason = super().check(msg)
        if not ok:
            vt = reason.upper().replace(" ", "_").replace("-", "_")
            self.violations.append({"round": self._round, "agents": [msg.sender_id, msg.recipient_id], "type": vt})
            key = (msg.sender_id, msg.recipient_id, vt)  # budget per edge and violation type
            self._retries[key] = self._retries.get(key, 0) + 1
            if self._retries[key] <= MAX_RETRIES:
                self._note(msg.sender_id, {"verifier": "LOGIC_VIOLATION", "violation_type": vt,
                                           "blocked_recipient": msg.recipient_id,
                                           "retries_left": MAX_RETRIES - self._retries[key]})
        return ok, reason

    def record(self, msg: SimMessage) -> None:
        super().record(msg)
        vals = self._values(msg) if self.seen is not None else None
        if vals:
            self.seen.setdefault(msg.recipient_id, set()).update(vals)

    def end_round(self, agents: dict, barrier: bool = True) -> None:
        """Layer 3: run the epoch barrier, then deliver queued feedback to each agent."""
        if barrier:
            answers = {aid: a.submission["answer"] for aid, a in agents.items() if a.submitted}
            found = self._generic(agents, answers) + (self.barrier(answers, self) if self.barrier else [])
            reopened: set[int] = set()  # one reopen (budget unit) per agent per round
            for ids, vt in found:
                self.violations.append({"round": self._round, "agents": ids, "type": vt})
                for aid in ids:
                    if aid in reopened or self._reopens.get(aid, 0) < MAX_RETRIES:
                        if aid not in reopened:
                            self._reopens[aid] = self._reopens.get(aid, 0) + 1
                            reopened.add(aid)
                            agents[aid].reopen()
                        vector = {"verifier": "EPOCH_BARRIER", "violation_type": vt,
                                  "action": "submission rejected, fix and resubmit"}
                        others = [x for x in ids if x != aid]
                        if others:  # who to coordinate with: agent ids only, never values
                            vector["with_agents"] = others
                        self._note(aid, vector)
        for aid, notes in self._notes.items():
            agents[aid].notify("[verifier]\n" + "\n".join(notes))
        self._notes.clear()
        self._round += 1

    def _note(self, aid: int, vector: dict) -> None:
        self._notes.setdefault(aid, []).append(json.dumps(vector, default=str))
