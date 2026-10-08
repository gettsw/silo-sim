"""Runner that executes a SILO-BENCH benchmark case through the Simulator.

Entry point: run_silo_case()

This reproduces the original benchmark's agent behaviour using the new
simulation architecture:
  - SiloBenchAgent handles XML tool-call parsing
  - Network mediates all inter-agent message delivery
  - Metrics are computed by the original src/utils/metrics.py functions

Usage::

    from silo_sim import make_openai_fn
    from silo_sim.silo_runner import run_silo_case

    llm_fn = make_openai_fn(api_base="...", api_key="...", model="gpt-4o")
    result = run_silo_case(
        case_path="benchmarks/I-01_n2.json",
        protocol="broadcast",
        llm_fn=llm_fn,
        max_rounds=10,
    )
    print(result.summary())

For testing, pass make_mock_fn([...]) as llm_fn — no API key required.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from silo_sim.network import Network
from silo_sim.ns_verifier import NSVerifier
from silo_sim.silo_agent import LLMFn, SiloBenchAgent
from silo_sim.simulator import SimulationStep, Simulator
from silo_sim.verification import AnswerVerifier, VerificationResult
from silo_sim.utils.metrics import (
    compute_communication_density,
    compute_partial_correctness,
    compute_success_rate,
    compute_token_consumption,
)
from silo_sim.utils.prompts import generate_system_prompt

# Paradigm label → level string expected by compute_partial_correctness
_PARADIGM_LEVEL = {
    "Paradigm I": "I",
    "Paradigm II": "II",
    "Paradigm III": "III",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiloResult:
    """Outcome of running one SILO-BENCH case through the Simulator.

    Attributes
    ----------
    case_id:        e.g. "I-01"
    protocol:       "msg" | "broadcast"
    rounds_run:     actual rounds simulated
    all_submitted:  True if every agent called submit_result
    submissions:    list of {"agent_id": int, "answer": Any}
    S:              success rate ∈ [0, 1]
    P:              partial correctness ∈ [0, 1]
    C:              avg output tokens per round
    D:              communication density (may exceed 1)
    steps:          full SimulationStep history
    """

    case_id: str
    protocol: str
    rounds_run: int
    all_submitted: bool
    submissions: tuple[dict[str, Any], ...]
    S: float
    P: float
    C: float
    D: float
    steps: tuple[SimulationStep, ...]
    verification: VerificationResult | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.case_id} | {self.protocol}] "
            f"rounds={self.rounds_run} all_submitted={self.all_submitted} "
            f"S={self.S:.3f} P={self.P:.3f} C={self.C:.1f} D={self.D:.3f}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_silo_case(
    case_path: str | Path,
    protocol: str,
    llm_fn: LLMFn,
    max_rounds: int = 10,
    topology: str = "fully_connected",
    seed: int = 42,
    loss_rate: float = 0.0,
    token_counter: Callable[[str], int] | None = None,
    topology_hook: Any | None = None,
    routing: Any | None = None,
    max_workers: int = 1,
    ns_verifier: bool = False,
) -> SiloResult:
    """Run one SILO-BENCH benchmark case through the Simulator.

    Parameters
    ----------
    case_path:
        Path to a benchmark JSON file (e.g. "benchmarks/I-01_n2.json").
    protocol:
        "msg" | "broadcast".
    llm_fn:
        Provider-agnostic LLM callable — same interface as LLMAgent.
        Use make_openai_fn() for real runs, make_mock_fn() for tests.
    max_rounds:
        Hard cap on simulation rounds.  The run stops earlier if all
        agents have submitted.
    topology:
        "fully_connected" (default) | "ring" | "star:<center_id>".
        SILO-BENCH assumes any agent can message any other, so
        fully_connected is the canonical choice.
    seed:
        RNG seed for reproducibility.
    loss_rate:
        Fraction of messages to drop randomly [0.0, 1.0].
    token_counter:
        Optional callable(raw: str) -> int for exact output-token counts.
        Falls back to word-count proxy if None.
    routing:
        Optional silo_sim.routing.RoutingController; used as the
        topology hook (overrides topology_hook) and attached to the agents.
    max_workers:
        Agents deciding concurrently per round (use num_agents for LLM runs).
    ns_verifier:
        Attach the neuro-symbolic verifier (message invariants, sender feedback,
        epoch barrier). See src/simulation/ns_verifier.py.

    Returns
    -------
    SiloResult
    """
    case = _load_case(case_path)
    num_agents: int = case["metadata"]["num_agents"]
    agent_ids = list(range(num_agents))

    # Build network
    rng = random.Random(seed)
    network = _build_network(topology, agent_ids, rng, loss_rate)
    ns = NSVerifier.for_case(case) if ns_verifier else None
    network.msg_filter = ns

    # Build agents
    agents: dict[int, SiloBenchAgent] = {}
    for cfg in case["agent_configs"]:
        aid: int = cfg["agent_id"]
        sys_prompt = generate_system_prompt(protocol, aid, num_agents)
        agents[aid] = SiloBenchAgent(
            agent_id=aid,
            llm_fn=llm_fn,
            protocol=protocol,
            system_prompt=sys_prompt,
            task_prompt=cfg["user_prompt"],
            num_agents=num_agents,
            token_counter=token_counter,
        )

    # Run Simulator
    if routing is not None:
        routing.attach(agents)
        topology_hook = routing
    sim = Simulator(agents=agents, network=network, topology_hook=topology_hook,
                    max_workers=max_workers)
    steps: list[SimulationStep] = []

    for r in range(max_rounds):
        step = sim.step()
        steps.append(step)
        if ns is not None:
            # no barrier on the last round: a reopened agent would lose its answer with no turn left
            ns.end_round(agents, barrier=r < max_rounds - 1)
        if all(a.submitted for a in agents.values()):
            break

    # a reopened agent that never resubmitted keeps its rejected answer rather than scoring 0
    restored = ns.finish(agents) if ns is not None else 0
    all_submitted = all(a.submitted for a in agents.values())

    # Collect submissions
    submissions = [
        a.submission
        for a in agents.values()
        if a.submission is not None
    ]

    # Aggregate telemetry
    total_output_tokens = sum(a._output_tokens for a in agents.values())
    total_messages_sent = sum(a._messages_sent for a in agents.values())

    # Compute SILO-BENCH metrics
    paradigm: str = case.get("paradigm", "Paradigm I")
    level = _PARADIGM_LEVEL.get(paradigm, "I")
    expected_output: dict = case["expected_output"]

    S = compute_success_rate(submissions, expected_output)
    P = compute_partial_correctness(submissions, expected_output, level)
    C = compute_token_consumption(total_output_tokens, len(steps))
    D = compute_communication_density(total_messages_sent, num_agents)

    verifier = AnswerVerifier(expected_output, paradigm_level=level)
    verification = verifier.verify(
        epoch=1,
        agent_states={aid: a.state for aid, a in agents.items()},
        steps=steps,
    )

    telemetry = _telemetry(agents, submissions, steps)
    trace = _trace(agents, submissions, steps, routing, expected_output, level)
    if ns is not None:
        telemetry["ns_violations"] = len(ns.violations)
        telemetry["ns_restored"] = restored
        trace["violations"] = ns.violations

    return SiloResult(
        case_id=case["case_id"],
        protocol=protocol,
        rounds_run=len(steps),
        all_submitted=all_submitted,
        submissions=tuple(submissions),
        S=S,
        P=P,
        C=C,
        D=D,
        steps=tuple(steps),
        verification=verification,
        telemetry=telemetry,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _telemetry(agents: dict[int, SiloBenchAgent], submissions: list[dict],
               steps: list[SimulationStep]) -> dict[str, Any]:
    """Failure-mode proxies (paper Sec. 5.2) and routing compliance for one run."""
    n = len(agents)
    ks = [a.state["known_at_submit"] for a in agents.values() if "known_at_submit" in a.state]
    routed = sum(a._sends_routed for a in agents.values())
    offered = sum(a._route_offered for a in agents.values())
    return {
        "n_submitted": len(submissions),
        "distinct_answers": len({json.dumps(s["answer"], sort_keys=True, default=str) for s in submissions}),
        # fraction of the N data shards the submitter had (transitively) seen
        "known_frac_at_submit": sum(ks) / (len(ks) * n) if ks else 0.0,
        # submitted without having seen every shard: premature for Level I/III tasks
        "frac_submit_incomplete": sum(k < n for k in ks) / len(ks) if ks else 0.0,
        "messages_sent": sum(a._messages_sent for a in agents.values()),
        # filter mode drops off-route sends silently in the network, so sent != delivered
        # (messages still queued when the run ends are not counted)
        "delivered": sum(st.messages_delivered for st in steps),
        "bounced": sum(a._bounced for a in agents.values()),
        # of sends made while a route existed, share that went to an allowed target
        "follow_rate": sum(a._sends_on_route for a in agents.values()) / routed if routed else None,
        # of allowed targets offered, share actually messaged
        "route_utilization": sum(a._route_used for a in agents.values()) / offered if offered else None,
    }


def _trace(agents: dict[int, SiloBenchAgent], submissions: list[dict], steps: list[SimulationStep],
           routing: Any, expected_output: dict, level: str) -> dict[str, Any]:
    """Per-round record of routing decisions and outcomes, for learning from past runs.

    rounds[r] describes round r+1:
      route      -- targets the router allowed/suggested at the top of the round (None = no router)
      delivered  -- [sender, recipient] delivered at the start of the round (sent the round before)
      sent       -- [sender, recipient] sent during the round (off-route sends included in filter mode)
      known      -- per agent (index = id), ids whose data it had seen at the end of the round
      submitted  -- ids that had submitted by the end of the round
    q[i] is agent i's own partial-correctness score (P = mean(q); 0 if it never submitted).
    """
    n = len(agents)
    by_agent = {s["agent_id"]: s for s in submissions}
    # every level's P is (1/N) * sum over submitters of q_i, so one submission recovers q_i
    q = [n * compute_partial_correctness([by_agent[i]], expected_output, level) if i in by_agent else 0.0
         for i in range(n)]
    routes = getattr(routing, "history", None) or []
    rounds = []
    for r, st in enumerate(steps):
        route = routes[r] if r < len(routes) else None
        rounds.append({
            "route": {str(k): list(v) for k, v in route.items()} if route is not None else None,
            "delivered": [list(e) for e in st.delivered_edges],
            "sent": [list(e) for e in st.sent_edges],
            "known": [st.agent_states[i].get("known", [i]) for i in range(n)],
            "submitted": [i for i in range(n) if "submission" in st.agent_states[i]],
        })
    return {"q": q, "rounds": rounds}


def _load_case(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def _build_network(
    topology: str,
    agent_ids: list[int],
    rng: random.Random,
    loss_rate: float,
) -> Network:
    if topology == "fully_connected":
        return Network.fully_connected(agent_ids, rng, loss_rate)
    if topology == "ring":
        return Network.ring(agent_ids, rng, loss_rate)
    if topology.startswith("star:"):
        center_id = int(topology.split(":")[1])
        return Network.star(center_id, agent_ids, rng, loss_rate)
    raise ValueError(
        f"Unknown topology '{topology}'. "
        "Use 'fully_connected', 'ring', or 'star:<center_id>'."
    )
