"""Rule-based routers and the enforce/tell controller for topology ablations.

A *router* picks, each round, which agents every agent may send to:

    router(round_num, states, agent_ids, rng) -> {sender_id: [recipient_ids]}

``states`` is Simulator.all_states(); each SiloBenchAgent publishes
``states[i]["known"]`` (ids whose original data agent i has seen, via message
provenance) and ``states[i]["submission"]`` once it has submitted.

The *controller* is the Simulator topology_hook. Two independent switches give
the 2x2 ablation grid:

    enforce  tell   mode
    False    False  free-form (use no controller at all)
    True     False  filter  -- network silently drops off-route sends
    False    True   advise  -- agents are told the route; nothing is blocked
    True     True   hybrid  -- agents are told, and off-route sends bounce with an error

Restricted routers (random, gossip) always include the ring successor i+1 so
every agent's data can eventually reach everyone. The static templates (star,
chain) are the paper's optimal topologies for Levels I and II; both are connected.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

Router = Callable[[int, dict, list[int], random.Random], dict[int, list[int]]]


def _known(states: dict, aid: int) -> set[int]:
    return set(states.get(aid, {}).get("known", [aid]))


def _ring_next(agent_ids: list[int], i: int) -> int:
    return agent_ids[(agent_ids.index(i) + 1) % len(agent_ids)]


def full_router(round_num: int, states: dict, agent_ids: list[int], rng: random.Random) -> dict[int, list[int]]:
    return {i: [j for j in agent_ids if j != i] for i in agent_ids}


def star_router(round_num: int, states: dict, agent_ids: list[int], rng: random.Random) -> dict[int, list[int]]:
    """Static star around the lowest id: leaves send to the hub, the hub sends to everyone.

    The paper's optimal topology for Level I (aggregation); agent 0 is also the
    leader that emerges on its own there (Fig. 7).
    """
    hub, *leaves = agent_ids
    return {hub: list(leaves)} | {i: [hub] for i in leaves}


def chain_router(round_num: int, states: dict, agent_ids: list[int], rng: random.Random) -> dict[int, list[int]]:
    """Static bidirectional chain i <-> i+1 with no wraparound.

    The paper's optimal topology for Level II (mesh: prefix sum, moving average, ...).
    """
    n = len(agent_ids)
    return {a: [agent_ids[j] for j in (k - 1, k + 1) if 0 <= j < n] for k, a in enumerate(agent_ids)}


def make_random_router(k: int) -> Router:
    """Ring successor + k other agents chosen uniformly at random each round."""
    def router(round_num, states, agent_ids, rng):
        adj = {}
        for i in agent_ids:
            nxt = _ring_next(agent_ids, i)
            others = [j for j in agent_ids if j not in (i, nxt)]
            adj[i] = [nxt] + rng.sample(others, min(k, len(others)))
        return adj
    return router


def make_gossip_router(k: int) -> Router:
    """Ring successor + the k unsubmitted agents that lack the most of what i knows.

    Gain(i -> j) = |K_i \\ K_j|; only positive-gain targets are added, ties broken randomly.
    """
    def router(round_num, states, agent_ids, rng):
        known = {i: _known(states, i) for i in agent_ids}
        open_ = [j for j in agent_ids if "submission" not in states.get(j, {})]
        adj = {}
        for i in agent_ids:
            nxt = _ring_next(agent_ids, i)
            cands = [j for j in open_ if j not in (i, nxt)]
            rng.shuffle(cands)  # random tie-break; sort below is stable
            cands.sort(key=lambda j: len(known[i] - known[j]), reverse=True)
            adj[i] = [nxt] + [j for j in cands[:k] if known[i] - known[j]]
        return adj
    return router


def make_router(name: str, k: int) -> Router:
    if name == "full":
        return full_router
    if name == "star":
        return star_router
    if name == "chain":
        return chain_router
    if name == "random":
        return make_random_router(k)
    if name == "gossip":
        return make_gossip_router(k)
    raise ValueError(f"unknown router '{name}' (full | star | chain | random | gossip)")


def routing_note(round_num: int, targets: list[int], n_agents: int, enforce: bool) -> str:
    """Per-round message shown to an agent when tell=True."""
    ts = sorted(targets)
    if len(ts) == n_agents - 1:
        return f"[Routing, round {round_num}] You may send messages to any agent this round."
    if enforce:
        if not ts:
            return f"[Routing, round {round_num}] You cannot send messages this round."
        return (
            f"[Routing, round {round_num}] This round you can send messages ONLY to agents {ts}. "
            "Messages to any other agent will be rejected. If you need information from an agent "
            "not listed, ask one of these agents to forward it. The connections change every round."
        )
    if not ts:
        return f"[Routing suggestion, round {round_num}] No exchange is suggested for you this round."
    return (
        f"[Routing suggestion, round {round_num}] Exchange information with agents {ts} this round."
    )


@dataclass
class RoutingController:
    """Topology hook that applies a router with the given enforce/tell switches."""

    router: Router
    enforce: bool
    tell: bool
    rng: random.Random
    agents: dict[int, Any] = field(default_factory=dict, repr=False)
    history: list[dict[int, list[int]]] = field(default_factory=list, repr=False)  # route per round

    def attach(self, agents: dict[int, Any]) -> None:
        self.agents = agents
        for a in agents.values():
            a.enforce = self.enforce and self.tell  # bounce only when agents know the rules

    def __call__(self, round_num: int, states: dict) -> dict[int, list[int]] | None:
        ids = sorted(self.agents)
        adj = self.router(round_num, states, ids, self.rng)
        self.history.append(adj)
        for aid, a in self.agents.items():
            targets = adj.get(aid, [])
            a.allowed_targets = set(targets)  # also drives follow-rate telemetry in every mode
            a.routing_note = routing_note(round_num, targets, len(ids), self.enforce) if self.tell else ""
        return adj if self.enforce else None
