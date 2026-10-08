"""Round-based simulator.

The Simulator owns the clock and enforces the per-round contract:

    1. Network.deliver()  — push queued messages into per-agent inboxes
    2. agent.observe()    — every agent receives its inbox simultaneously
    3. agent.decide()     — every agent runs its policy and emits messages
    4. network.send()     — emitted messages are queued for next round
    5. clock += 1

All agents observe *before* any agent decides, which gives correct
simultaneous-round semantics regardless of the iteration order.

Usage::

    import random
    from silo_sim.agent import RuleAgent, SimMessage
    from silo_sim.network import Network
    from silo_sim.simulator import Simulator

    rng = random.Random(42)
    agent_ids = list(range(5))
    network = Network.ring(agent_ids, rng=rng)

    def echo_policy(agent):
        # forward max seen so far to both neighbours
        for msg in agent.inbox:
            agent.state["max"] = max(agent.state.get("max", msg.content), msg.content)
        return [
            SimMessage(agent.agent_id, n, agent.state["max"], 0)
            for n in agent.state.get("neighbors", [])
        ]

    agents = {
        i: RuleAgent(i, echo_policy, _state={"max": i, "neighbors": list(network.neighbors(i))})
        for i in agent_ids
    }
    sim = Simulator(agents=agents, network=network)
    history = sim.run(rounds=10)
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from silo_sim.agent import BaseAgent, SimMessage
from silo_sim.network import Network


# ---------------------------------------------------------------------------
# SimulationStep — per-round snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationStep:
    """Immutable record of one simulation round.

    Attributes:
        round:              Round number (1-indexed; 0 is before any step).
        messages_sent:      Messages placed into the network this round.
        messages_delivered: Messages that reached their recipients.
        messages_dropped:   Messages lost to packet loss.
        agent_states:       Shallow copy of each agent's state dict.
        sent_edges:         (sender, recipient) of every message sent this round
                            (before the network's topology check).
        delivered_edges:    (sender, recipient) of every message delivered at the
                            start of this round (i.e. sent last round).
    """

    round: int
    messages_sent: int
    messages_delivered: int
    messages_dropped: int
    agent_states: dict[int, Any]
    sent_edges: tuple = ()
    delivered_edges: tuple = ()


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class Simulator:
    """Round-based multi-agent simulator.

    Attributes:
        agents:   Dict mapping agent_id → BaseAgent (RuleAgent, LLMAgent, …).
        network:  The Network that routes messages between agents.
        clock:    Current round number.  Starts at 0; incremented by step().
    """

    agents: dict[int, BaseAgent]
    network: Network
    clock: int = 0
    _history: list[SimulationStep] = field(default_factory=list, repr=False)
    _epoch_history: list[EpochResult] = field(default_factory=list, repr=False)
    topology_hook: Any = field(default=None, repr=False)
    """Optional callable invoked at the **top** of every round, before message
    delivery.  Signature::

        new_adj = topology_hook(round_num, agent_states)

    *round_num* is the current clock value (0-based, pre-increment).
    *agent_states* is the snapshot returned by :meth:`all_states`.

    The callable must return either a dict[int, list[int]] (which the
    Simulator applies to the Network via
    :meth:`~silo_sim.network.Network.update_adjacency`)
    or None to leave the topology unchanged.
    """
    max_workers: int = 1
    """Agents deciding concurrently per round (>1 for LLM agents; decisions are
    independent within a round, and sends are still queued in agent order)."""

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self) -> SimulationStep:
        """Execute one simulation round and return a step summary.

        Round contract (enforced in this order):
            1. Deliver queued messages → per-agent inboxes
            2. All agents observe their inbox simultaneously
            3. All agents decide and emit outbound messages
            4. Outbound messages are queued in the network
            5. Clock advances by 1
        """
        # Phase 0: topology hook — rewrite adjacency before this round's delivery
        if self.topology_hook is not None:
            new_adj = self.topology_hook(self.clock, self.all_states())
            if new_adj is not None:
                self.network.update_adjacency(new_adj)

        # Phase 1: deliver
        result = self.network.deliver()

        # Phase 2: observe (all agents before any decides)
        for agent_id, agent in self.agents.items():
            inbox = result.inboxes.get(agent_id, [])
            agent.observe(inbox)

        # Phase 3: decide + send
        messages_sent = 0
        agents = list(self.agents.values())
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                outbounds = list(ex.map(lambda a: a.decide(), agents))
        else:
            outbounds = [a.decide() for a in agents]
        sent_edges = tuple((m.sender_id, m.recipient_id) for out in outbounds for m in out)
        for outbound in outbounds:
            for msg in outbound:
                self.network.send(msg)
                messages_sent += 1

        # Phase 4: advance clock
        self.clock += 1

        step = SimulationStep(
            round=self.clock,
            messages_sent=messages_sent,
            messages_delivered=result.delivered,
            messages_dropped=result.dropped,
            agent_states={aid: dict(a.state) for aid, a in self.agents.items()},
            sent_edges=sent_edges,
            delivered_edges=tuple((m.sender_id, m.recipient_id)
                                  for inbox in result.inboxes.values() for m in inbox),
        )
        self._history.append(step)
        return step

    def run(self, rounds: int) -> list[SimulationStep]:
        """Execute *rounds* steps and return the full history.

        Args:
            rounds: Number of rounds to simulate.  Must be >= 0.

        Returns:
            List of SimulationStep, one per round executed.
        """
        if rounds < 0:
            raise ValueError(f"rounds must be >= 0, got {rounds}")
        return [self.step() for _ in range(rounds)]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[SimulationStep]:
        """All recorded steps since the simulator was created."""
        return list(self._history)

    def agent_state(self, agent_id: int) -> dict[str, Any]:
        """Return the current state of a single agent."""
        return dict(self.agents[agent_id].state)

    def all_states(self) -> dict[int, Any]:
        """Return a snapshot of every agent's state."""
        return {aid: dict(a.state) for aid, a in self.agents.items()}

    def run_epoch(
        self,
        rounds: int,
        *,
        epoch: int | None = None,
        verifier: "Any | None" = None,
    ) -> "EpochResult":
        """Run one epoch of *rounds* rounds, optionally verifying at the end.

        Parameters
        ----------
        rounds:   Number of simulation rounds in this epoch.
        epoch:    Explicit epoch number (1-indexed).  Defaults to
                  ``len(self._epoch_history) + 1``.
        verifier: Optional :class:`~silo_sim.verification.Verifier`.
                  If provided, ``verifier.verify()`` is called after all
                  rounds complete and the result is stored in the returned
                  :class:`EpochResult`.

        Returns
        -------
        EpochResult
        """
        epoch_num = epoch if epoch is not None else (len(self._epoch_history) + 1)
        steps = self.run(rounds)

        verification = None
        if verifier is not None:
            verification = verifier.verify(epoch_num, self.all_states(), steps)

        result = EpochResult(
            epoch=epoch_num,
            rounds_run=len(steps),
            steps=tuple(steps),
            verification=verification,
        )
        self._epoch_history.append(result)
        return result

    def run_epochs(
        self,
        num_epochs: int,
        rounds_per_epoch: int,
        verifier: "Any | None" = None,
        stop_fn: "Any | None" = None,
    ) -> "list[EpochResult]":
        """Run *num_epochs* epochs, each of *rounds_per_epoch* rounds.

        Parameters
        ----------
        num_epochs:        Total number of epochs to run (upper bound).
        rounds_per_epoch:  Rounds per epoch.
        verifier:          Optional :class:`~silo_sim.verification.Verifier`
                           called after each epoch.
        stop_fn:           Optional ``Callable[[EpochResult], bool]``.  If it
                           returns True after an epoch, the loop stops early.
                           Useful for convergence checks or early termination
                           on perfect verification score.

        Returns
        -------
        list[EpochResult] — one entry per epoch actually run.

        Example::

            results = sim.run_epochs(
                num_epochs=5,
                rounds_per_epoch=3,
                verifier=PassVerifier(),
                stop_fn=lambda r: r.verification and r.verification.score == 1.0,
            )
        """
        results: list[EpochResult] = []
        for i in range(1, num_epochs + 1):
            result = self.run_epoch(rounds_per_epoch, epoch=i, verifier=verifier)
            results.append(result)
            if stop_fn is not None and stop_fn(result):
                break
        return results

    @property
    def epoch_history(self) -> "list[EpochResult]":
        """All EpochResults recorded since the Simulator was created."""
        return list(self._epoch_history)




# ---------------------------------------------------------------------------
# EpochResult — per-epoch snapshot (defined here to avoid circular imports;
# verification.py imports SimulationStep from this module, not vice-versa)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochResult:
    """Summary of one epoch of simulation.

    An epoch is a named group of rounds.  After all rounds complete the
    attached Verifier (if any) runs and its result is stored here.

    Attributes
    ----------
    epoch:        Epoch number, 1-indexed.
    rounds_run:   Number of rounds executed in this epoch.
    steps:        Per-round SimulationStep records.
    verification: Result of the attached Verifier, or None if no Verifier
                  was provided.
    """

    epoch: int
    rounds_run: int
    steps: tuple[SimulationStep, ...]
    verification: Any  # VerificationResult | None — typed as Any to avoid
                       # circular import; use TYPE_CHECKING annotation below


# Patch the type annotation without importing at module level
EpochResult.__annotations__["verification"] = "VerificationResult | None"
