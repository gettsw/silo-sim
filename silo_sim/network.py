"""Explicit network model: topology + message delivery.

The Network is the *only* path through which agents exchange information.
Agents call network.send(); the simulator calls network.deliver() once per
round to push queued messages into per-agent inboxes.

Key design decisions:
- Messages sent in round R are delivered at the *start* of round R+1
  (one-round latency, matching SILO-BENCH's existing semantics).
- Topology is an adjacency dict: sender → set of reachable recipients.
  Sending to a non-neighbor raises ValueError.
- Packet loss and the RNG live here so experiments are reproducible
  when the caller provides a seeded random.Random instance.

Topology factory methods cover the common cases:
    Network.ring(...)
    Network.fully_connected(...)
    Network.star(center_id, ...)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import NamedTuple

from silo_sim.agent import SimMessage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from silo_sim.msg_filter import MessageFilter



# ---------------------------------------------------------------------------
# DeliveryResult — returned by Network.deliver()
# ---------------------------------------------------------------------------

class DeliveryResult(NamedTuple):
    """Summary of a single deliver() call."""

    inboxes: dict[int, list[SimMessage]]
    """Per-agent inboxes: agent_id → list of messages delivered this round."""

    delivered: int
    """Number of messages that reached their recipient."""

    dropped: int
    """Number of messages lost due to packet loss."""


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@dataclass
class Network:
    """Models topology, message queuing, and delivery with optional packet loss.

    Attributes:
        topology:  Directed adjacency dict — agent_id → set of neighbor ids
                   that the agent may send to.  Use the factory methods to
                   construct common topologies.
        rng:       A seeded random.Random instance.  Pass the same seed
                   across runs for reproducible packet-loss behaviour.
        loss_rate: Probability in [0, 1] that any individual message is
                   dropped during delivery.  Default 0.0 (lossless).
    """

    topology: dict[int, set[int]]
    rng: random.Random
    loss_rate: float = 0.0
    _queue: list[SimMessage] = field(default_factory=list, repr=False)
    msg_filter: "MessageFilter | None" = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def send(self, msg: SimMessage) -> None:
        """Queue a message for delivery next round.

        Silently drops the message if recipient is not a valid neighbor of sender.
        This allows the GNN topology hook to zero out edges without crashing
        agents that still attempt to broadcast to all peers.
        """
        neighbors = self.topology.get(msg.sender_id, set())
        if msg.recipient_id not in neighbors:
            return  # ponytail: silent drop; log if drop-rate monitoring matters
        if self.msg_filter is not None:
            ok, _ = self.msg_filter.check(msg)
            if not ok:
                return
            self.msg_filter.record(msg)
        self._queue.append(msg)

    def deliver(self) -> DeliveryResult:
        """Deliver queued messages, applying packet loss, then clear the queue.

        Returns a DeliveryResult with per-agent inboxes and delivery statistics.
        Messages are shuffled before loss is applied so that drop decisions are
        independent of send order.
        """
        queued = list(self._queue)
        self._queue = []

        self.rng.shuffle(queued)

        inboxes: dict[int, list[SimMessage]] = {}
        delivered = 0
        dropped = 0

        for msg in queued:
            if self.loss_rate > 0.0 and self.rng.random() < self.loss_rate:
                dropped += 1
                continue
            inboxes.setdefault(msg.recipient_id, []).append(msg)
            delivered += 1

        return DeliveryResult(inboxes=inboxes, delivered=delivered, dropped=dropped)

    def neighbors(self, agent_id: int) -> set[int]:
        """Return the set of agents that *agent_id* can send messages to."""
        return set(self.topology.get(agent_id, set()))

    def queue_size(self) -> int:
        """Number of messages currently in transit (not yet delivered)."""
        return len(self._queue)

    def update_adjacency(self, new_adj: dict[int, list[int]]) -> None:
        """Replace topology entries in-place from a new adjacency mapping.

        Used by topology hooks (e.g. GNNTopologyHook) to rewrite the graph
        each round without constructing a new Network object.

        Parameters
        ----------
        new_adj : dict mapping agent_id → list[int] of neighbor ids.
                  Each entry overwrites the corresponding key in self.topology.
                  Agent ids absent from new_adj keep their existing neighbors.
        """
        for src, targets in new_adj.items():
            self.topology[src] = set(targets)

    # ------------------------------------------------------------------
    # Topology factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def ring(
        agent_ids: list[int],
        rng: random.Random,
        loss_rate: float = 0.0,
        directed: bool = False,
    ) -> Network:
        """Bidirectional ring: each agent can send to its two immediate neighbors.

        With directed=True, edges go only clockwise (each agent has one out-neighbor).

        Args:
            agent_ids:  Ordered list of agent IDs.
            rng:        Seeded RNG for reproducibility.
            loss_rate:  Per-message drop probability.
            directed:   If True, create a directed (clockwise) ring.
        """
        n = len(agent_ids)
        if n < 2:
            raise ValueError("Ring topology requires at least 2 agents.")
        topology: dict[int, set[int]] = {}
        for i, aid in enumerate(agent_ids):
            right = agent_ids[(i + 1) % n]
            left = agent_ids[(i - 1) % n]
            if directed:
                topology[aid] = {right}
            else:
                topology[aid] = {left, right}
        return Network(topology=topology, rng=rng, loss_rate=loss_rate)

    @staticmethod
    def fully_connected(
        agent_ids: list[int],
        rng: random.Random,
        loss_rate: float = 0.0,
    ) -> Network:
        """Every agent can send to every other agent (all-to-all)."""
        id_set = set(agent_ids)
        topology = {aid: id_set - {aid} for aid in agent_ids}
        return Network(topology=topology, rng=rng, loss_rate=loss_rate)

    @staticmethod
    def star(
        center_id: int,
        agent_ids: list[int],
        rng: random.Random,
        loss_rate: float = 0.0,
    ) -> Network:
        """Star topology: all agents connect to a central hub.

        The center can send to all leaf nodes; each leaf can only send to the
        center.

        Args:
            center_id:  ID of the hub agent (must be in agent_ids).
            agent_ids:  All agent IDs (including center).
            rng:        Seeded RNG.
            loss_rate:  Per-message drop probability.
        """
        if center_id not in agent_ids:
            raise ValueError(f"center_id {center_id} must be in agent_ids.")
        leaves = [a for a in agent_ids if a != center_id]
        topology: dict[int, set[int]] = {center_id: set(leaves)}
        for leaf in leaves:
            topology[leaf] = {center_id}
        return Network(topology=topology, rng=rng, loss_rate=loss_rate)

    @staticmethod
    def from_edges(
        edges: list[tuple[int, int]],
        rng: random.Random,
        loss_rate: float = 0.0,
        bidirectional: bool = True,
    ) -> Network:
        """Build a network from an explicit edge list.

        Args:
            edges:          List of (sender, recipient) pairs.
            rng:            Seeded RNG.
            loss_rate:      Per-message drop probability.
            bidirectional:  If True, add reverse edges automatically.
        """
        topology: dict[int, set[int]] = {}
        for src, dst in edges:
            topology.setdefault(src, set()).add(dst)
            if bidirectional:
                topology.setdefault(dst, set()).add(src)
        return Network(topology=topology, rng=rng, loss_rate=loss_rate)
