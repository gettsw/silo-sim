"""Basic tests for src/simulation — milestone 1.

Covers:
    - Agent creation and state
    - Message creation
    - Network topology factories (ring, star, fully-connected)
    - send() / deliver() mechanics
    - Packet loss
    - Simulator round execution and clock
    - Information propagation across a ring
    - Determinism with a fixed seed
    - SILO-BENCH smoke test (existing code still importable)
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from silo_sim.agent import RuleAgent, SimMessage
from silo_sim.network import Network
from silo_sim.simulator import Simulator, SimulationStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_policy(agent: RuleAgent) -> list[SimMessage]:
    """A policy that never sends anything."""
    return []


def _make_ring_sim(
    n: int,
    initial_values: list[Any],
    seed: int = 0,
    loss_rate: float = 0.0,
) -> Simulator:
    """Build a ring of n RuleAgents that propagate their max seen value."""
    rng = random.Random(seed)
    agent_ids = list(range(n))
    network = Network.ring(agent_ids, rng=rng, loss_rate=loss_rate)

    def max_propagation_policy(agent: RuleAgent) -> list[SimMessage]:
        for msg in agent.inbox:
            if msg.content > agent.state["max"]:
                agent.state["max"] = msg.content
        return [
            SimMessage(agent.agent_id, neighbor, agent.state["max"], 0)
            for neighbor in agent.state["neighbors"]
        ]

    agents = {
        i: RuleAgent(
            agent_id=i,
            policy_fn=max_propagation_policy,
            _state={
                "max": initial_values[i],
                "neighbors": sorted(network.neighbors(i)),
            },
        )
        for i in agent_ids
    }
    return Simulator(agents=agents, network=network)


# ===========================================================================
# Agent tests
# ===========================================================================

class TestRuleAgent:
    def test_creation(self):
        agent = RuleAgent(agent_id=7, policy_fn=_null_policy)
        assert agent.agent_id == 7

    def test_initial_state_empty(self):
        agent = RuleAgent(agent_id=0, policy_fn=_null_policy)
        assert agent.state == {}

    def test_initial_state_provided(self):
        agent = RuleAgent(agent_id=0, policy_fn=_null_policy, _state={"x": 99})
        assert agent.state["x"] == 99

    def test_state_is_mutable(self):
        agent = RuleAgent(agent_id=0, policy_fn=_null_policy)
        agent.state["value"] = 42
        assert agent.state["value"] == 42

    def test_observe_stores_messages(self):
        agent = RuleAgent(agent_id=0, policy_fn=_null_policy)
        msgs = [SimMessage(1, 0, "hello", 1)]
        agent.observe(msgs)
        assert agent.inbox == msgs

    def test_decide_clears_inbox(self):
        agent = RuleAgent(agent_id=0, policy_fn=_null_policy)
        agent.observe([SimMessage(1, 0, "hi", 1)])
        agent.decide()
        assert agent.inbox == []

    def test_policy_receives_inbox(self):
        seen = []

        def capturing_policy(a: RuleAgent) -> list[SimMessage]:
            seen.extend(a.inbox)
            return []

        agent = RuleAgent(agent_id=0, policy_fn=capturing_policy)
        msg = SimMessage(1, 0, "data", 1)
        agent.observe([msg])
        agent.decide()
        assert seen == [msg]

    def test_policy_can_mutate_state(self):
        def state_writing_policy(a: RuleAgent) -> list[SimMessage]:
            a.state["rounds"] = a.state.get("rounds", 0) + 1
            return []

        agent = RuleAgent(agent_id=0, policy_fn=state_writing_policy)
        agent.observe([])
        agent.decide()
        agent.observe([])
        agent.decide()
        assert agent.state["rounds"] == 2

    def test_satisfies_base_agent_protocol(self):
        from silo_sim.agent import BaseAgent
        agent = RuleAgent(agent_id=0, policy_fn=_null_policy)
        assert isinstance(agent, BaseAgent)


# ===========================================================================
# SimMessage tests
# ===========================================================================

class TestSimMessage:
    def test_fields(self):
        msg = SimMessage(sender_id=1, recipient_id=2, content=42, sent_at=3)
        assert msg.sender_id == 1
        assert msg.recipient_id == 2
        assert msg.content == 42
        assert msg.sent_at == 3

    def test_immutable(self):
        msg = SimMessage(0, 1, "x", 0)
        with pytest.raises(Exception):
            msg.content = "y"  # type: ignore[misc]

    def test_hashable(self):
        msg = SimMessage(0, 1, "x", 0)
        s = {msg}
        assert msg in s


# ===========================================================================
# Network topology tests
# ===========================================================================

class TestNetworkRing:
    def test_ring_each_node_has_two_neighbors(self):
        rng = random.Random(0)
        net = Network.ring(list(range(6)), rng=rng)
        for i in range(6):
            assert len(net.neighbors(i)) == 2

    def test_ring_neighbors_are_adjacent(self):
        rng = random.Random(0)
        ids = list(range(5))
        net = Network.ring(ids, rng=rng)
        for i, aid in enumerate(ids):
            expected = {ids[(i - 1) % 5], ids[(i + 1) % 5]}
            assert net.neighbors(aid) == expected

    def test_ring_wraps_around(self):
        rng = random.Random(0)
        net = Network.ring([0, 1, 2], rng=rng)
        assert 2 in net.neighbors(0)
        assert 0 in net.neighbors(2)

    def test_ring_directed_one_neighbor(self):
        rng = random.Random(0)
        net = Network.ring(list(range(4)), rng=rng, directed=True)
        for i in range(4):
            assert len(net.neighbors(i)) == 1

    def test_ring_requires_at_least_two(self):
        with pytest.raises(ValueError):
            Network.ring([0], rng=random.Random(0))


class TestNetworkStar:
    def test_star_center_reaches_all_leaves(self):
        rng = random.Random(0)
        net = Network.star(center_id=0, agent_ids=[0, 1, 2, 3], rng=rng)
        assert net.neighbors(0) == {1, 2, 3}

    def test_star_leaves_only_reach_center(self):
        rng = random.Random(0)
        net = Network.star(center_id=0, agent_ids=[0, 1, 2, 3], rng=rng)
        for leaf in [1, 2, 3]:
            assert net.neighbors(leaf) == {0}

    def test_star_center_must_be_in_agent_ids(self):
        with pytest.raises(ValueError):
            Network.star(center_id=99, agent_ids=[0, 1, 2], rng=random.Random(0))


class TestNetworkFullyConnected:
    def test_fully_connected_n_minus_1_neighbors(self):
        rng = random.Random(0)
        net = Network.fully_connected(list(range(5)), rng=rng)
        for i in range(5):
            assert len(net.neighbors(i)) == 4
            assert i not in net.neighbors(i)


class TestNetworkFromEdges:
    def test_from_edges_bidirectional(self):
        rng = random.Random(0)
        net = Network.from_edges([(0, 1), (1, 2)], rng=rng, bidirectional=True)
        assert 1 in net.neighbors(0)
        assert 0 in net.neighbors(1)
        assert 2 in net.neighbors(1)
        assert 1 in net.neighbors(2)

    def test_from_edges_directed(self):
        rng = random.Random(0)
        net = Network.from_edges([(0, 1)], rng=rng, bidirectional=False)
        assert 1 in net.neighbors(0)
        assert 0 not in net.neighbors(1)


# ===========================================================================
# Network delivery tests
# ===========================================================================

class TestNetworkDelivery:
    def _two_agent_net(self, seed=0, loss_rate=0.0):
        rng = random.Random(seed)
        return Network.fully_connected([0, 1], rng=rng, loss_rate=loss_rate)

    def test_send_valid_queues_message(self):
        net = self._two_agent_net()
        assert net.queue_size() == 0
        net.send(SimMessage(0, 1, "hi", 0))
        assert net.queue_size() == 1

    def test_send_to_non_neighbor_drops_silently(self):
        rng = random.Random(0)
        net4 = Network.ring([0, 1, 2, 3], rng=rng)
        # In a 4-ring: 0's neighbors are 1 and 3; 0 cannot reach 2 — silently dropped
        net4.send(SimMessage(0, 2, "x", 0))
        assert net4.queue_size() == 0

    def test_deliver_clears_queue(self):
        net = self._two_agent_net()
        net.send(SimMessage(0, 1, "msg", 0))
        net.deliver()
        assert net.queue_size() == 0

    def test_deliver_puts_message_in_recipient_inbox(self):
        net = self._two_agent_net()
        msg = SimMessage(0, 1, "hello", 0)
        net.send(msg)
        result = net.deliver()
        assert 1 in result.inboxes
        assert msg in result.inboxes[1]

    def test_deliver_returns_correct_counts(self):
        net = self._two_agent_net()
        net.send(SimMessage(0, 1, "a", 0))
        net.send(SimMessage(1, 0, "b", 0))
        result = net.deliver()
        assert result.delivered == 2
        assert result.dropped == 0

    def test_messages_not_delivered_same_round_as_sent(self):
        """Messages sent this round should NOT appear until deliver() is called."""
        net = self._two_agent_net()
        net.send(SimMessage(0, 1, "x", 0))
        # before deliver: no inbox for agent 1
        result1 = net.deliver()
        # after deliver: one message
        assert result1.delivered == 1
        # queue is now empty
        result2 = net.deliver()
        assert result2.delivered == 0

    def test_no_loss_delivers_all(self):
        net = self._two_agent_net(loss_rate=0.0)
        for _ in range(10):
            net.send(SimMessage(0, 1, "x", 0))
        result = net.deliver()
        assert result.delivered == 10
        assert result.dropped == 0

    def test_full_loss_drops_all(self):
        net = self._two_agent_net(loss_rate=1.0)
        for _ in range(10):
            net.send(SimMessage(0, 1, "x", 0))
        result = net.deliver()
        assert result.delivered == 0
        assert result.dropped == 10

    def test_partial_loss_statistics(self):
        """With loss_rate=0.5 over many messages, roughly half are dropped."""
        rng = random.Random(12345)
        net = Network.fully_connected([0, 1], rng=rng, loss_rate=0.5)
        for _ in range(1000):
            net.send(SimMessage(0, 1, "x", 0))
        result = net.deliver()
        # Expect roughly 500 delivered; allow ±20% margin
        assert 300 < result.delivered < 700


# ===========================================================================
# Simulator tests
# ===========================================================================

class TestSimulator:
    def test_initial_clock_zero(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy), 1: RuleAgent(1, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        assert sim.clock == 0

    def test_step_advances_clock(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy), 1: RuleAgent(1, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        sim.step()
        assert sim.clock == 1
        sim.step()
        assert sim.clock == 2

    def test_run_returns_correct_number_of_steps(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy), 1: RuleAgent(1, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        history = sim.run(rounds=5)
        assert len(history) == 5

    def test_step_returns_simulation_step(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy), 1: RuleAgent(1, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        result = sim.step()
        assert isinstance(result, SimulationStep)

    def test_step_round_number_matches_clock(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy), 1: RuleAgent(1, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        step = sim.step()
        assert step.round == 1

    def test_history_accumulated(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy), 1: RuleAgent(1, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        sim.run(7)
        assert len(sim.history) == 7

    def test_agent_states_in_step(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {
            0: RuleAgent(0, _null_policy, _state={"v": 10}),
            1: RuleAgent(1, _null_policy, _state={"v": 20}),
        }
        sim = Simulator(agents=agents, network=net)
        step = sim.step()
        assert step.agent_states[0]["v"] == 10
        assert step.agent_states[1]["v"] == 20

    def test_messages_sent_counted(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)

        def always_send(agent: RuleAgent) -> list[SimMessage]:
            neighbor = [aid for aid in [0, 1] if aid != agent.agent_id][0]
            return [SimMessage(agent.agent_id, neighbor, "ping", 0)]

        agents = {0: RuleAgent(0, always_send), 1: RuleAgent(1, always_send)}
        sim = Simulator(agents=agents, network=net)
        step = sim.step()
        # Both agents send one message each
        assert step.messages_sent == 2

    def test_run_zero_rounds(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy), 1: RuleAgent(1, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        history = sim.run(rounds=0)
        assert history == []
        assert sim.clock == 0

    def test_run_negative_rounds_raises(self):
        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng=rng)
        agents = {0: RuleAgent(0, _null_policy)}
        sim = Simulator(agents=agents, network=net)
        with pytest.raises(ValueError):
            sim.run(rounds=-1)


# ===========================================================================
# Information propagation test
# ===========================================================================

class TestInformationPropagation:
    def test_max_propagates_around_ring(self):
        """After n-1 rounds, all agents on a ring should know the global max."""
        n = 10
        values = list(range(n))  # agent i starts with value i
        global_max = max(values)

        sim = _make_ring_sim(n=n, initial_values=values, seed=42)

        # In a ring of n nodes, information takes at most ceil(n/2) hops.
        # We run n rounds to be safe.
        sim.run(rounds=n)

        for agent_id, agent in sim.agents.items():
            assert agent.state["max"] == global_max, (
                f"Agent {agent_id} has max={agent.state['max']}, expected {global_max}"
            )

    def test_propagation_requires_enough_rounds(self):
        """After only 1 round on a large ring, not all agents know the global max."""
        n = 10
        # Put the max value only at agent 0; agents far away won't see it yet
        values = [100] + [0] * (n - 1)

        sim = _make_ring_sim(n=n, initial_values=values, seed=42)
        sim.run(rounds=1)

        # Agent 5 is furthest from 0 on a 10-ring; should not know max yet
        assert sim.agents[5].state["max"] != 100

    def test_propagation_with_packet_loss_slower(self):
        """With loss_rate=0.5, propagation still eventually completes with enough rounds."""
        n = 6
        values = [99] + [0] * (n - 1)

        # More rounds than needed for lossless to account for drops
        sim = _make_ring_sim(n=n, initial_values=values, seed=0, loss_rate=0.5)
        sim.run(rounds=40)

        # Even with 50% loss, after 40 rounds all agents should know 99
        for agent_id, agent in sim.agents.items():
            assert agent.state["max"] == 99, (
                f"Agent {agent_id} still has max={agent.state['max']} after lossy propagation"
            )


# ===========================================================================
# Determinism test
# ===========================================================================

class TestDeterminism:
    def _run_and_collect(self, seed: int) -> list[dict]:
        n = 6
        values = [i * 10 for i in range(n)]
        sim = _make_ring_sim(n=n, initial_values=values, seed=seed, loss_rate=0.3)
        history = sim.run(rounds=10)
        return [
            {aid: s["max"] for aid, s in step.agent_states.items()}
            for step in history
        ]

    def test_same_seed_same_trace(self):
        trace1 = self._run_and_collect(seed=7)
        trace2 = self._run_and_collect(seed=7)
        assert trace1 == trace2

    def test_different_seed_different_trace(self):
        trace1 = self._run_and_collect(seed=1)
        trace2 = self._run_and_collect(seed=2)
        # With 30% packet loss the traces should diverge (extremely unlikely to match)
        assert trace1 != trace2
