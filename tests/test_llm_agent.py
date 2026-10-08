"""Tests for LLMAgent and adapter factories.

All tests use make_mock_fn — no real LLM calls are made.
"""

from __future__ import annotations

import json
import random

import pytest

from silo_sim.adapters import make_mock_fn
from silo_sim.agent import BaseAgent, SimMessage
from silo_sim.llm_agent import LLMAgent
from silo_sim.network import Network
from silo_sim.simulator import Simulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(sender: int, recipient: int, content, sent_at: int = 0) -> SimMessage:
    return SimMessage(sender_id=sender, recipient_id=recipient,
                      content=content, sent_at=sent_at)


def _agent(responses: list[str], agent_id: int = 0, loop: bool = True, **kw) -> LLMAgent:
    return LLMAgent(agent_id=agent_id, llm_fn=make_mock_fn(responses, loop=loop), **kw)


def _envelope(send=None, remember=None) -> str:
    d: dict = {}
    if send is not None:
        d["send"] = send
    if remember is not None:
        d["remember"] = remember
    return json.dumps(d)


# ---------------------------------------------------------------------------
# TestLLMAgentProtocol
# ---------------------------------------------------------------------------

class TestLLMAgentProtocol:
    def test_satisfies_base_agent_protocol(self):
        agent = _agent(['{}'])
        assert isinstance(agent, BaseAgent)

    def test_has_agent_id(self):
        agent = _agent(['{}'], agent_id=7)
        assert agent.agent_id == 7

    def test_state_returns_dict(self):
        agent = _agent(['{}'])
        assert isinstance(agent.state, dict)

    def test_observe_stores_inbox(self):
        agent = _agent(['{}'])
        msgs = [_msg(1, 0, "hello")]
        agent.observe(msgs)
        assert agent._inbox == msgs

    def test_decide_clears_inbox(self):
        agent = _agent(['{}'])
        agent.observe([_msg(1, 0, "x")])
        agent.decide()
        assert agent._inbox == []

    def test_decide_returns_list(self):
        agent = _agent(['{}'])
        result = agent.decide()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestResponseParsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_empty_envelope_no_messages(self):
        agent = _agent(['{}'])
        assert agent.decide() == []

    def test_send_produces_sim_messages(self):
        resp = _envelope(send=[{"to": 2, "content": "hi"}])
        agent = _agent([resp])
        out = agent.decide()
        assert len(out) == 1
        assert out[0].sender_id == 0
        assert out[0].recipient_id == 2
        assert out[0].content == "hi"

    def test_multiple_sends(self):
        resp = _envelope(send=[
            {"to": 1, "content": "a"},
            {"to": 2, "content": "b"},
            {"to": 3, "content": "c"},
        ])
        agent = _agent([resp])
        out = agent.decide()
        assert len(out) == 3
        assert [m.recipient_id for m in out] == [1, 2, 3]

    def test_remember_updates_state(self):
        resp = _envelope(remember={"key": "value", "count": 42})
        agent = _agent([resp])
        agent.decide()
        assert agent.state["key"] == "value"
        assert agent.state["count"] == 42

    def test_send_and_remember_together(self):
        resp = _envelope(
            send=[{"to": 1, "content": "ping"}],
            remember={"pinged": True},
        )
        agent = _agent([resp])
        out = agent.decide()
        assert len(out) == 1
        assert agent.state["pinged"] is True

    def test_missing_send_field_defaults_to_empty(self):
        resp = _envelope(remember={"x": 1})
        agent = _agent([resp])
        assert agent.decide() == []

    def test_missing_remember_field_leaves_state_unchanged(self):
        resp = _envelope(send=[{"to": 1, "content": "hi"}])
        agent = _agent([resp], agent_id=0)
        agent._state["existing"] = "preserved"
        agent.decide()
        assert agent.state["existing"] == "preserved"

    def test_malformed_json_returns_empty_and_records_error(self):
        agent = _agent(["not json at all"])
        out = agent.decide()
        assert out == []
        assert "_last_parse_error" in agent.state

    def test_non_object_json_records_error(self):
        agent = _agent(['["list", "not", "object"]'])
        out = agent.decide()
        assert out == []
        assert "_last_parse_error" in agent.state

    def test_error_cleared_on_successful_parse(self):
        agent = _agent(["bad json", _envelope(remember={"ok": True})], loop=False)
        agent.decide()   # sets _last_parse_error
        assert "_last_parse_error" in agent.state
        agent.decide()   # successful — should clear it
        assert "_last_parse_error" not in agent.state

    def test_markdown_fenced_json_parsed(self):
        fenced = "```json\n" + _envelope(send=[{"to": 1, "content": "x"}]) + "\n```"
        agent = _agent([fenced])
        out = agent.decide()
        assert len(out) == 1

    def test_malformed_send_item_skipped(self):
        # item missing 'to' key
        resp = json.dumps({"send": [{"content": "oops"}, {"to": 2, "content": "ok"}]})
        agent = _agent([resp])
        out = agent.decide()
        assert len(out) == 1
        assert out[0].recipient_id == 2

    def test_non_integer_to_skipped(self):
        resp = json.dumps({"send": [{"to": "bad", "content": "x"}]})
        agent = _agent([resp])
        assert agent.decide() == []

    def test_content_can_be_any_json_value(self):
        payload = {"nested": [1, 2, 3]}
        resp = json.dumps({"send": [{"to": 1, "content": payload}]})
        agent = _agent([resp])
        out = agent.decide()
        assert out[0].content == payload


# ---------------------------------------------------------------------------
# TestConversationHistory
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_history_empty_initially(self):
        agent = _agent(['{}'])
        assert agent.history == []

    def test_history_grows_after_decide(self):
        agent = _agent(['{}', '{}'], loop=False)
        agent.decide()
        assert len(agent.history) == 2  # user + assistant

    def test_history_accumulates_across_rounds(self):
        agent = _agent(['{}', '{}'], loop=False)
        agent.decide()
        agent.decide()
        assert len(agent.history) == 4

    def test_history_disabled_does_not_accumulate(self):
        agent = _agent(['{}', '{}'], loop=False, keep_history=False)
        agent.decide()
        agent.decide()
        assert agent.history == []

    def test_reset_history_clears_without_touching_state(self):
        agent = _agent(['{}'])
        agent._state["x"] = 1
        agent.decide()
        agent.reset_history()
        assert agent.history == []
        assert agent.state["x"] == 1

    def test_history_roles_alternate_user_assistant(self):
        agent = _agent(['{}'])
        agent.decide()
        assert agent.history[0]["role"] == "user"
        assert agent.history[1]["role"] == "assistant"

    def test_user_message_contains_inbox_json(self):
        agent = _agent(['{}'])
        agent.observe([_msg(1, 0, "hello")])
        agent.decide()
        user_turn = json.loads(agent.history[0]["content"])
        assert "inbox" in user_turn
        assert user_turn["inbox"][0]["from"] == 1
        assert user_turn["inbox"][0]["content"] == "hello"

    def test_prompt_includes_history_when_enabled(self):
        captured = []

        def spy_fn(messages):
            captured.append(list(messages))
            return '{}'

        agent = LLMAgent(agent_id=0, llm_fn=spy_fn, keep_history=True)
        agent.decide()  # round 1 — history is empty before this call
        agent.decide()  # round 2 — history should be present

        # Round 2 prompt: system + user1 + assistant1 + user2
        r2_prompt = captured[1]
        assert r2_prompt[0]["role"] == "system"
        assert r2_prompt[1]["role"] == "user"
        assert r2_prompt[2]["role"] == "assistant"
        assert r2_prompt[3]["role"] == "user"

    def test_prompt_excludes_history_when_disabled(self):
        captured = []

        def spy_fn(messages):
            captured.append(list(messages))
            return '{}'

        agent = LLMAgent(agent_id=0, llm_fn=spy_fn, keep_history=False)
        agent.decide()
        agent.decide()

        # Every call: system + current user only (2 messages)
        assert len(captured[1]) == 2


# ---------------------------------------------------------------------------
# TestMockFnAdapter
# ---------------------------------------------------------------------------

class TestMockFnAdapter:
    def test_cycles_by_default(self):
        fn = make_mock_fn(["a", "b"])
        assert fn([]) == "a"
        assert fn([]) == "b"
        assert fn([]) == "a"  # cycles

    def test_loop_false_exhausts(self):
        fn = make_mock_fn(["only"], loop=False)
        fn([])
        with pytest.raises(StopIteration):
            fn([])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            make_mock_fn([])

    def test_single_response_loops(self):
        fn = make_mock_fn(['{}'])
        for _ in range(10):
            assert fn([]) == '{}'


# ---------------------------------------------------------------------------
# TestLLMAgentInSimulator — integration
# ---------------------------------------------------------------------------

class TestLLMAgentInSimulator:
    def _ring_sim(self, n: int, responses: list[str]) -> Simulator:
        rng = random.Random(0)
        net = Network.ring(list(range(n)), rng)
        agents = {
            i: LLMAgent(agent_id=i, llm_fn=make_mock_fn(responses))
            for i in range(n)
        }
        return Simulator(agents=agents, network=net)

    def test_simulator_runs_with_llm_agents(self):
        sim = self._ring_sim(3, ['{}'])
        steps = sim.run(5)
        assert len(steps) == 5

    def test_llm_agent_messages_delivered_next_round(self):
        # Agent 0 sends to agent 1 every round; agent 1 sends nothing.
        send_to_1 = json.dumps({"send": [{"to": 1, "content": "ping"}]})
        silence = json.dumps({})

        rng = random.Random(0)
        net = Network.ring([0, 1, 2], rng)  # 0↔1↔2↔0
        agents = {
            0: LLMAgent(agent_id=0, llm_fn=make_mock_fn([send_to_1])),
            1: LLMAgent(agent_id=1, llm_fn=make_mock_fn([silence])),
            2: LLMAgent(agent_id=2, llm_fn=make_mock_fn([silence])),
        }
        sim = Simulator(agents=agents, network=net)

        step0 = sim.step()
        # Round 0: agent 0 decides to send — 1 message queued
        assert step0.messages_sent == 1
        assert step0.messages_delivered == 0  # not yet delivered

        step1 = sim.step()
        # Round 1: queued message delivered to agent 1
        assert step1.messages_delivered == 1

    def test_state_persists_across_rounds(self):
        # Each round the agent remembers a counter incremented by the mock.
        responses = [
            json.dumps({"remember": {"round": i}}) for i in range(5)
        ]
        agent = LLMAgent(agent_id=0, llm_fn=make_mock_fn(responses, loop=False))
        for i in range(5):
            agent.decide()
            assert agent.state["round"] == i

    def test_mixed_rule_and_llm_agents(self):
        """Simulator handles a mix of RuleAgent and LLMAgent."""
        from silo_sim.agent import RuleAgent

        rng = random.Random(1)
        net = Network.ring([0, 1], rng)

        rule = RuleAgent(
            agent_id=0,
            policy_fn=lambda a: [SimMessage(0, 1, "rule", 0)],
        )
        llm = LLMAgent(agent_id=1, llm_fn=make_mock_fn(['{}']))

        sim = Simulator(agents={0: rule, 1: llm}, network=net)
        steps = sim.run(3)
        assert len(steps) == 3
        # After round 0 the rule agent sends a message; round 1 delivers it.
        assert steps[1].messages_delivered >= 1
