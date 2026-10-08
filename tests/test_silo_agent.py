"""Tests for SiloBenchAgent and run_silo_case.

All LLM calls use make_mock_fn — no API key required.
Integration tests load the real benchmark file benchmarks/I-01_n2.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from silo_sim.adapters import make_mock_fn
from silo_sim.agent import BaseAgent, SimMessage
from silo_sim.network import Network
from silo_sim.silo_agent import SiloBenchAgent
from silo_sim.silo_runner import SiloResult, run_silo_case
from silo_sim.simulator import Simulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"


def _agent(
    responses: list[str],
    protocol: str = "broadcast",
    agent_id: int = 0,
    num_agents: int = 3,
    loop: bool = True,
) -> SiloBenchAgent:
    return SiloBenchAgent(
        agent_id=agent_id,
        llm_fn=make_mock_fn(responses, loop=loop),
        protocol=protocol,
        system_prompt="system",
        task_prompt="task",
        num_agents=num_agents,
    )


def _xml_tool(tool: str, **params) -> str:
    param_xml = "".join(f"<{k}>{v}</{k}>" for k, v in params.items())
    return f"<tool_call><tool>{tool}</tool><parameters>{param_xml}</parameters></tool_call>"


# ---------------------------------------------------------------------------
# TestSiloBenchAgentProtocol
# ---------------------------------------------------------------------------

class TestSiloBenchAgentProtocol:
    def test_satisfies_base_agent_protocol(self):
        a = _agent([""])
        assert isinstance(a, BaseAgent)

    def test_agent_id(self):
        a = _agent([""], agent_id=5)
        assert a.agent_id == 5

    def test_state_returns_dict(self):
        a = _agent([""])
        assert isinstance(a.state, dict)

    def test_observe_stores_inbox(self):
        a = _agent([""])
        msgs = [SimMessage(1, 0, "hi", 0)]
        a.observe(msgs)
        assert a._inbox == msgs

    def test_decide_clears_inbox(self):
        a = _agent([""])
        a.observe([SimMessage(1, 0, "hi", 0)])
        a.decide()
        assert a._inbox == []

    def test_decide_returns_list(self):
        a = _agent([""])
        assert isinstance(a.decide(), list)

    def test_submitted_false_initially(self):
        a = _agent([""])
        assert a.submitted is False

    def test_submission_none_initially(self):
        a = _agent([""])
        assert a.submission is None


# ---------------------------------------------------------------------------
# TestToolDispatch
# ---------------------------------------------------------------------------

class TestToolDispatch:
    def test_send_message_produces_sim_message(self):
        resp = _xml_tool("send_message", target_id=2, content="hello")
        a = _agent([resp], protocol="msg")
        out = a.decide()
        assert len(out) == 1
        assert out[0].recipient_id == 2
        assert out[0].content == "hello"
        assert out[0].sender_id == 0

    def test_send_message_increments_messages_sent(self):
        resp = _xml_tool("send_message", target_id=1, content="x")
        a = _agent([resp], protocol="msg")
        a.decide()
        assert a._messages_sent == 1

    def test_broadcast_sends_to_all_others(self):
        resp = _xml_tool("broadcast_message", content="ping")
        a = _agent([resp], protocol="broadcast", agent_id=0, num_agents=4)
        out = a.decide()
        assert len(out) == 3  # 4 agents - self
        recipients = {m.recipient_id for m in out}
        assert recipients == {1, 2, 3}

    def test_broadcast_does_not_send_to_self(self):
        resp = _xml_tool("broadcast_message", content="x")
        a = _agent([resp], protocol="broadcast", agent_id=1, num_agents=3)
        out = a.decide()
        for m in out:
            assert m.recipient_id != 1

    def test_receive_messages_returns_inbox(self):
        captured_result = []

        def spy_fn(messages):
            # second call: we can inspect what was appended to history
            return _xml_tool("wait")

        a = SiloBenchAgent(
            agent_id=0,
            llm_fn=spy_fn,
            protocol="broadcast",
            system_prompt="s",
            task_prompt="t",
            num_agents=2,
        )
        a.observe([SimMessage(1, 0, "data", 0)])
        # Call with a mock that records inbox and then waits
        fn = make_mock_fn([_xml_tool("receive_messages") + _xml_tool("wait")])
        a2 = SiloBenchAgent(
            agent_id=0, llm_fn=fn, protocol="broadcast",
            system_prompt="s", task_prompt="t", num_agents=2,
        )
        a2.observe([SimMessage(1, 0, "hello", 0)])
        a2.decide()
        # Check tool result was added to history
        assert len(a2._history) == 2  # assistant + results
        results_turn = a2._history[1]["content"]
        assert "hello" in results_turn

    def test_list_agents_returns_all_ids(self):
        fn = make_mock_fn([_xml_tool("list_agents") + _xml_tool("wait")])
        a = SiloBenchAgent(
            agent_id=0, llm_fn=fn, protocol="broadcast",
            system_prompt="s", task_prompt="t", num_agents=3,
        )
        a.decide()
        results_turn = a._history[1]["content"]
        assert "0" in results_turn
        assert "1" in results_turn
        assert "2" in results_turn

    def test_wait_is_terminal(self):
        # wait + send_message: send_message should be ignored
        resp = _xml_tool("wait") + _xml_tool("send_message", target_id=1, content="x")
        a = _agent([resp], protocol="msg")
        out = a.decide()
        assert len(out) == 0  # wait came first, send ignored

    def test_submit_result_stores_submission(self):
        resp = _xml_tool("submit_result", answer=42)
        a = _agent([resp])
        a.decide()
        assert a.submitted is True
        assert a.submission == {"agent_id": 0, "answer": 42}

    def test_submit_result_is_terminal(self):
        resp = _xml_tool("submit_result", answer=1) + _xml_tool("broadcast_message", content="x")
        a = _agent([resp])
        out = a.decide()
        assert len(out) == 0  # broadcast after submit is ignored

    def test_after_submit_decide_returns_empty(self):
        resp = _xml_tool("submit_result", answer=1)
        a = _agent([resp, _xml_tool("broadcast_message", content="x")], loop=False)
        a.decide()           # submits
        out = a.decide()     # post-submit
        assert out == []

    def test_unknown_tool_skipped(self):
        resp = _xml_tool("fly_to_moon") + _xml_tool("wait")
        a = _agent([resp])
        out = a.decide()
        assert out == []

    def test_empty_response_no_crash(self):
        a = _agent([""])
        out = a.decide()
        assert out == []

    def test_send_message_sent_at_is_current_round(self):
        resp = _xml_tool("send_message", target_id=1, content="x")
        a = _agent([resp, resp], protocol="msg", loop=False)
        out0 = a.decide()
        out1 = a.decide()
        assert out0[0].sent_at == 0
        assert out1[0].sent_at == 1


# ---------------------------------------------------------------------------
# TestConversationHistory
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_history_grows_each_round(self):
        a = _agent([""])
        assert a._history == []
        a.decide()
        # assistant + tool results (empty results → only assistant added? No — check logic)
        # With empty response → no tool calls → no results → only assistant appended
        assert len(a._history) >= 1

    def test_prompt_includes_prior_history(self):
        captured = []

        def spy(messages):
            captured.append(list(messages))
            return _xml_tool("wait")

        a = SiloBenchAgent(
            agent_id=0, llm_fn=spy, protocol="broadcast",
            system_prompt="SYS", task_prompt="TASK", num_agents=2,
        )
        a.decide()  # round 0
        a.decide()  # round 1

        # Round 1 prompt: [system, task, assistant_r0, results_r0]
        r1_prompt = captured[1]
        assert r1_prompt[0]["content"] == "SYS"
        assert r1_prompt[1]["content"] == "TASK"
        assert r1_prompt[2]["role"] == "assistant"

    def test_reset_clears_history_and_submission(self):
        resp = _xml_tool("submit_result", answer=99)
        a = _agent([resp])
        a.decide()
        assert a.submitted
        a.reset()
        assert not a.submitted
        assert a._history == []
        assert a.submission is None


# ---------------------------------------------------------------------------
# TestTokenCounting
# ---------------------------------------------------------------------------

class TestTokenCounting:
    def test_default_token_counter_adds_to_output_tokens(self):
        resp = "word " * 10  # 10 words
        a = _agent([resp])
        a.decide()
        assert a._output_tokens > 0

    def test_custom_token_counter_used(self):
        a = SiloBenchAgent(
            agent_id=0,
            llm_fn=make_mock_fn(["hello world"]),
            protocol="broadcast",
            system_prompt="s",
            task_prompt="t",
            num_agents=2,
            token_counter=lambda r: 999,
        )
        a.decide()
        assert a._output_tokens == 999


# ---------------------------------------------------------------------------
# TestSiloBenchAgentInSimulator
# ---------------------------------------------------------------------------

class TestSiloBenchAgentInSimulator:
    """Full round-trip through the Simulator with SiloBenchAgents."""

    def _broadcast_sim(self, n: int, responses_per_agent: list[list[str]]) -> Simulator:
        rng = random.Random(0)
        net = Network.fully_connected(list(range(n)), rng)
        agents = {
            i: SiloBenchAgent(
                agent_id=i,
                llm_fn=make_mock_fn(responses_per_agent[i]),
                protocol="broadcast",
                system_prompt="sys",
                task_prompt=f"task for agent {i}",
                num_agents=n,
            )
            for i in range(n)
        }
        return Simulator(agents=agents, network=net)

    def test_simulator_runs_without_error(self):
        sim = self._broadcast_sim(3, [[""], [""], [""]])
        steps = sim.run(3)
        assert len(steps) == 3

    def test_broadcast_delivered_next_round(self):
        # Agent 0 broadcasts round 0; agents 1 and 2 should have it in inbox round 1.
        r0 = _xml_tool("broadcast_message", content="hello")
        silence = _xml_tool("wait")

        rng = random.Random(0)
        net = Network.fully_connected([0, 1, 2], rng)
        agents = {
            0: SiloBenchAgent(
                agent_id=0, llm_fn=make_mock_fn([r0, silence]),
                protocol="broadcast", system_prompt="s", task_prompt="t", num_agents=3,
            ),
            1: SiloBenchAgent(
                agent_id=1, llm_fn=make_mock_fn([silence]),
                protocol="broadcast", system_prompt="s", task_prompt="t", num_agents=3,
            ),
            2: SiloBenchAgent(
                agent_id=2, llm_fn=make_mock_fn([silence]),
                protocol="broadcast", system_prompt="s", task_prompt="t", num_agents=3,
            ),
        }
        sim = Simulator(agents=agents, network=net)
        step0 = sim.step()
        step1 = sim.step()
        assert step0.messages_sent == 2      # broadcast from agent 0 → agents 1, 2
        assert step1.messages_delivered == 2

    def test_two_round_global_max_broadcast(self):
        """
        Simulated Global Max with 2 agents:
        Round 0: each broadcasts its local max.
        Round 1: each receives the other's local max, computes global max, submits.
        """
        local_max_0 = 827
        local_max_1 = 780
        global_max = 827

        r0_agent0 = _xml_tool("broadcast_message", content=str(local_max_0))
        r0_agent1 = _xml_tool("broadcast_message", content=str(local_max_1))

        # Round 1: receive + compute + submit (we skip receive_messages in mock)
        r1_agent0 = _xml_tool("submit_result", answer=global_max)
        r1_agent1 = _xml_tool("submit_result", answer=global_max)

        rng = random.Random(0)
        net = Network.fully_connected([0, 1], rng)
        agents = {
            0: SiloBenchAgent(
                agent_id=0,
                llm_fn=make_mock_fn([r0_agent0, r1_agent0], loop=False),
                protocol="broadcast", system_prompt="s",
                task_prompt="t", num_agents=2,
            ),
            1: SiloBenchAgent(
                agent_id=1,
                llm_fn=make_mock_fn([r0_agent1, r1_agent1], loop=False),
                protocol="broadcast", system_prompt="s",
                task_prompt="t", num_agents=2,
            ),
        }
        sim = Simulator(agents=agents, network=net)

        for _ in range(5):
            sim.step()
            if all(a.submitted for a in agents.values()):
                break

        assert agents[0].submitted
        assert agents[1].submitted
        assert agents[0].submission["answer"] == global_max
        assert agents[1].submission["answer"] == global_max


# ---------------------------------------------------------------------------
# TestRunSiloCase
# ---------------------------------------------------------------------------

class TestRunSiloCase:
    """Integration tests for run_silo_case() using the real benchmark file."""

    CASE = BENCHMARK_DIR / "I-01_n2.json"

    def _mock_solve(self, answer: int = 827) -> list[str]:
        """LLM responses that broadcast local max then submit the answer."""
        return [
            _xml_tool("broadcast_message", content=str(answer)),
            _xml_tool("submit_result", answer=answer),
        ]

    def test_returns_silo_result(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=5,
        )
        assert isinstance(result, SiloResult)

    def test_case_id_correct(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=5,
        )
        assert result.case_id == "I-01"

    def test_protocol_recorded(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=5,
        )
        assert result.protocol == "broadcast"

    def test_perfect_solve_success_rate_one(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve(answer=827)),
            max_rounds=5,
        )
        assert result.S == pytest.approx(1.0)

    def test_wrong_answer_success_rate_zero(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve(answer=0)),
            max_rounds=5,
        )
        assert result.S == pytest.approx(0.0)

    def test_all_submitted_flag(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=5,
        )
        assert result.all_submitted is True

    def test_not_all_submitted_when_max_rounds_too_small(self):
        # Only 1 round: agents broadcast but never get to submit
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn([_xml_tool("broadcast_message", content="827")]),
            max_rounds=1,
        )
        assert result.all_submitted is False

    def test_submissions_count(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=5,
        )
        assert len(result.submissions) == 2  # 2 agents in I-01_n2

    def test_communication_density_positive(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=5,
        )
        assert result.D > 0

    def test_rounds_run_lte_max_rounds(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=10,
        )
        assert result.rounds_run <= 10

    def test_summary_string(self):
        result = run_silo_case(
            case_path=self.CASE,
            protocol="broadcast",
            llm_fn=make_mock_fn(self._mock_solve()),
            max_rounds=5,
        )
        s = result.summary()
        assert "I-01" in s
        assert "broadcast" in s
        assert "S=" in s

    def test_msg_protocol_works(self):
        # With msg protocol agent 0 sends to agent 1, agent 1 sends to agent 0
        round0 = _xml_tool("send_message", target_id=1, content="827")
        round1 = _xml_tool("submit_result", answer=827)
        result = run_silo_case(
            case_path=self.CASE,
            protocol="msg",
            llm_fn=make_mock_fn([round0, round1]),
            max_rounds=5,
        )
        assert isinstance(result, SiloResult)
        assert result.all_submitted

    def test_deterministic_with_same_seed(self):
        fn = make_mock_fn(self._mock_solve())
        r1 = run_silo_case(self.CASE, "broadcast", fn, seed=7)
        fn2 = make_mock_fn(self._mock_solve())
        r2 = run_silo_case(self.CASE, "broadcast", fn2, seed=7)
        assert r1.S == r2.S
        assert r1.D == r2.D
        assert r1.rounds_run == r2.rounds_run
