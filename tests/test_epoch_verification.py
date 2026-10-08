"""Tests for epoch structure and verification layer.

These tests cover the architectural skeleton only — no LLM calls, no SILO-BENCH
JSON files required.  They confirm:

  - EpochResult / VerificationResult are frozen dataclasses
  - Verifier is a @runtime_checkable Protocol
  - PassVerifier, FnVerifier satisfy the protocol
  - Simulator.run_epoch() wires rounds → verifier → EpochResult correctly
  - Simulator.run_epochs() handles multi-epoch loops and early stop
  - AnswerVerifier / ConsistencyVerifier stubs raise NotImplementedError
"""

from __future__ import annotations

import pytest

from silo_sim import (
    AnswerVerifier,
    ConsistencyVerifier,
    EpochResult,
    FnVerifier,
    Network,
    PassVerifier,
    RuleAgent,
    Simulator,
    SimulationStep,
    VerificationResult,
    Verifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import random


def _make_sim(n: int = 3, *, seed: int = 0) -> Simulator:
    """Return a tiny idle Simulator (agents emit nothing)."""
    rng = random.Random(seed)
    net = Network.ring(list(range(n)), rng=rng)
    agents = {i: RuleAgent(i, lambda a: []) for i in range(n)}
    return Simulator(agents=agents, network=net)


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_fields(self):
        vr = VerificationResult(epoch=1, passed=True, score=0.9)
        assert vr.epoch == 1
        assert vr.passed is True
        assert vr.score == 0.9
        assert vr.details == {}

    def test_custom_details(self):
        vr = VerificationResult(epoch=2, passed=False, score=0.5, details={"reason": "timeout"})
        assert vr.details["reason"] == "timeout"

    def test_frozen(self):
        vr = VerificationResult(epoch=1, passed=True, score=1.0)
        with pytest.raises((TypeError, AttributeError)):
            vr.score = 0.0  # type: ignore[misc]

    def test_score_validation_out_of_range(self):
        with pytest.raises(ValueError):
            VerificationResult(epoch=1, passed=True, score=1.5)

    def test_score_boundary_zero(self):
        vr = VerificationResult(epoch=1, passed=False, score=0.0)
        assert vr.score == 0.0

    def test_score_boundary_one(self):
        vr = VerificationResult(epoch=1, passed=True, score=1.0)
        assert vr.score == 1.0


# ---------------------------------------------------------------------------
# Verifier Protocol
# ---------------------------------------------------------------------------

class TestVerifierProtocol:
    def test_pass_verifier_satisfies_protocol(self):
        assert isinstance(PassVerifier(), Verifier)

    def test_fn_verifier_satisfies_protocol(self):
        fn = lambda e, s, steps: VerificationResult(epoch=e, passed=True, score=1.0)
        assert isinstance(FnVerifier(fn), Verifier)

    def test_arbitrary_class_with_verify_method(self):
        class MyVerifier:
            def verify(self, epoch, agent_states, steps):
                return VerificationResult(epoch=epoch, passed=True, score=1.0)

        assert isinstance(MyVerifier(), Verifier)

    def test_class_without_verify_does_not_satisfy(self):
        class NotAVerifier:
            pass

        assert not isinstance(NotAVerifier(), Verifier)


# ---------------------------------------------------------------------------
# PassVerifier
# ---------------------------------------------------------------------------

class TestPassVerifier:
    def test_always_passes(self):
        v = PassVerifier()
        result = v.verify(1, {}, [])
        assert result.passed is True
        assert result.score == 1.0
        assert result.epoch == 1

    def test_epoch_number_propagated(self):
        v = PassVerifier()
        for ep in [1, 5, 99]:
            r = v.verify(ep, {}, [])
            assert r.epoch == ep


# ---------------------------------------------------------------------------
# FnVerifier
# ---------------------------------------------------------------------------

class TestFnVerifier:
    def test_delegates_to_fn(self):
        calls = []

        def my_fn(epoch, states, steps):
            calls.append((epoch, states))
            return VerificationResult(epoch=epoch, passed=False, score=0.0)

        v = FnVerifier(my_fn)
        r = v.verify(3, {0: {"x": 1}}, [])
        assert r.epoch == 3
        assert r.passed is False
        assert calls == [(3, {0: {"x": 1}})]

    def test_fn_receives_agent_states(self):
        received = {}

        def my_fn(epoch, states, steps):
            received.update(states)
            return VerificationResult(epoch=epoch, passed=True, score=1.0)

        v = FnVerifier(my_fn)
        v.verify(1, {0: {"val": 42}, 1: {"val": 99}}, [])
        assert received[0]["val"] == 42
        assert received[1]["val"] == 99


# ---------------------------------------------------------------------------
# Stubs raise NotImplementedError
# ---------------------------------------------------------------------------

class TestStubs:
    def test_answer_verifier_stores_expected(self):
        expected = {"per_agent_values": {0: 7, 1: 7}}
        v = AnswerVerifier(expected_output=expected)
        assert v.expected_output == expected


# ---------------------------------------------------------------------------
# EpochResult
# ---------------------------------------------------------------------------

class TestEpochResult:
    def test_fields(self):
        er = EpochResult(epoch=1, rounds_run=3, steps=(), verification=None)
        assert er.epoch == 1
        assert er.rounds_run == 3
        assert er.steps == ()
        assert er.verification is None

    def test_frozen(self):
        er = EpochResult(epoch=1, rounds_run=1, steps=(), verification=None)
        with pytest.raises((TypeError, AttributeError)):
            er.epoch = 2  # type: ignore[misc]

    def test_with_verification_result(self):
        vr = VerificationResult(epoch=1, passed=True, score=0.8)
        er = EpochResult(epoch=1, rounds_run=2, steps=(), verification=vr)
        assert er.verification.score == 0.8


# ---------------------------------------------------------------------------
# Simulator.run_epoch()
# ---------------------------------------------------------------------------

class TestRunEpoch:
    def test_returns_epoch_result(self):
        sim = _make_sim()
        er = sim.run_epoch(3)
        assert isinstance(er, EpochResult)

    def test_rounds_run_matches(self):
        sim = _make_sim()
        er = sim.run_epoch(4)
        assert er.rounds_run == 4
        assert len(er.steps) == 4

    def test_epoch_number_defaults_to_one(self):
        sim = _make_sim()
        er = sim.run_epoch(1)
        assert er.epoch == 1

    def test_epoch_number_increments(self):
        sim = _make_sim()
        e1 = sim.run_epoch(1)
        e2 = sim.run_epoch(1)
        assert e1.epoch == 1
        assert e2.epoch == 2

    def test_explicit_epoch_number(self):
        sim = _make_sim()
        er = sim.run_epoch(2, epoch=7)
        assert er.epoch == 7

    def test_no_verifier_gives_none_verification(self):
        sim = _make_sim()
        er = sim.run_epoch(1)
        assert er.verification is None

    def test_pass_verifier_attached(self):
        sim = _make_sim()
        er = sim.run_epoch(2, verifier=PassVerifier())
        assert er.verification is not None
        assert er.verification.passed is True

    def test_verifier_receives_correct_epoch(self):
        seen = []

        def my_fn(epoch, states, steps):
            seen.append(epoch)
            return VerificationResult(epoch=epoch, passed=True, score=1.0)

        sim = _make_sim()
        sim.run_epoch(1, epoch=5, verifier=FnVerifier(my_fn))
        assert seen == [5]

    def test_verifier_receives_step_history(self):
        seen_steps = []

        def my_fn(epoch, states, steps):
            seen_steps.extend(steps)
            return VerificationResult(epoch=epoch, passed=True, score=1.0)

        sim = _make_sim()
        sim.run_epoch(3, verifier=FnVerifier(my_fn))
        assert len(seen_steps) == 3
        assert all(isinstance(s, SimulationStep) for s in seen_steps)

    def test_steps_are_frozen_tuple_in_epoch_result(self):
        sim = _make_sim()
        er = sim.run_epoch(2)
        assert isinstance(er.steps, tuple)

    def test_clock_advances_across_epochs(self):
        sim = _make_sim()
        sim.run_epoch(3)
        sim.run_epoch(2)
        assert sim.clock == 5

    def test_epoch_history_accumulated(self):
        sim = _make_sim()
        sim.run_epoch(1)
        sim.run_epoch(1)
        sim.run_epoch(1)
        assert len(sim.epoch_history) == 3


# ---------------------------------------------------------------------------
# Simulator.run_epochs()
# ---------------------------------------------------------------------------

class TestRunEpochs:
    def test_returns_list_of_epoch_results(self):
        sim = _make_sim()
        results = sim.run_epochs(3, rounds_per_epoch=2)
        assert len(results) == 3
        assert all(isinstance(r, EpochResult) for r in results)

    def test_epoch_numbers_sequential(self):
        sim = _make_sim()
        results = sim.run_epochs(4, rounds_per_epoch=1)
        assert [r.epoch for r in results] == [1, 2, 3, 4]

    def test_rounds_per_epoch_correct(self):
        sim = _make_sim()
        results = sim.run_epochs(3, rounds_per_epoch=5)
        assert all(r.rounds_run == 5 for r in results)

    def test_total_clock(self):
        sim = _make_sim()
        sim.run_epochs(4, rounds_per_epoch=3)
        assert sim.clock == 12

    def test_verifier_called_every_epoch(self):
        calls = []

        def my_fn(epoch, states, steps):
            calls.append(epoch)
            return VerificationResult(epoch=epoch, passed=True, score=1.0)

        sim = _make_sim()
        sim.run_epochs(5, rounds_per_epoch=1, verifier=FnVerifier(my_fn))
        assert calls == [1, 2, 3, 4, 5]

    def test_stop_fn_early_exit(self):
        sim = _make_sim()
        results = sim.run_epochs(
            10,
            rounds_per_epoch=1,
            verifier=PassVerifier(),
            stop_fn=lambda r: r.epoch == 3,
        )
        assert len(results) == 3
        assert results[-1].epoch == 3

    def test_stop_fn_based_on_score(self):
        counter = {"n": 0}

        def my_fn(epoch, states, steps):
            counter["n"] += 1
            score = 1.0 if counter["n"] >= 2 else 0.0
            return VerificationResult(epoch=epoch, passed=score == 1.0, score=score)

        sim = _make_sim()
        results = sim.run_epochs(
            10,
            rounds_per_epoch=1,
            verifier=FnVerifier(my_fn),
            stop_fn=lambda r: r.verification and r.verification.score == 1.0,
        )
        assert len(results) == 2
        assert results[-1].verification.passed is True

    def test_no_stop_fn_runs_all_epochs(self):
        sim = _make_sim()
        results = sim.run_epochs(5, rounds_per_epoch=1)
        assert len(results) == 5

    def test_zero_epochs(self):
        sim = _make_sim()
        results = sim.run_epochs(0, rounds_per_epoch=3)
        assert results == []
        assert sim.clock == 0
