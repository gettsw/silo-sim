"""Verification layer for epoch-based simulation.

A Verifier sits between epochs: it inspects every agent's state and the
step history for the just-completed epoch, then returns a VerificationResult.

Implementations can be as simple as a lambda wrapped in FnVerifier, or
as complex as a GNS calibration check.  The simulation core (simulator.py)
never imports a concrete verifier — only the Verifier Protocol and
VerificationResult, which keep the coupling one-directional.

Built-in implementations
------------------------
PassVerifier        — always passes; useful as a smoke-test stand-in
FnVerifier          — wraps any callable as a Verifier

Built-in stubs (bodies raise NotImplementedError)
-------------------------------------------------
AnswerVerifier      — checks per-agent submission against expected output
ConsistencyVerifier — checks that agent beliefs are mutually consistent

Planned (not yet stubbed)
-------------------------
GNSVerifier         — validates GNN scoring confidence against ground truth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from silo_sim.simulator import SimulationStep


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one verification pass.

    Attributes
    ----------
    epoch:   Which epoch this covers (1-indexed).
    passed:  True if verification criteria were met.
    score:   Summary scalar in [0, 1]; 1.0 = fully correct / all pass.
    details: Verifier-specific diagnostics (open-ended, not inspected by
             the simulation core).
    """

    epoch: int
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")


# ---------------------------------------------------------------------------
# Verifier Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Verifier(Protocol):
    """Duck-typed interface for a pluggable verification layer.

    A Verifier is called once per epoch, after all rounds in that epoch
    complete and before the next epoch begins.  It receives:

        epoch         — which epoch just finished (1-indexed)
        agent_states  — {agent_id: state_dict} snapshot at epoch end
        steps         — list[SimulationStep] for the epoch's rounds

    The Simulator calls verifier.verify(...) and attaches the result to
    the EpochResult without interpreting it — concrete logic lives in
    the implementation, not in the simulation core.
    """

    def verify(
        self,
        epoch: int,
        agent_states: dict[int, Any],
        steps: list["SimulationStep"],
    ) -> VerificationResult:
        """Return a VerificationResult for this epoch."""
        ...


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------

class PassVerifier:
    """Trivial verifier that always passes.

    Useful as a placeholder when you want the epoch structure in place
    but have not yet wired up real verification logic.
    """

    def verify(
        self,
        epoch: int,
        agent_states: dict[int, Any],
        steps: list["SimulationStep"],
    ) -> VerificationResult:
        return VerificationResult(epoch=epoch, passed=True, score=1.0)


class FnVerifier:
    """Wrap an arbitrary callable as a Verifier.

    Parameters
    ----------
    fn:
        Called with (epoch, agent_states, steps); must return a
        VerificationResult.  This is the escape hatch for one-off
        verification logic without subclassing.

    Example::

        def my_check(epoch, states, steps):
            all_done = all("submission" in s for s in states.values())
            return VerificationResult(epoch=epoch, passed=all_done,
                                      score=1.0 if all_done else 0.0)

        verifier = FnVerifier(my_check)
    """

    def __init__(
        self,
        fn: Callable[
            [int, dict[int, Any], list["SimulationStep"]],
            VerificationResult,
        ],
    ) -> None:
        self._fn = fn

    def verify(
        self,
        epoch: int,
        agent_states: dict[int, Any],
        steps: list["SimulationStep"],
    ) -> VerificationResult:
        return self._fn(epoch, agent_states, steps)


# ---------------------------------------------------------------------------
# Stubs — architecture in place; bodies to be implemented
# ---------------------------------------------------------------------------

class AnswerVerifier:
    """Check per-agent submissions against expected benchmark output.

    Parameters
    ----------
    expected_output:
        The ``expected_output`` dict from a SILO-BENCH benchmark JSON.
    paradigm_level:
        "I", "II", or "III" — controls partial-correctness scoring.
    """

    _PARADIGM_LEVEL = {"Paradigm I": "I", "Paradigm II": "II", "Paradigm III": "III"}

    def __init__(
        self,
        expected_output: dict[str, Any],
        paradigm_level: str = "I",
    ) -> None:
        self.expected_output = expected_output
        # Accept either "I" or "Paradigm I"
        self.level = self._PARADIGM_LEVEL.get(paradigm_level, paradigm_level)

    def verify(
        self,
        epoch: int,
        agent_states: dict[int, Any],
        steps: list["SimulationStep"],
    ) -> VerificationResult:
        from silo_sim.utils.metrics import compute_partial_correctness, compute_success_rate

        submissions = [
            {"agent_id": aid, "answer": state["submission"]["answer"]}
            for aid, state in agent_states.items()
            if state.get("submission") is not None
        ]
        n_total = len(agent_states)
        S = compute_success_rate(submissions, self.expected_output) if submissions else 0.0
        P = compute_partial_correctness(submissions, self.expected_output, self.level) if submissions else 0.0
        return VerificationResult(
            epoch=epoch,
            passed=S == 1.0 and len(submissions) == n_total,
            score=P,
            details={
                "S": S,
                "P": P,
                "submitted": len(submissions),
                "total": n_total,
                "submissions": submissions,
            },
        )


class ConsistencyVerifier:
    """Check that all submitting agents converged on the same answer.

    Does not require knowledge of the ground truth — just checks unanimity.
    Pair with AnswerVerifier to separate "agreed but wrong" from "disagreed".
    """

    def verify(
        self,
        epoch: int,
        agent_states: dict[int, Any],
        steps: list["SimulationStep"],
    ) -> VerificationResult:
        answers = [
            state["submission"]["answer"]
            for state in agent_states.values()
            if state.get("submission") is not None
        ]
        if not answers:
            return VerificationResult(epoch=epoch, passed=False, score=0.0,
                                      details={"submitted": 0})
        unanimous = len(set(str(a) for a in answers)) == 1
        score = 1.0 if unanimous else len(answers) / len(agent_states)
        return VerificationResult(
            epoch=epoch,
            passed=unanimous and len(answers) == len(agent_states),
            score=score,
            details={"unanimous": unanimous, "answers": answers,
                     "submitted": len(answers), "total": len(agent_states)},
        )
