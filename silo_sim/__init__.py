"""Multi-agent simulator core.

Provides a clean Agent → Network → Agent abstraction that is completely
independent of SILO-BENCH's benchmark runner.  The existing engine.py,
batch_run.py, and protocol tool modules are untouched.

Public API::

    from silo_sim import SimMessage, RuleAgent, LLMAgent, SiloBenchAgent
    from silo_sim import Network, Simulator, SimulationStep
    from silo_sim import EpochResult
    from silo_sim import VerificationResult, Verifier
    from silo_sim import PassVerifier, FnVerifier
    from silo_sim import AnswerVerifier, ConsistencyVerifier
    from silo_sim import make_openai_fn, make_mock_fn
    from silo_sim import run_silo_case, SiloResult
"""

from silo_sim.adapters import make_mock_fn, make_openai_fn
from silo_sim.agent import BaseAgent, RuleAgent, SimMessage
from silo_sim.llm_agent import LLMAgent
from silo_sim.network import Network
from silo_sim.silo_agent import SiloBenchAgent
from silo_sim.silo_runner import SiloResult, run_silo_case
from silo_sim.simulator import EpochResult, SimulationStep, Simulator
from silo_sim.verification import (
    AnswerVerifier,
    ConsistencyVerifier,
    FnVerifier,
    PassVerifier,
    VerificationResult,
    Verifier,
)

__all__ = [
    # Agents
    "BaseAgent",
    "RuleAgent",
    "LLMAgent",
    "SiloBenchAgent",
    # Messages
    "SimMessage",
    # Network
    "Network",
    # Simulator
    "Simulator",
    "SimulationStep",
    # Epochs
    "EpochResult",
    # Verification layer
    "VerificationResult",
    "Verifier",
    "PassVerifier",
    "FnVerifier",
    "AnswerVerifier",
    "ConsistencyVerifier",
    # Adapter factories
    "make_openai_fn",
    "make_mock_fn",
    # SILO-BENCH runner
    "run_silo_case",
    "SiloResult",
]
