"""Agent interface and rule-based agent implementation.

Every agent in the simulator must satisfy the BaseAgent Protocol:

    observe(messages)  – receive this round's inbound messages
    decide()           – return outbound messages for this round

RuleAgent is the fast, synchronous, LLM-free implementation driven by a
plain Python callable (policy_fn).  An LLMAgent wrapping call_llm() will
follow in a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Message — the unit of inter-agent communication
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimMessage:
    """An immutable message passed between agents through the Network.

    Attributes:
        sender_id:    ID of the originating agent.
        recipient_id: ID of the intended recipient.
        content:      Arbitrary payload (int, str, dict, …).
        sent_at:      Simulation round in which the message was created.
        known:        Agent ids whose original data the sender had seen when
                      sending (upper bound on what this message can carry).
    """

    sender_id: int
    recipient_id: int
    content: Any
    sent_at: int
    known: frozenset = frozenset()


# ---------------------------------------------------------------------------
# BaseAgent — structural Protocol (duck-typed, no inheritance required)
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseAgent(Protocol):
    """Minimal interface every simulator agent must implement.

    The simulator calls these two methods once per round, in order:
        1. observe() — deliver inbound messages (from Network)
        2. decide()  — collect outbound messages (returned to Network)

    Agents must *not* mutate each other's state directly.
    """

    agent_id: int

    def observe(self, messages: list[SimMessage]) -> None:
        """Store/process messages delivered this round."""
        ...

    def decide(self) -> list[SimMessage]:
        """Produce outbound messages for this round."""
        ...

    @property
    def state(self) -> dict[str, Any]:
        """Read-only snapshot of the agent's local state."""
        ...


# ---------------------------------------------------------------------------
# RuleAgent — fast synthetic agent, no LLM
# ---------------------------------------------------------------------------

@dataclass
class RuleAgent:
    """A lightweight, rule-driven agent.

    The *policy_fn* is a plain Python callable::

        def my_policy(agent: RuleAgent) -> list[SimMessage]:
            ...
            return [SimMessage(agent.agent_id, neighbor, value, clock)]

    The agent exposes its internal ``_state`` dict through the ``state``
    property and its current-round inbox through the ``inbox`` property.
    Both are available inside policy_fn.

    Example — max-propagation policy on a ring::

        def max_policy(agent):
            for msg in agent.inbox:
                if msg.content > agent.state.get("max", float("-inf")):
                    agent.state["max"] = msg.content
            return [
                SimMessage(agent.agent_id, n, agent.state["max"], 0)
                for n in agent.state.get("neighbors", [])
            ]
    """

    agent_id: int
    policy_fn: Callable[[RuleAgent], list[SimMessage]]
    _state: dict[str, Any] = field(default_factory=dict)
    _inbox: list[SimMessage] = field(default_factory=list)

    # ------------------------------------------------------------------
    # BaseAgent protocol implementation
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """Mutable local state dict — readable and writable by policy_fn."""
        return self._state

    def observe(self, messages: list[SimMessage]) -> None:
        """Store inbound messages; available as self.inbox during decide()."""
        self._inbox = list(messages)

    def decide(self) -> list[SimMessage]:
        """Run policy_fn and clear the inbox."""
        outbound = self.policy_fn(self)
        self._inbox = []
        return outbound if outbound is not None else []

    # ------------------------------------------------------------------
    # Convenience accessor for policy_fn
    # ------------------------------------------------------------------

    @property
    def inbox(self) -> list[SimMessage]:
        """Messages received this round (non-empty only during decide())."""
        return self._inbox
