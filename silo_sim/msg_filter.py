"""Rule-based semantic filter for inter-agent messages.

A MessageFilter holds an ordered list of rules. Each rule is a callable:
    (msg: SimMessage, history: list[SimMessage]) -> (passed: bool, reason: str)

Base rules cover the general case. Pass extra rules at construction for
deployment-specific contracts.

Usage:
    f = MessageFilter(rules=BASE_RULES + [my_rule])
    net = Network(..., msg_filter=f)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from silo_sim.agent import SimMessage

Rule = Callable[[SimMessage, list[SimMessage]], tuple[bool, str]]

CONTEXT_WINDOW = 20  # ponytail: fixed window; raise if long-episode consistency matters


# ---------------------------------------------------------------------------
# Base rules — always-on general contracts
# ---------------------------------------------------------------------------

def rule_nonempty(msg: SimMessage, _: list[SimMessage]) -> tuple[bool, str]:
    """Block None or empty-string content."""
    if msg.content is None or msg.content == "":
        return False, "empty content"
    if isinstance(msg.content, str) and not msg.content.strip():
        return False, "whitespace-only content"
    return True, ""


def rule_no_duplicate(msg: SimMessage, history: list[SimMessage]) -> tuple[bool, str]:
    """Block identical content re-sent by the same agent to the same recipient."""
    for h in history:
        if (h.sender_id == msg.sender_id
                and h.recipient_id == msg.recipient_id
                and h.content == msg.content):
            return False, "duplicate message"
    return True, ""


def rule_no_self_send(msg: SimMessage, _: list[SimMessage]) -> tuple[bool, str]:
    """Block self-addressed messages."""
    if msg.sender_id == msg.recipient_id:
        return False, "self-send"
    return True, ""


BASE_RULES: list[Rule] = [rule_nonempty, rule_no_duplicate, rule_no_self_send]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@dataclass
class MessageFilter:
    rules: list[Rule] = field(default_factory=lambda: list(BASE_RULES))
    _history: list[SimMessage] = field(default_factory=list, repr=False)

    def check(self, msg: SimMessage) -> tuple[bool, str]:
        """Return (True, '') if all rules pass, else (False, first failure reason)."""
        for rule in self.rules:
            ok, reason = rule(msg, self._history)
            if not ok:
                return False, reason
        return True, ""

    def record(self, msg: SimMessage) -> None:
        """Add a passed message to history; prune to CONTEXT_WINDOW."""
        self._history.append(msg)
        if len(self._history) > CONTEXT_WINDOW:
            self._history = self._history[-CONTEXT_WINDOW:]

    def reset(self) -> None:
        """Clear history between episodes."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from silo_sim.network import Network

    f = MessageFilter()
    m = SimMessage(sender_id=0, recipient_id=1, content="hello", sent_at=1)

    ok, _ = f.check(m); assert ok
    f.record(m)
    ok, reason = f.check(m); assert not ok and reason == "duplicate message", reason

    empty = SimMessage(0, 1, "", 2)
    ok, reason = f.check(empty); assert not ok and reason == "empty content", reason

    whitespace = SimMessage(0, 1, "   ", 2)
    ok, reason = f.check(whitespace); assert not ok and reason == "whitespace-only content", reason

    self_send = SimMessage(0, 0, "hi", 3)
    ok, reason = f.check(self_send); assert not ok and reason == "self-send", reason

    # Network integration — assign after construction
    net = Network.fully_connected([0, 1, 2], random.Random(42))
    net.msg_filter = f
    net.send(m)  # duplicate — blocked
    assert net.queue_size() == 0
    net.send(SimMessage(0, 1, "world", 2))
    assert net.queue_size() == 1

    print("all checks passed")
