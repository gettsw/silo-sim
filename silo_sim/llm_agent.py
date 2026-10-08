"""LLM-backed agent satisfying the BaseAgent protocol.

The agent is deliberately decoupled from any specific LLM provider.
It receives a ``llm_fn`` — a plain callable that accepts an OpenAI-style
message list and returns a content string — so the same class works with
OpenAI, Ollama, Claude, or a mock function in tests.

Response envelope (JSON, all fields optional)::

    {
        "send": [{"to": <int>, "content": <any>}, ...],
        "remember": {"key": "value", ...}
    }

* ``send``    – outbound SimMessages to enqueue this round (default: [])
* ``remember``– dict merged into agent._state after parsing (default: {})

A response that is not valid JSON, or valid JSON that omits both fields,
is treated as a no-op (no messages sent, no state update).  A warning is
stored in ``_state["_last_parse_error"]`` so experiments can detect it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from silo_sim.agent import SimMessage


# Type alias — keeps the rest of the code readable.
LLMFn = Callable[[list[dict[str, str]]], str]

_SYSTEM_DEFAULT = (
    "You are an agent in a multi-agent network simulation. "
    "Each round you receive messages from neighbouring agents in your inbox. "
    "Reply with a JSON object: "
    '{"send": [{"to": <agent_id>, "content": <any>}], "remember": {<key>: <value>}}. '
    "Both fields are optional. Only send to agents you are directly connected to."
)


@dataclass
class LLMAgent:
    """Agent whose decisions are made by an LLM.

    Parameters
    ----------
    agent_id:
        Unique integer identifier, must match the Network topology key.
    llm_fn:
        Callable(messages: list[dict]) -> str.  Receives the full
        conversation history (system + prior turns + current user turn)
        and returns the assistant's response string.
    system_prompt:
        Injected as the first ``system`` message on every call.  Defaults
        to a generic simulation prompt.
    keep_history:
        If True, prior assistant/user turns are retained across rounds so
        the LLM has conversational memory.  If False, only the system
        prompt and the current round's user message are sent each call
        (lower token cost, no cross-round recall).
    """

    agent_id: int
    llm_fn: LLMFn
    system_prompt: str = _SYSTEM_DEFAULT
    keep_history: bool = True

    _state: dict[str, Any] = field(default_factory=dict, repr=False)
    _inbox: list[SimMessage] = field(default_factory=list, repr=False)
    _history: list[dict[str, str]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # BaseAgent protocol
    # ------------------------------------------------------------------

    def observe(self, messages: list[SimMessage]) -> None:
        """Store incoming messages; called by Simulator before decide()."""
        self._inbox = list(messages)

    def decide(self) -> list[SimMessage]:
        """Ask the LLM what to do, parse the response, return outbound msgs."""
        user_content = self._build_user_message()
        messages = self._build_prompt(user_content)

        raw = self.llm_fn(messages)

        outbound = self._parse_response(raw)

        # Append this turn to history (before clearing inbox).
        if self.keep_history:
            self._history.append({"role": "user", "content": user_content})
            self._history.append({"role": "assistant", "content": raw})

        self._inbox = []
        return outbound

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self) -> str:
        """Serialize the current inbox as a JSON user turn."""
        inbox_data = [
            {
                "from": m.sender_id,
                "content": m.content,
                "sent_at": m.sent_at,
            }
            for m in self._inbox
        ]
        return json.dumps({"inbox": inbox_data}, ensure_ascii=False)

    def _build_prompt(self, user_content: str) -> list[dict[str, str]]:
        """Assemble the full message list for this call."""
        prompt: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if self.keep_history:
            prompt.extend(self._history)
        prompt.append({"role": "user", "content": user_content})
        return prompt

    def _parse_response(self, raw: str) -> list[SimMessage]:
        """Parse LLM output into SimMessages and optional state updates.

        Any parse failure is non-fatal: the round produces no messages and
        the error is recorded in _state["_last_parse_error"].
        """
        # Strip markdown code fences if the LLM wrapped its JSON.
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # drop opening fence (```json or ```) and closing fence
            inner = [l for l in lines[1:] if l.strip() != "```"]
            text = "\n".join(inner).strip()

        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            self._state["_last_parse_error"] = str(exc)
            return []

        if not isinstance(envelope, dict):
            self._state["_last_parse_error"] = "response was not a JSON object"
            return []

        # Apply state updates first.
        remember = envelope.get("remember", {})
        if isinstance(remember, dict):
            self._state.update(remember)
            self._state.pop("_last_parse_error", None)

        # Build outbound messages.
        outbound: list[SimMessage] = []
        for item in envelope.get("send", []):
            if not isinstance(item, dict):
                continue
            try:
                recipient = int(item["to"])
                content = item.get("content", "")
                # sent_at is unknown here (Simulator increments clock after
                # decide()); use -1 as sentinel — Simulator can patch if needed.
                outbound.append(
                    SimMessage(
                        sender_id=self.agent_id,
                        recipient_id=recipient,
                        content=content,
                        sent_at=-1,
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue

        return outbound

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Clear conversation history without touching _state."""
        self._history = []

    @property
    def history(self) -> list[dict[str, str]]:
        """Read-only view of the conversation history."""
        return list(self._history)
