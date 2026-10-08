"""SimMessage <-> A2A Message (spec v1.0, proto3 JSON), dependency-free.

The seam for a future A2A transport: an A2A layer converts inbound messages with
from_a2a(), runs them through the unchanged Network / NSVerifier path, and sends
outbound ones with to_a2a(). Round semantics (one-round latency, topology) stay in
the simulator; A2A only carries the payload.

A2A Message: {messageId, contextId, taskId, role, parts, metadata}
A2A Part:    exactly one of {text, raw, url, data} (+ metadata, filename, mediaType)
A2A has no recipient field (addressing is by endpoint), so SILO-BENCH routing
fields ride in metadata["silo"].

Verifier mapping for a future A2A layer (not built): a message NSVerifier blocks is
not forwarded and feedback_to_a2a() goes back to the sender; a barrier reopen of a
submitted agent maps to its task moving COMPLETED -> TASK_STATE_INPUT_REQUIRED.
"""
from __future__ import annotations

import uuid
from typing import Any

from silo_sim.agent import SimMessage

ROLE_AGENT = "ROLE_AGENT"


def to_a2a(msg: SimMessage, context_id: str = "", task_id: str = "") -> dict:
    """SimMessage -> A2A Message dict. Text content -> text part; anything else -> data part."""
    part = {"text": msg.content} if isinstance(msg.content, str) else {"data": msg.content}
    return {
        "messageId": uuid.uuid4().hex,
        "contextId": context_id,
        "taskId": task_id,
        "role": ROLE_AGENT,
        "parts": [part],
        "metadata": {"silo": {"sender": msg.sender_id, "recipient": msg.recipient_id,
                              "sentAt": msg.sent_at, "known": sorted(msg.known)}},
    }


def from_a2a(m: dict) -> SimMessage:
    """A2A Message dict -> SimMessage. Needs metadata["silo"] sender/recipient (set by to_a2a
    or by the A2A layer from the endpoint). raw/url parts are ignored: no file payloads here."""
    silo = (m.get("metadata") or {}).get("silo") or {}
    texts = [p["text"] for p in m.get("parts", []) if "text" in p]
    datas = [p["data"] for p in m.get("parts", []) if "data" in p]
    content: Any
    if datas and not texts and len(datas) == 1:
        content = datas[0]
    elif texts and not datas:
        content = "\n".join(texts)
    else:  # mixed parts: keep both; NSVerifier.payload_values scans every dict field
        content = {"text": "\n".join(texts), "data": datas}
    return SimMessage(sender_id=int(silo["sender"]), recipient_id=int(silo["recipient"]),
                      content=content, sent_at=int(silo.get("sentAt", 0)),
                      known=frozenset(silo.get("known", ())))


def feedback_to_a2a(recipient: int, vector: dict, context_id: str = "", task_id: str = "") -> dict:
    """Verifier feedback vector (NSVerifier._note payload) -> A2A Message with one data part."""
    return {
        "messageId": uuid.uuid4().hex,
        "contextId": context_id,
        "taskId": task_id,
        "role": ROLE_AGENT,
        "parts": [{"data": vector}],
        "metadata": {"silo": {"sender": "verifier", "recipient": recipient}},
    }
