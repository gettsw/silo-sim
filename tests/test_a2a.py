"""SimMessage <-> A2A (v1.0 JSON) round trips, and the verifier reads A2A-shaped content."""
import json
from collections import Counter

from silo_sim.a2a import feedback_to_a2a, from_a2a, to_a2a
from silo_sim.agent import SimMessage
from silo_sim.ns_verifier import NSVerifier


def test_round_trip_text_and_data():
    for content in ("DATA:[12, 57, 60]", {"data": [12, 57, 60]}, [1, 2, 3]):
        m = SimMessage(0, 3, content, 4, frozenset({0, 2}))
        a = to_a2a(m, context_id="ep1")
        json.dumps(a)                                        # wire-serializable
        assert a["role"] == "ROLE_AGENT" and len(a["parts"]) == 1
        assert ("text" in a["parts"][0]) == isinstance(content, str)
        assert from_a2a(a) == m


def test_mixed_parts_and_verifier_reads_them():
    a = {"messageId": "x", "role": "ROLE_AGENT",
         "parts": [{"text": "yours:"}, {"data": {"values": [12, 57, 60]}}],
         "metadata": {"silo": {"sender": 0, "recipient": 1}}}
    msg = from_a2a(a)
    assert msg.content == {"text": "yours:", "data": [{"values": [12, 57, 60]}]}
    v = NSVerifier(seen={0: {12, 57}}, inputs=Counter({12: 1, 57: 1, 60: 1}))
    ok, reason = v.check(msg)                                # 60 never reached agent 0
    assert not ok and reason == "HALLUCINATED_VALUE"
    ok, _ = v.check(from_a2a(to_a2a(SimMessage(0, 1, "have [12, 57, 12]", 1))))
    assert ok


def test_feedback_message_shape():
    a = feedback_to_a2a(2, {"verifier": "EPOCH_BARRIER", "violation_type": "DUPLICATE_VALUE", "with_agents": [4]})
    assert a["parts"] == [{"data": {"verifier": "EPOCH_BARRIER", "violation_type": "DUPLICATE_VALUE", "with_agents": [4]}}]
    assert a["metadata"]["silo"]["recipient"] == 2
