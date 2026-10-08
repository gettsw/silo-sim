"""Neuro-symbolic verifier: invariants (L2), feedback + barrier (L3)."""
import json
from collections import Counter

from silo_sim.adapters import make_mock_fn
from silo_sim.agent import SimMessage
from silo_sim.ns_verifier import MAX_RETRIES, NSVerifier, payload_values, sort_barrier
from silo_sim.silo_agent import SiloBenchAgent
from silo_sim.silo_runner import run_silo_case


class _Agent:
    def __init__(self, answer=None):
        self.submitted = answer is not None
        self.submission = {"answer": answer} if self.submitted else None
        self.notes = []

    def reopen(self):
        self.rejected, self.submitted, self.submission = self.submission, False, None

    def restore_rejected(self):
        if self.submitted or not getattr(self, "rejected", None):
            return False
        self.submitted, self.submission = True, self.rejected
        return True

    def notify(self, text):
        self.notes.append(text)


def test_provenance_blocks_and_feedback_is_capped():
    v = NSVerifier(seen={0: {1, 2}, 1: {3}})
    ok, _ = v.check(SimMessage(0, 1, {"text": "t", "values": [1, 2]}, 0))
    assert ok
    v.record(SimMessage(0, 1, {"text": "t", "values": [1, 2]}, 0))
    assert v.seen[1] == {1, 2, 3}                       # recipient can now relay 1, 2
    for k in range(MAX_RETRIES + 1):
        ok, reason = v.check(SimMessage(0, 1, {"text": "t", "values": [99 + k]}, k))
        assert not ok and reason == "HALLUCINATED_VALUE"
    a = {0: _Agent(), 1: _Agent()}
    v.end_round(a)
    assert len(a[0].notes) == 1 and a[0].notes[0].count("LOGIC_VIOLATION") == MAX_RETRIES
    assert a[1].notes == []                             # feedback goes to the sender only
    assert [x["type"] for x in v.violations] == ["HALLUCINATED_VALUE"] * (MAX_RETRIES + 1)
    assert v.check(SimMessage(0, 1, "free text 12345", 5))[0]  # free text is not provenance-checked


def _v(seen, inputs=(), n=0):
    return NSVerifier(seen=seen, inputs=Counter(inputs), n_agents=n, barrier=sort_barrier)


def _types(answers, v):
    return {vt for _, vt in sort_barrier(answers, v)}


def test_sort_barrier():
    v = _v({0: {1, 5, 9}, 1: {2, 3, 7}})
    assert _types({0: [1, 9], 1: [2, 3]}, v) == {"BOUNDARY_VIOLATION"}
    assert _types({0: [5, 1], 1: [2, 8]}, v) == {"LOCAL_ORDER", "HALLUCINATED_VALUE", "BOUNDARY_VIOLATION"}
    assert _types({0: [1], 1: [2, 3]}, v) == set()
    assert _types({0: "done"}, v) == {"MALFORMED_ANSWER"}


def test_sort_barrier_disjoint_and_conservation():
    seen = {0: {1, 2, 3, 9}, 1: {2, 3, 4, 9}, 2: {5, 9}}
    v = _v(seen, inputs=[1, 2, 3, 4, 5, 9], n=3)
    # 3 placed twice (only one exists), agents 0 and 1 blamed; agent 2 not yet submitted
    out = sort_barrier({0: [1, 3], 1: [3, 4]}, v)
    assert ([0, 1], "DUPLICATE_VALUE") in out
    assert "MISSING_VALUE" not in {vt for _, vt in out}   # unsubmitted agent may hold the rest
    # everyone submitted: 2 falls inside agent 0's range, 9 is above everyone -> last agent
    out = sort_barrier({0: [1, 3], 1: [4], 2: [5]}, v)
    missing = {tuple(ids) for ids, vt in out if vt == "MISSING_VALUE"}
    assert missing == {(0,), (2,)}, missing
    assert sort_barrier({0: [1, 2, 3], 1: [4], 2: [5, 9]}, v) == []


def test_text_lists_are_checked_but_bounds_pass():
    universe = {12, 57, 60, 81}
    assert payload_values("take [12, 57, 60] please", universe) == [12, 57, 60]
    assert payload_values("ranges [0, 250, 500, 750]", universe) is None     # bounds, not data
    assert payload_values("pair [12, 57]", universe) is None                 # < 3 numbers
    assert payload_values({"text": "also [57, 60, 81]", "values": [12]}, universe) == [12, 57, 60, 81]
    assert payload_values({"data": [12, 57, 60]}, universe) == [12, 57, 60]        # agent-chosen key
    assert payload_values({"sorted_data": [57, 60, 81]}, universe) == [57, 60, 81]
    assert payload_values({"ranges": [0, 250, 500]}, universe) is None              # bounds still pass
    assert payload_values("DataFrom3:12,57,60", universe) == [12, 57, 60]           # no brackets
    v = NSVerifier(seen={0: {12, 57}}, inputs=Counter(universe))
    ok, reason = v.check(SimMessage(0, 1, "here: [12, 57, 60]", 0))
    assert not ok and reason == "HALLUCINATED_VALUE"
    assert "60" not in json.dumps(v._notes)                 # feedback never names values


def test_barrier_reopens_at_most_max_retries():
    v = _v({0: {1, 9}, 1: {2}})
    a = {0: _Agent([1, 9]), 1: _Agent([2])}
    for _ in range(MAX_RETRIES):
        v.end_round(a)
        assert not a[0].submitted and not a[1].submitted
        a[0].submitted, a[0].submission = True, {"answer": [1, 9]}
        a[1].submitted, a[1].submission = True, {"answer": [2]}
    v.end_round(a)
    assert a[0].submitted and a[1].submitted            # budget spent: accept as-is


def test_run_silo_case_with_verifier():
    send = ('<tool_call><tool>send_message</tool><parameters><target_id>1</target_id>'
            '<content>{"text": "take these", "values": [123456]}</content></parameters></tool_call>')
    submit = ('<tool_call><tool>submit_result</tool><parameters><answer>[3, 1]</answer>'
              '</parameters></tool_call>')
    r = run_silo_case("benchmarks/III-21_n2.json", "msg", make_mock_fn([send, submit]),
                      max_rounds=4, ns_verifier=True)
    types = {x["type"] for x in r.trace["violations"]}
    assert "HALLUCINATED_VALUE" in types and "LOCAL_ORDER" in types, types
    assert r.telemetry["ns_violations"] == len(r.trace["violations"])
    r0 = run_silo_case("benchmarks/III-21_n2.json", "msg", make_mock_fn([send, submit]), max_rounds=4)
    assert "violations" not in r0.trace


def test_rejected_answer_restored_if_never_resubmitted():
    submit = "<tool_call><tool>submit_result</tool><parameters><answer>[3, 1]</answer></parameters></tool_call>"
    a = SiloBenchAgent(0, make_mock_fn([submit]), "msg", "", "", 2)
    a.decide()
    a.reopen()
    assert not a.submitted and a.submission is None
    assert a.restore_rejected() and a.submission["answer"] == [3, 1] and "known_at_submit" in a.state
    assert not a.restore_rejected()                        # already submitted: no-op
    v = _v({0: {1, 9}, 1: {2}})
    agents = {0: _Agent([1, 9]), 1: _Agent([2])}
    v.end_round(agents)                                    # boundary violation: both reopened
    agents[1].submitted, agents[1].submission = True, {"answer": [1]}   # agent 1 resubmits
    assert v.finish(agents) == 1                           # only agent 0 is restored
    assert agents[0].submission == {"answer": [1, 9]} and agents[1].submission == {"answer": [1]}


def test_resends_pass_and_budget_per_violation_type():
    v = NSVerifier(seen={0: {1, 2, 3}})
    m = SimMessage(0, 1, "hello [1, 2, 3]", 0)
    assert v.check(m)[0]
    v.record(m)
    assert v.check(m)[0]                                   # a resend of the same data is delivered
    assert v.violations == []
    for _ in range(MAX_RETRIES + 1):
        v.check(SimMessage(0, 0, "hi again", 1))           # self-send budget spent
    v.check(SimMessage(0, 1, {"text": "t", "values": [99]}, 2))
    a = {0: _Agent(), 1: _Agent()}
    v.end_round(a)
    assert a[0].notes[0].count("SELF_SEND") == MAX_RETRIES
    assert "HALLUCINATED_VALUE" in a[0].notes[0]           # separate budget, still delivered


def test_generic_flags_from_case_metadata():
    load = lambda f: NSVerifier.for_case(json.load(open(f, encoding="utf-8")))
    i01, ii11, iii21, iii22 = (load(f"benchmarks/{c}_n5.json") for c in ("I-01", "II-11", "III-21", "III-22"))
    assert (i01.needs_all_data, i01.consensus) == (True, True)
    assert (ii11.needs_all_data, ii11.consensus) == (False, False)   # mesh, segmented
    assert (iii21.needs_all_data, iii21.consensus) == (True, False)  # sort: per-agent answers
    assert (iii22.needs_all_data, iii22.consensus) == (True, True)


def test_premature_submission_and_disagreement():
    v = NSVerifier(n_agents=3, needs_all_data=True, consensus=True)
    a = {i: _Agent(ans) for i, ans in enumerate([7, 7, 9])}
    for i, k in enumerate([3, 1, 3]):
        a[i].state = {"known_at_submit": k}              # agent 1 saw 1 of 3 shards
    out = v._generic(a, {i: x.submission["answer"] for i, x in a.items()})
    assert ([1], "PREMATURE_SUBMISSION") in out
    assert [ids for ids, vt in out if vt == "DISAGREEMENT"] == [[2]]   # 7 is the majority
    assert {tuple(ids) for ids, vt in v._generic(a, {0: 7, 2: 9}) if vt == "DISAGREEMENT"} == {(0,), (2,)}  # tie
    v.end_round(a)
    assert not a[1].submitted and not a[2].submitted and a[0].submitted
    assert "PREMATURE_SUBMISSION" in a[1].notes[0] and "DISAGREEMENT" in a[2].notes[0]


def test_generic_checks_run_on_every_case():
    submit = "<tool_call><tool>submit_result</tool><parameters><answer>5</answer></parameters></tool_call>"
    wait = "<tool_call><tool>wait</tool><parameters></parameters></tool_call>"
    # round 1: both submit with only their own shard -> reopened; round 2: both only wait
    r = run_silo_case("benchmarks/I-01_n2.json", "msg", make_mock_fn([submit, submit, wait, wait]),
                      max_rounds=2, ns_verifier=True)
    assert "PREMATURE_SUBMISSION" in {x["type"] for x in r.trace["violations"]}
    assert r.telemetry["ns_restored"] == 2 and r.all_submitted  # never resubmitted -> answers restored


def test_multi_agent_feedback_names_partners_not_values():
    v = _v({0: {1, 3}, 1: {3, 4}}, inputs=[1, 3, 4], n=2)
    a = {0: _Agent([1, 3]), 1: _Agent([3, 4])}               # 3 placed twice; no boundary break (3 <= 3)
    v.end_round(a)
    note0, note1 = json.loads(a[0].notes[0].split("\n")[1]), json.loads(a[1].notes[0].split("\n")[1])
    assert note0["violation_type"] == "DUPLICATE_VALUE" and note0["with_agents"] == [1]
    assert note1["with_agents"] == [0]
    assert "3" not in a[0].notes[0].replace('"with_agents": [1]', "")  # no values leaked
