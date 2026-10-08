import random

from silo_sim.agent import SimMessage
from silo_sim.routing import RoutingController, make_gossip_router, make_random_router
from silo_sim.silo_agent import SiloBenchAgent

IDS = list(range(5))


def _send(target: int) -> str:
    return (f"<tool_call><tool>send_message</tool><parameters><target_id>{target}</target_id>"
            f"<content>hi</content></parameters></tool_call>"
            "<tool_call><tool>wait</tool><parameters></parameters></tool_call>")


def _agent(aid: int, reply: str) -> SiloBenchAgent:
    return SiloBenchAgent(agent_id=aid, llm_fn=lambda _msgs: reply, protocol="msg",
                          system_prompt="", task_prompt="", num_agents=len(IDS))


def test_gossip_picks_ring_successor_plus_max_gain_target():
    states = {0: {"known": [0, 1, 2]}, 1: {"known": [1]}, 2: {"known": [2]},
              3: {"known": [3]}, 4: {"known": [0, 1, 2, 4]}}
    adj = make_gossip_router(k=1)(0, states, IDS, random.Random(0))
    # ring successor 1, then 3 (gain {0,1,2}) beats 2 (gain {0,1}) and 4 (gain 0)
    assert adj[0] == [1, 3]


def test_gossip_skips_submitted_and_zero_gain_targets():
    states = {i: {"known": IDS} for i in IDS}
    states[3]["submission"] = {"answer": 1}
    adj = make_gossip_router(k=3)(0, states, IDS, random.Random(0))
    assert all(v == [(i + 1) % 5] for i, v in adj.items())  # nobody lacks anything -> ring only


def test_random_router_shape():
    adj = make_random_router(k=2)(0, {}, IDS, random.Random(0))
    for i, targets in adj.items():
        assert targets[0] == (i + 1) % 5 and len(set(targets)) == 3 and i not in targets


def test_hybrid_bounces_off_route_send_and_advise_does_not():
    route = lambda *a: {i: [(i + 1) % 5] for i in IDS}
    for enforce, tell, expect_sent in [(True, True, False), (False, True, True), (True, False, True)]:
        a = _agent(0, _send(3))
        ctl = RoutingController(route, enforce=enforce, tell=tell, rng=random.Random(0))
        ctl.attach({0: a})
        ctl(0, {})
        out = a.decide()
        assert (len(out) == 1) == expect_sent, (enforce, tell)
        assert a._bounced == (0 if expect_sent else 1)
        assert a._sends_routed == 1 and a._sends_on_route == 0  # 3 was off-route in every mode
        assert bool(a.routing_note) == tell


def test_knowledge_propagates_transitively():
    a = _agent(0, _send(1))
    a.observe([SimMessage(sender_id=2, recipient_id=0, content="x", sent_at=0, known=frozenset({2, 4}))])
    assert a.state["known"] == [0, 2, 4]
    (msg,) = a.decide()
    assert msg.known == frozenset({0, 2, 4})


def test_run_trace_matches_result():
    import pytest
    from silo_sim import make_mock_fn, run_silo_case
    from silo_sim.routing import make_router
    answer = "<tool_call><tool>submit_result</tool><parameters><answer>827</answer></parameters></tool_call>"
    for routing in (None, RoutingController(make_router("full", 1), enforce=True, tell=True, rng=random.Random(0))):
        r = run_silo_case("benchmarks/I-01_n2.json", "msg", make_mock_fn([_send(1), answer]),
                          max_rounds=4, routing=routing)
        t = r.trace
        assert len(t["rounds"]) == r.rounds_run
        assert sum(t["q"]) / len(t["q"]) == pytest.approx(r.P)
        assert t["rounds"][0]["sent"] == [[0, 1]]
        assert t["rounds"][1]["delivered"] == [[0, 1]]
        assert (t["rounds"][0]["route"] is None) == (routing is None)
        assert t["rounds"][-1]["submitted"] == [0, 1]


def test_star_and_chain_templates():
    from silo_sim.routing import chain_router, routing_note, star_router
    star = star_router(0, {}, IDS, random.Random(0))
    assert star == {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
    assert "any agent" in routing_note(0, star[0], 5, enforce=True)  # hub is unrestricted
    chain = chain_router(0, {}, IDS, random.Random(0))
    assert chain == {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
