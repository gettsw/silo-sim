"""Max-propagation on a ring, then one SILO-BENCH case with a mock LLM (no API key)."""
import random

from silo_sim import Network, RuleAgent, SimMessage, Simulator, make_mock_fn, run_silo_case

# --- 1. Rule agents on a ring: every agent learns the global max -------------
ids = list(range(5))
net = Network.ring(ids, rng=random.Random(0))


def policy(agent):
    for msg in agent.inbox:
        agent.state["max"] = max(agent.state["max"], msg.content)
    return [SimMessage(agent.agent_id, n, agent.state["max"], 0) for n in net.neighbors(agent.agent_id)]


agents = {i: RuleAgent(i, policy, _state={"max": i * 10}) for i in ids}
sim = Simulator(agents=agents, network=net)
sim.run(rounds=5)
print({i: sim.agent_state(i)["max"] for i in ids})  # every value is 40

# --- 2. A SILO-BENCH case driven by a scripted (mock) LLM --------------------
llm = make_mock_fn(["<tool_call><tool>wait</tool><parameters></parameters></tool_call>"])
result = run_silo_case("benchmarks/I-01_n2.json", "broadcast", llm, max_rounds=3)
print(result.summary() if hasattr(result, "summary") else result)
