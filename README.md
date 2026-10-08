# silo-sim

A small round-based multi-agent simulation SDK: agents → network → simulator → verifiers.
Extracted from the [SILO-BENCH](https://arxiv.org/abs/2603.01045) environment.

## Quickstart

```bash
pip install -e .
python examples/quickstart.py   # run from the repo root
```

```pythonimport random

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
```

Expected output:

```
{0: 40, 1: 40, 2: 40, 3: 40, 4: 40}
[I-01 | broadcast] rounds=3 all_submitted=False S=0.000 P=0.000 C=2.0 D=0.000
```

- `silo_sim.agent` / `llm_agent` / `silo_agent` — agent protocol, rule agents, LLM agents, SILO-BENCH agent
- `silo_sim.network` — topology + message delivery (loss, latency, ring/star/full/custom)
- `silo_sim.simulator` — rounds and epochs
- `silo_sim.verification` / `ns_verifier` / `msg_filter` — verification layer between epochs
- `silo_sim.routing` — rule-based routers (full, star, chain, random, gossip)
- `silo_sim.a2a` — SimMessage ↔ A2A message adapter
- `silo_sim.silo_runner` — `run_silo_case` on SILO-BENCH case files (samples in `benchmarks/`)

```bash
pip install -e ".[dev]" && pytest
```

Unlicense (public domain).
