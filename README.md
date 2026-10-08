# silo-sim

A small, dependency-light SDK for **round-based multi-agent simulation**: agents exchange messages over an explicit network, a simulator drives the clock, and verifiers check the results. Extracted from the [SILO-BENCH](https://arxiv.org/abs/2603.01045) environment for evaluating coordination among LLM agents that each hold a private data shard.

## Features

- **Agents** — a tiny `BaseAgent` protocol (`observe` / `decide` / `state`) with rule-based, LLM-backed and SILO-BENCH implementations. Bring your own LLM via a plain `llm_fn(messages) -> str`.
- **Network** — explicit topology (ring, star, fully connected, custom edges), seeded packet loss, and an optional message filter.
- **Simulator** — synchronous rounds (deliver, observe, decide, send) with per-round history, plus epoch-based runs with a verifier between epochs.
- **Verification** — pluggable verifiers (`PassVerifier`, `FnVerifier`, `AnswerVerifier`, `ConsistencyVerifier`) and a neuro-symbolic `NSVerifier` that checks messages and submissions against task invariants.
- **Routing** — rule-based routers (full, star, chain, random, gossip) to restrict who may message whom each round.
- **SILO-BENCH runner** — `run_silo_case` runs a benchmark case end to end and reports success (S), partial correctness (P), tokens per round (C) and communication density (D).
- **A2A adapter** — convert `SimMessage` to and from A2A message dicts.
- **Reproducible** — everything random takes a seed; LLM calls can be mocked, so tests need no API key.

## Install

Requires Python 3.10+.

```bash
git clone <this-repo-url> && cd silo-sim
pip install -e .
```

For real LLM runs, pass `make_openai_fn(...)` (any OpenAI-compatible endpoint) as `llm_fn`; see `configs/config.example.yaml` for the settings it reads.

## Quickstart

```bash
pip install -e .
python examples/quickstart.py   # run from the repo root
```

```python
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
```

Expected output:

```
{0: 40, 1: 40, 2: 40, 3: 40, 4: 40}
[I-01 | broadcast] rounds=3 all_submitted=False S=0.000 P=0.000 C=2.0 D=0.000
```

## Modules

| Module | Purpose |
|---|---|
| `silo_sim.agent`, `llm_agent`, `silo_agent` | Agent protocol, rule agents, LLM agents, SILO-BENCH agent |
| `silo_sim.network` | Topology, message queue and delivery, packet loss |
| `silo_sim.simulator` | Rounds, epochs, step history |
| `silo_sim.verification`, `ns_verifier`, `msg_filter` | Verification layer and message filters |
| `silo_sim.routing` | Rule-based routers |
| `silo_sim.a2a` | SimMessage and A2A message adapter |
| `silo_sim.adapters` | `make_openai_fn`, `make_mock_fn` |
| `silo_sim.silo_runner` | `run_silo_case` and `SiloResult` |

Sample SILO-BENCH cases (N=2 and N=5) live in `benchmarks/`; the full benchmark and generator are in the upstream SILO-BENCH repository.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use the SILO-BENCH environment, please cite the paper: <https://arxiv.org/abs/2603.01045>.

## License

[Unlicense](LICENSE) (public domain).
