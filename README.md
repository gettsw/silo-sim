<div align="center">

# silo-sim

**A lightweight SDK for round-based multi-agent simulation.**

Agents exchange messages over an explicit network, a simulator drives the clock, and verifiers check the outcome.

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/arXiv-2603.01045-b31b1b.svg)](https://arxiv.org/abs/2603.01045)

</div>

---

## Overview

`silo-sim` is the simulation core of [SILO-BENCH](https://arxiv.org/abs/2603.01045), a benchmark for distributed coordination among LLM agents that each hold a private data shard. It is packaged as a standalone SDK so you can build, run, and verify your own multi-agent experiments, with or without an LLM.

The design goals are:

- **Small surface area.** An agent is three methods: `observe`, `decide`, `state`.
- **Explicit communication.** Who can talk to whom is a first-class `Network` object, not an implicit side effect.
- **Provider-agnostic.** LLM calls go through a plain `llm_fn(messages) -> str`; swap providers or mock them freely.
- **Reproducible.** Every source of randomness takes a seed, and runs need no API key when mocked.

## Features

| Area | What you get |
|---|---|
| **Agents** | `BaseAgent` protocol, `RuleAgent` (policy function), `LLMAgent`, and `SiloBenchAgent` |
| **Network** | Ring, star, fully connected, and custom-edge topologies; seeded packet loss; optional message filter |
| **Simulator** | Synchronous rounds (deliver → observe → decide → send), step history, and epoch runs with a verifier between epochs |
| **Verification** | `PassVerifier`, `FnVerifier`, `AnswerVerifier`, `ConsistencyVerifier`, and a neuro-symbolic `NSVerifier` |
| **Routing** | Rule-based routers: `full`, `star`, `chain`, `random`, `gossip` |
| **Benchmark runner** | `run_silo_case` executes a SILO-BENCH case and reports S, P, C, D (see [Metrics](#metrics)) |
| **Interop** | `SimMessage` ↔ A2A message adapter |

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/gettsw/silo-sim.git
cd silo-sim
pip install -e .
```

With development tools (pytest):

```bash
pip install -e ".[dev]"
```

## Quickstart

Run it from the repository root (no API key required):

```bash
python examples/quickstart.py
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
print(result.summary())
```

Expected output:

```
{0: 40, 1: 40, 2: 40, 3: 40, 4: 40}
[I-01 | broadcast] rounds=3 all_submitted=False S=0.000 P=0.000 C=2.0 D=0.000
```

> The mock agent only calls `wait`, so the case is not solved (S = 0). The example demonstrates the runner, not a solved task.

## Core concepts

### Simulation loop

Each round the simulator performs, in order:

1. `Network.deliver()` pushes queued messages into per-agent inboxes.
2. Every agent `observe`s its inbox. All agents observe before any agent decides, so results do not depend on iteration order.
3. Every agent `decide`s and emits messages.
4. Emitted messages are queued with `Network.send()` for the next round.

### Agents

Anything with `observe(messages)`, `decide() -> list[SimMessage]`, and a `state` dict satisfies `BaseAgent`. Use `RuleAgent` for hand-written policies and `LLMAgent` / `SiloBenchAgent` for LLM-driven behavior.

### Using a real LLM

Pass any callable that takes OpenAI-style chat messages and returns a string. For OpenAI-compatible endpoints:

```python
from silo_sim import make_openai_fn

llm = make_openai_fn(...)   # see configs/config.example.yaml for the settings it reads
```

### Metrics

`run_silo_case` returns a `SiloResult` with:

| Metric | Meaning |
|---|---|
| `S` | Success rate in [0, 1] |
| `P` | Partial correctness in [0, 1] |
| `C` | Average output tokens per round |
| `D` | Communication density (may exceed 1) |

It also exposes `rounds_run`, `all_submitted`, `submissions`, and the full step history in `steps`.

## Project layout

```
silo_sim/
├── agent.py            # BaseAgent protocol, RuleAgent, SimMessage
├── llm_agent.py        # LLM-backed agent
├── silo_agent.py       # SILO-BENCH agent
├── network.py          # Topology, delivery, packet loss
├── simulator.py        # Rounds, epochs, history
├── verification.py     # Verifier protocol and built-in verifiers
├── ns_verifier.py      # Neuro-symbolic verifier
├── msg_filter.py       # Message filter rules
├── routing.py          # Rule-based routers
├── a2a.py              # A2A message adapter
├── adapters.py         # make_openai_fn, make_mock_fn
├── silo_runner.py      # run_silo_case, SiloResult
└── utils/              # Config, LLM client, metrics, parsing, prompts
benchmarks/             # Sample SILO-BENCH cases (N=2 and N=5)
examples/quickstart.py
tests/
```

The full benchmark and case generator live in the upstream SILO-BENCH repository.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use the SILO-BENCH environment in your work, please cite the paper:

> *SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems.* arXiv:2603.01045. <https://arxiv.org/abs/2603.01045>

## License

Released into the public domain under the [Unlicense](LICENSE).
