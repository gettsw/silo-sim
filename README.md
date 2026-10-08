# silo-sim

A small round-based multi-agent simulation SDK: agents → network → simulator → verifiers.
Extracted from the [SILO-BENCH](https://arxiv.org/abs/2603.01045) environment.

```python
from silo_sim import RuleAgent, Network, Simulator, SimMessage

# see silo_sim/simulator.py for a runnable example
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
