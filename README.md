# LatentRiskBenchmark

Self-contained, runnable copy of the WM-Defense / latent-risk agent-defense benchmark,
bundled together with the minimal SafePred core it depends on.

## Layout

```
LatentRiskBenchmark/
└── SafePred/                         # minimal SafePred core (the benchmark imports this)
    ├── __init__.py, wrapper.py       # SafePredWrapper entry point
    ├── core/ models/ agent/          # world model, SafeAgent, trajectory graph
    ├── config/ adapters/ utils/      # config, benchmark adapters, llm_client/logger
    ├── setup.py, requirements.txt
    └── benchmark/
        └── wm-defense-benchmark/     # the benchmark itself
            ├── runners/              # run.py, run_matrix.py
            ├── attacks/ defenses/    # attack & defense handlers
            ├── agents/ metrics/      # planner agents, outcome/world-model metrics
            ├── attack_generator/     # compositional / adaptive attack generation
            ├── schemas/ tooling/ tools/ validators/ policy_oracle/
            └── data/                 # benchmark INPUTS only (see below)
```

> The `SafePred/benchmark/wm-defense-benchmark/` nesting is kept intentionally:
> the runner resolves the package root via fixed `Path(__file__).parents[N]` offsets,
> so this depth lets it run with **zero code changes**.

## What's included vs. excluded

**Included** — everything needed to run the benchmark:
- Full benchmark code + the SafePred core modules it imports.
- Benchmark **input** data: `tasks*.jsonl`, `base_tasks.jsonl`, `policies.json`,
  `tool_registry.json`, and all `attack_payloads/`.

**Excluded** — experiment outputs and dev cruft (not needed to reproduce a run):
- `data/results_*.{json,csv}`, `data/attack_generation_logs/`, `data/wm_gcg_logs/`,
  `data/results_p4_ablation/`
- SafePred `logs/`, `tests/`, all `__pycache__/`, and sibling benchmarks
  (`horizon-defense-benchmark`, `worldmodel`).

## Setup & run

```bash
cd LatentRiskBenchmark
pip install -r SafePred/requirements.txt          # numpy, requests, pyyaml, + LLM client deps
# LLM API key/URL are read from a .env in the benchmark dir (see config.yaml)

cd SafePred/benchmark/wm-defense-benchmark
export PYTHONPATH="$(cd ../../.. && pwd)"          # so `import SafePred` resolves

python list_registry.py                            # list registered attacks/defenses/agents
python validate_all_data.py                        # validate all input data
python run.py --defense-mode world_model --max-tasks 5
```

## State-conditioned (SC) construction tooling

New top-down pipeline for building genuinely *state-conditioned* risk items (risk that is
invisible in the action surface and only emerges after aggregating the committed prefix into
state). All paths under `SafePred/benchmark/wm-defense-benchmark/`:

- `schemas/state_schema.py` — structured trajectory-state schema (the 4 SC mechanisms),
  `STATE_ONLY_PATHS` (legal injection sites), `canonical_state()` (path-invariance equality).
- `policy_oracle/struct_oracle.py` — `StructPolicyOracle` (`struct_v1`): machine-checkable
  conditions for P000–P004 over the structured state. `data/policies.json` carries the matching
  human-readable `state_condition` per policy.
- `attack_generator/sc_injector.py` — deterministic reconstruction `R` + the injector that flips
  one state-only field while holding the terminal action fixed; certifies pairs by construction.
- `data/sc_seeds.jsonl` — benign multi-step seed workflows and their leverage targets.

Build + certify the SC suite (no API needed):

```bash
cd SafePred/benchmark/wm-defense-benchmark
python tools/struct_oracle_demo.py        # schema + oracle smoke test
python tools/build_sc_suite.py            # -> data/tasks_sc.jsonl (risky + matched twins)
```

Diagnostic (needs an LLM backend — supports OpenRouter via `--api-url/--api-key`):

```bash
python tools/exclusion_battery.py --dry-run    # inspect the two views without API calls
```

See `SafePred/benchmark/wm-defense-benchmark/README.md` for full benchmark documentation.
