# LatentRiskBench

LatentRiskBench is a reproducible benchmark and tooling stack for evaluating state-dependent residual risk in enterprise agents. This repository packages the benchmark harness, the minimal `latentriskbench` core it depends on, and the assets needed to run the paper's controlled and e-commerce instantiations.

## Repository Overview

The main benchmark lives under `latentriskbench/benchmark/wm-defense-benchmark/`. That directory contains the runner, attacks, defenses, metrics, validators, schemas, and task data used by the paper.

The nested path is intentional: the benchmark runner resolves the package root through fixed relative offsets, so the current structure keeps the code runnable without modification.

## What Is Included

- Benchmark code and the minimal core modules it imports.
- Input data for the benchmark: `tasks*.jsonl`, `base_tasks.jsonl`, `policies.json`, `tool_registry.json`, and `attack_payloads/`.
- Construction tooling for the state-conditioned suite and its matched benign twins.

## What Is Excluded

- Generated outputs such as `results*.json`, `results*.csv`, and attack-generation logs.
- Local runtime noise such as `__pycache__/`, notebook checkpoints, `.env` files, and editor settings.
- Sibling benchmark folders that are not part of this release snapshot.

## Quick Start

```bash
cd LatentRiskBench
pip install -r latentriskbench/requirements.txt

cd latentriskbench/benchmark/wm-defense-benchmark
export PYTHONPATH="$(cd ../../.. && pwd)"

python list_registry.py
python validate_all_data.py
python run.py --defense-mode world_model --max-tasks 5
```

## State-Conditioned Suite

The repository includes a state-conditioned construction pipeline for cases where the decisive risk signal is not visible from the terminal action alone. The relevant tooling lives in the benchmark package:

- `schemas/state_schema.py` for the structured trajectory-state schema.
- `policy_oracle/struct_oracle.py` for machine-checkable policy conditions over state.
- `attack_generator/sc_injector.py` for deterministic state-only perturbations with fixed terminal actions.
- `data/sc_seeds.jsonl` for benign multi-step seed workflows.

To build and certify the state-conditioned suite locally:

```bash
cd latentriskbench/benchmark/wm-defense-benchmark
python tools/struct_oracle_demo.py
python tools/build_sc_suite.py
```

## Documentation

For the full benchmark specification, see [latentriskbench/benchmark/wm-defense-benchmark/README.md](latentriskbench/benchmark/wm-defense-benchmark/README.md).

## Citation

If you use this repository in your work, please cite the paper associated with LatentRiskBench.
