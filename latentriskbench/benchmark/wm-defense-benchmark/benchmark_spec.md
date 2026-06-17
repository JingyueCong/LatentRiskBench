# Conditional Agent Safety Benchmark Spec

Version: 0.1

## 1. Scope

This benchmark evaluates **agent behavior under configurable attack and defense conditions** with world-model-assisted safety measurement.

Evaluation score is defined over:

`F(agent, attack, defense, task, seed)`

- `agent`: evaluation subject (leaderboard target)
- `attack`: condition
- `defense`: condition
- `task`: goal + environment dynamics
- `seed`: stochastic control variable

## 2. Benchmark Axes

- Subject axis: `agent`
- Condition axis: `attack`, `defense`, `task`
- Instrument axis: `world model`, `metrics`, `harness`

## 3. Required Episode Fields

Each episode/task item MUST include:

- `task_id: str`
- `intent: str`
- `state` (offline) or `initial_state` (online)
- `candidate_actions: List[str]` (offline) or `steps[*].candidate_actions` (online)

Recommended fields (for factorized evaluation):

- `agent_id: str`
- `attack_family: str`
- `attack_depth: str`
- `attack_id: str`
- `defense_id: str`
- `seed: int`
- `env_suite: str`
- `orig_goal_success: bool` (whether the original task is feasible and should count toward utility)

If omitted, harness defaults are used.

## 4. Offline Protocol

Offline mode evaluates one decision point per task:

1. Build candidate action set
2. Evaluate with world-model-based defense
3. Record selection, risk, and policy-violation metadata
4. Compare against labeled unsafe candidate indices

No real environment step is required.

## 5. Online Protocol

Online mode evaluates sequential decisions:

1. Read current observation/state
2. Generate or load candidate actions
3. Run defense decision
4. Execute selected action in transition model / environment
5. Update trajectory through harness API
6. Repeat until done or stop condition

## 6. Output Contract

For each task:

- `agent_id`, `attack_family`, `attack_depth`, `attack_id`, `defense_id`, `task_id`, `seed`
- step/task utility and safety signals
- world-model risk outputs
- defense decision traces

## 7. Interpretation Rules

- Attack effect: compare `attack=none` vs `attack!=none` under same `(agent, defense, task, seed)`
- Defense gain: compare `defense=none` vs `defense!=none` under same `(agent, attack, task, seed)`
- Defense cost: compare utility drop under benign/no-attack slice

## 8. Current Harness Mapping

Implementation entrypoint:

- `benchmark/wm-defense-benchmark/runners/run.py`

Current modes:

- `--mode offline`
- `--mode online`

Protocol defaults support:

- `--protocol benchmark/wm-defense-benchmark/eval_protocol.yaml`
