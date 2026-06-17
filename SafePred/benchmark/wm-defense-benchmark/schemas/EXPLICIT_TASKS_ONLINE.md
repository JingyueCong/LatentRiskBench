# Explicit Online Task Schema

Explicit online tasks are stored in:

- `data/tasks_online.jsonl`

Each line must be one JSON object representing a multi-step trajectory with
fully materialized attack metadata.

## Required Fields

- `task_id: str`
- `attack_id: str`
- `attack_family: str`
- `attack_depth: str`
- `env_suite: str`
- `orig_goal_success: bool`
- `intent: str`
- `initial_state: object`
- `steps: List[object]`

## Required Per-Step Fields

- `candidate_actions: List[str]`
- `unsafe_action_indices: List[int]`
- `proposed_action_index: int`
- `current_response: str`

## Optional Irreversibility / Early-Warning Fields (task-level)

These annotate the *delayed-risk subset* used for horizon-aware lead-time
evaluation. They are optional, so tasks without them remain valid, but every
trajectory in the delayed-risk subset must carry them. **Step indices are
1-based**, matching the online runner.

- `policy_class: str` — one of `data_leakage` (credential / sensitive-data
  leakage), `phishing_propagation` (phishing / social-engineering propagation),
  `goal_hijacking` (adopting an attacker goal via untrusted instructions).
- `harm_event: str` — description of the irreversible harm.
- `irreversible_by_construction: bool` — `true` if the harm cannot be undone once
  the harm action executes (credential sent, message delivered, command run).
- `irreversible_harm_step: int` — 1-based step at which irreversible harm occurs
  on the unsafe path. That step must expose an unsafe action.
- `earliest_warning_step: int` — 1-based earliest step at which the risk is
  observable (the first step a perfect predictive defense could warn). Must be
  `<= irreversible_harm_step`. The **warning window** =
  `irreversible_harm_step - earliest_warning_step` is the maximum lead time any
  defense can achieve on this trajectory.
- `delayed_risk: bool` — `true` if the trajectory belongs to the delayed-risk
  subset (irreversible harm with a positive warning window).

## Optional Per-Step Fields

- `state: object`
- `next_state_by_action: object`
- `next_state: object`
- `done_actions: List[str]`
- `done: bool`

## Constraints

- `steps` must be non-empty
- each step must have non-empty `candidate_actions`
- each `proposed_action_index` must be in range
- each `unsafe_action_indices` entry must refer to a valid candidate
- when `next_state_by_action` is present, its values should be state objects
- `irreversible_harm_step` and `earliest_warning_step`, when present, must be
  1-based indices within `steps`, with `earliest_warning_step <= irreversible_harm_step`
- the `irreversible_harm_step` step must list at least one `unsafe_action_indices` entry

## Validation

Run:

```bash
python validate_tasks_online.py

# Canonical entrypoint
python validators/validate_tasks_online.py
```
