# Explicit Offline Task Schema

Explicit offline tasks are stored in:

- `data/tasks.jsonl`

Each line must be one JSON object representing a single decision point with
fully materialized attack metadata.

## Required Fields

- `task_id: str`
- `attack_id: str`
- `attack_family: str`
- `attack_depth: str`
- `env_suite: str`
- `orig_goal_success: bool`
- `intent: str`
- `state: object`
- `candidate_actions: List[str]`
- `unsafe_action_indices: List[int]`
- `proposed_action_index: int`
- `current_response: str`

## Optional Fields

- `agent_id: str`
- `defense_id: str`
- `seed: int`
- `action_history: List[str]`

## Constraints

- `candidate_actions` must be non-empty
- `proposed_action_index` must be within bounds of `candidate_actions`
- `unsafe_action_indices` must contain only valid candidate indices
- explicit tasks must already include attack metadata; unlike base tasks, they
  are ready to run directly

## Validation

Run:

```bash
python validate_tasks.py

# Canonical entrypoint
python validators/validate_tasks.py
```
