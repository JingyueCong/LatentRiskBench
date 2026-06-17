# Attack Payload Schema

Attack payloads are stored under:

- `data/attack_payloads/<attack_family>/*.jsonl`

Each line must be one JSON object with the following fields.

## Required Fields

- `payload_id: str`
  - stable identifier for the payload
- `attack_id: str`
  - canonical attack id used by the registry
- `attack_family: str`
  - must match the attack-family directory
- `attack_depth: str`
  - recommended values: `L1`, `L2`, `L3`, `L4`
- `overrides: object`
  - task fields to override when applying this payload to a base task

## Target Selectors

At least one of the following must be present:

- `target_task_ids: List[str]`
- `target_tags: List[str]`

These selectors determine which base tasks the payload can be applied to.

## Constraints

- `overrides` must not redefine:
  - `attack_id`
  - `attack_family`
  - `attack_depth`
- those fields are owned by the payload header and injected by the compose step

## Minimal Example

```json
{
  "payload_id": "profile_secret_injection",
  "attack_id": "attack_l1_injection",
  "attack_family": "prompt_injection",
  "attack_depth": "L1",
  "target_task_ids": ["base_profile_edit"],
  "overrides": {
    "task_id": "t2_phishing_like_injection",
    "candidate_actions": ["..."],
    "unsafe_action_indices": [0]
  }
}
```

## Validation

Run:

```bash
python validate_payloads.py

# Canonical entrypoint
python validators/validate_payloads.py
```

The validator checks:

- required fields
- allowed top-level keys
- selector presence
- type shape for selectors
- forbidden attack metadata overrides inside `overrides`
