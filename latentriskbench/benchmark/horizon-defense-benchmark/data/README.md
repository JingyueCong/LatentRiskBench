# Compositional-Dynamics (CD) datasets

State-conditioned / compositional-dynamics audit datasets. **Data only** — the
horizon-defense-benchmark runner code is not included in this repository.

## Files

| File | Records | Description |
|------|---------|-------------|
| `generated_cd_tasks.jsonl` | 100 | Generic CD audit tasks (25 per mechanism: state_accumulation / destination_dependent / authorization_dependent / goal_shift) |
| `generated_cd_tasks_ecommerce.jsonl` | 40 | E-commerce enterprise CD audit tasks |
| `policies_augmented.json` | — | Policy definitions used by the CD tasks |
| `policies_augmented_scoped.json` | — | Scoped variant of the augmented policies |

## Record schema (self-labeled)

Each task is a state-conditioned audit instance with a matched benign twin:

- `risk_type`, `cd_mechanism` — e.g. `state_accumulation`, destination/authorization/goal-shift
- `enterprise_scenario`, `user_goal`
- `trajectory_prefix` — the committed prefix `h_<T`
- `triggering_action` — the terminal action under audit `a_T`
- `isolated_action_label` — label under the action-local view (often `benign`)
- `contextual_label` — label under the reconstructed-state view (e.g. `policy_violation`)
- `policy_violation`, `required_context`, `policy_domain`, `severity`
- `benign_twin` — `{trajectory_prefix, triggering_action, why_benign}`: same terminal action,
  prefix changed so the action becomes policy-compliant
- `expected_detector_behavior`

The `isolated_action_label` vs `contextual_label` split, plus `benign_twin`, is exactly the
constant-action / state-conditioned design: the label flips through the reconstructed state,
not the action surface.
