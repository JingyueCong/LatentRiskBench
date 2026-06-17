#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


REQUIRED_ATTACK_METADATA = {
    "task_id",
    "attack_id",
    "attack_family",
    "attack_depth",
    "env_suite",
    "orig_goal_success",
    "intent",
}

OPTIONAL_COMMON_KEYS = {
    "agent_id",
    "defense_id",
    "seed",
    "skills",
}

OPTIONAL_OFFLINE_KEYS = {
    "action_history",
}

# Scoped enterprise policy classes for irreversible-harm annotation.
# Maps to the three high-severity classes the benchmark studies:
#   - "data_leakage": credential or sensitive-data leakage (e.g. typing a password
#     into a phishing page, exfiltrating a config/token to an external host).
#   - "phishing_propagation": phishing / social-engineering propagation
#     (e.g. forwarding or sending an attacker-supplied message/link onward).
#   - "goal_hijacking": adopting an attacker-controlled goal via untrusted
#     instructions (e.g. executing an injected command).
POLICY_CLASSES = {"data_leakage", "phishing_propagation", "goal_hijacking"}

# Irreversibility / early-warning annotation fields (online tasks only).
# Step indices are 1-based, matching the online runner's enumerate(start=1).
#   - policy_class: which scoped policy class the harm belongs to (see POLICY_CLASSES).
#   - harm_event: natural-language description of the irreversible harm.
#   - irreversible_by_construction: True if the harm cannot be undone once the
#     harm action executes (credential sent, message delivered, command run).
#   - irreversible_harm_step: 1-based step at which irreversible harm occurs on
#     the unsafe path. That step must expose an unsafe action.
#   - earliest_warning_step: 1-based earliest step at which the risk is observable,
#     i.e. the first step a perfect predictive defense could warn. Must be
#     <= irreversible_harm_step. The warning window = harm_step - warning_step is
#     the maximum lead time any defense can achieve on this trajectory.
#   - delayed_risk: True if this trajectory belongs to the delayed-risk subset
#     (irreversible harm with a positive warning window) used for lead-time claims.
OPTIONAL_ONLINE_KEYS = {
    "policy_class",
    "harm_event",
    "irreversible_by_construction",
    "irreversible_harm_step",
    "earliest_warning_step",
    "delayed_risk",
}

REQUIRED_OFFLINE_KEYS = REQUIRED_ATTACK_METADATA | {
    "state",
    "candidate_actions",
    "unsafe_action_indices",
    "proposed_action_index",
    "current_response",
}

REQUIRED_ONLINE_KEYS = REQUIRED_ATTACK_METADATA | {
    "initial_state",
    "steps",
}

STEP_REQUIRED_KEYS = {
    "candidate_actions",
    "unsafe_action_indices",
    "proposed_action_index",
    "current_response",
}

STEP_OPTIONAL_KEYS = {
    "state",
    "next_state_by_action",
    "next_state",
    "done_actions",
    "done",
}


def _validate_common_explicit_task(task: Dict[str, Any], source: str) -> List[str]:
    errors: List[str] = []

    for key in ["task_id", "attack_id", "attack_family", "attack_depth", "env_suite", "intent"]:
        value = task.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{source}: {key} must be a non-empty string")

    if not isinstance(task.get("orig_goal_success"), bool):
        errors.append(f"{source}: orig_goal_success must be a boolean")

    seed = task.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append(f"{source}: seed must be an integer when present")

    for key in ["agent_id", "defense_id"]:
        value = task.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{source}: {key} must be a non-empty string when present")

    skills = task.get("skills")
    if skills is not None and (
        not isinstance(skills, list)
        or not all(isinstance(skill, str) and skill.strip() for skill in skills)
    ):
        errors.append(f"{source}: skills must be a list of non-empty strings when present")

    return errors


def _validate_candidate_block(
    candidate_actions: Any,
    unsafe_action_indices: Any,
    proposed_action_index: Any,
    source: str,
) -> List[str]:
    errors: List[str] = []

    if not isinstance(candidate_actions, list) or not candidate_actions:
        errors.append(f"{source}: candidate_actions must be a non-empty list")
    elif not all(isinstance(action, str) and action.strip() for action in candidate_actions):
        errors.append(f"{source}: candidate_actions entries must be non-empty strings")

    if not isinstance(unsafe_action_indices, list):
        errors.append(f"{source}: unsafe_action_indices must be a list")
    else:
        for idx in unsafe_action_indices:
            if not isinstance(idx, int):
                errors.append(f"{source}: unsafe_action_indices entries must be integers")
                break

    if not isinstance(proposed_action_index, int):
        errors.append(f"{source}: proposed_action_index must be an integer")

    if isinstance(candidate_actions, list) and candidate_actions and isinstance(proposed_action_index, int):
        if proposed_action_index < 0 or proposed_action_index >= len(candidate_actions):
            errors.append(f"{source}: proposed_action_index out of range for candidate_actions")

    if isinstance(candidate_actions, list) and isinstance(unsafe_action_indices, list):
        max_index = len(candidate_actions) - 1
        for idx in unsafe_action_indices:
            if isinstance(idx, int) and (idx < 0 or idx > max_index):
                errors.append(f"{source}: unsafe_action_indices contains out-of-range index {idx}")
                break

    return errors


def validate_offline_task(task: Dict[str, Any], source: str = "<memory>") -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_OFFLINE_KEYS - set(task.keys()))
    if missing:
        errors.append(f"{source}: missing required keys: {', '.join(missing)}")

    unknown = sorted(set(task.keys()) - REQUIRED_OFFLINE_KEYS - OPTIONAL_COMMON_KEYS - OPTIONAL_OFFLINE_KEYS)
    if unknown:
        errors.append(f"{source}: unknown top-level keys: {', '.join(unknown)}")

    errors.extend(_validate_common_explicit_task(task, source))

    if not isinstance(task.get("state"), dict):
        errors.append(f"{source}: state must be an object")

    current_response = task.get("current_response")
    if not isinstance(current_response, str) or not current_response.strip():
        errors.append(f"{source}: current_response must be a non-empty string")

    action_history = task.get("action_history", [])
    if action_history and (
        not isinstance(action_history, list)
        or not all(isinstance(entry, str) and entry.strip() for entry in action_history)
    ):
        errors.append(f"{source}: action_history must be a list of non-empty strings when present")

    errors.extend(
        _validate_candidate_block(
            task.get("candidate_actions"),
            task.get("unsafe_action_indices"),
            task.get("proposed_action_index"),
            source,
        )
    )
    return errors


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_irreversibility(
    task: Dict[str, Any], n_steps: int, steps: Any, source: str
) -> List[str]:
    """Validate the optional irreversibility / early-warning annotation block.

    Only fields that are present are checked, so existing tasks without the
    block remain valid. Step indices are 1-based.
    """
    errors: List[str] = []

    policy_class = task.get("policy_class")
    if policy_class is not None and policy_class not in POLICY_CLASSES:
        errors.append(
            f"{source}: policy_class must be one of {sorted(POLICY_CLASSES)} when present"
        )

    harm_event = task.get("harm_event")
    if harm_event is not None and (not isinstance(harm_event, str) or not harm_event.strip()):
        errors.append(f"{source}: harm_event must be a non-empty string when present")

    for bool_key in ("irreversible_by_construction", "delayed_risk"):
        value = task.get(bool_key)
        if value is not None and not isinstance(value, bool):
            errors.append(f"{source}: {bool_key} must be a boolean when present")

    harm_step = task.get("irreversible_harm_step")
    warning_step = task.get("earliest_warning_step")
    for step_key, step_val in (
        ("irreversible_harm_step", harm_step),
        ("earliest_warning_step", warning_step),
    ):
        if step_val is None:
            continue
        if not _is_int(step_val):
            errors.append(f"{source}: {step_key} must be a 1-based integer step index when present")
        elif n_steps and (step_val < 1 or step_val > n_steps):
            errors.append(f"{source}: {step_key}={step_val} out of range for {n_steps} steps")

    if _is_int(harm_step) and _is_int(warning_step) and warning_step > harm_step:
        errors.append(
            f"{source}: earliest_warning_step ({warning_step}) must be "
            f"<= irreversible_harm_step ({harm_step})"
        )

    # The harm step must actually expose an unsafe action on the unsafe path.
    if _is_int(harm_step) and isinstance(steps, list) and 1 <= harm_step <= n_steps:
        harm_step_obj = steps[harm_step - 1]
        if isinstance(harm_step_obj, dict):
            unsafe = harm_step_obj.get("unsafe_action_indices")
            if isinstance(unsafe, list) and not unsafe:
                errors.append(
                    f"{source}: irreversible_harm_step={harm_step} but that step "
                    f"lists no unsafe_action_indices"
                )

    return errors


def validate_online_task(task: Dict[str, Any], source: str = "<memory>") -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_ONLINE_KEYS - set(task.keys()))
    if missing:
        errors.append(f"{source}: missing required keys: {', '.join(missing)}")

    unknown = sorted(
        set(task.keys()) - REQUIRED_ONLINE_KEYS - OPTIONAL_COMMON_KEYS - OPTIONAL_ONLINE_KEYS
    )
    if unknown:
        errors.append(f"{source}: unknown top-level keys: {', '.join(unknown)}")

    errors.extend(_validate_common_explicit_task(task, source))

    if not isinstance(task.get("initial_state"), dict):
        errors.append(f"{source}: initial_state must be an object")

    steps = task.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append(f"{source}: steps must be a non-empty list")
        return errors

    errors.extend(_validate_irreversibility(task, len(steps), steps, source))

    for idx, step in enumerate(steps, start=1):
        step_source = f"{source}.steps[{idx}]"
        if not isinstance(step, dict):
            errors.append(f"{step_source}: step must be an object")
            continue

        missing_step = sorted(STEP_REQUIRED_KEYS - set(step.keys()))
        if missing_step:
            errors.append(f"{step_source}: missing required keys: {', '.join(missing_step)}")

        unknown_step = sorted(set(step.keys()) - STEP_REQUIRED_KEYS - STEP_OPTIONAL_KEYS)
        if unknown_step:
            errors.append(f"{step_source}: unknown step keys: {', '.join(unknown_step)}")

        current_response = step.get("current_response")
        if not isinstance(current_response, str) or not current_response.strip():
            errors.append(f"{step_source}: current_response must be a non-empty string")

        if "state" in step and not isinstance(step.get("state"), dict):
            errors.append(f"{step_source}: state must be an object when present")
        if "next_state" in step and not isinstance(step.get("next_state"), dict):
            errors.append(f"{step_source}: next_state must be an object when present")

        next_state_by_action = step.get("next_state_by_action")
        if next_state_by_action is not None:
            if not isinstance(next_state_by_action, dict):
                errors.append(f"{step_source}: next_state_by_action must be an object when present")
            else:
                for action, state in next_state_by_action.items():
                    if not isinstance(action, str) or not action.strip():
                        errors.append(f"{step_source}: next_state_by_action keys must be non-empty strings")
                        break
                    if not isinstance(state, dict):
                        errors.append(f"{step_source}: next_state_by_action values must be objects")
                        break

        done_actions = step.get("done_actions", [])
        if done_actions and (
            not isinstance(done_actions, list)
            or not all(isinstance(action, str) and action.strip() for action in done_actions)
        ):
            errors.append(f"{step_source}: done_actions must be a list of non-empty strings when present")

        done = step.get("done")
        if done is not None and not isinstance(done, bool):
            errors.append(f"{step_source}: done must be a boolean when present")

        errors.extend(
            _validate_candidate_block(
                step.get("candidate_actions"),
                step.get("unsafe_action_indices"),
                step.get("proposed_action_index"),
                step_source,
            )
        )

    return errors


def _validate_file(path: Path, validator) -> List[str]:
    errors: List[str] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                import json

                task = json.loads(line)
            except Exception as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(task, dict):
                errors.append(f"{path}:{line_no}: task must be a JSON object")
                continue
            errors.extend(validator(task, source=f"{path}:{line_no}"))
            task_id = task.get("task_id")
            if isinstance(task_id, str):
                if task_id in seen_ids:
                    errors.append(f"{path}:{line_no}: duplicate task_id '{task_id}'")
                seen_ids.add(task_id)
    return errors


def validate_offline_task_file(path: Path) -> List[str]:
    return _validate_file(path, validate_offline_task)


def validate_online_task_file(path: Path) -> List[str]:
    return _validate_file(path, validate_online_task)
