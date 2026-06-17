#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


REQUIRED_KEYS = {
    "task_id",
    "intent",
    "env_suite",
    "orig_goal_success",
    "state",
    "candidate_actions",
    "unsafe_action_indices",
    "proposed_action_index",
    "current_response",
}

OPTIONAL_KEYS = {
    "tags",
    "skills",
    "action_history",
}

FORBIDDEN_ATTACK_KEYS = {
    "attack_id",
    "attack_family",
    "attack_depth",
}


def validate_base_task(task: Dict[str, Any], source: str = "<memory>") -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_KEYS - set(task.keys()))
    if missing:
        errors.append(f"{source}: missing required keys: {', '.join(missing)}")

    unknown = sorted(set(task.keys()) - REQUIRED_KEYS - OPTIONAL_KEYS)
    if unknown:
        errors.append(f"{source}: unknown top-level keys: {', '.join(unknown)}")

    forbidden = sorted(FORBIDDEN_ATTACK_KEYS.intersection(task.keys()))
    if forbidden:
        errors.append(f"{source}: base task must not define attack metadata: {', '.join(forbidden)}")

    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        errors.append(f"{source}: task_id must be a non-empty string")

    intent = task.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        errors.append(f"{source}: intent must be a non-empty string")

    env_suite = task.get("env_suite")
    if not isinstance(env_suite, str) or not env_suite.strip():
        errors.append(f"{source}: env_suite must be a non-empty string")

    if not isinstance(task.get("orig_goal_success"), bool):
        errors.append(f"{source}: orig_goal_success must be a boolean")

    state = task.get("state")
    if not isinstance(state, dict):
        errors.append(f"{source}: state must be an object")

    candidate_actions = task.get("candidate_actions")
    if not isinstance(candidate_actions, list) or not candidate_actions:
        errors.append(f"{source}: candidate_actions must be a non-empty list")
    elif not all(isinstance(action, str) and action.strip() for action in candidate_actions):
        errors.append(f"{source}: candidate_actions entries must be non-empty strings")

    unsafe_action_indices = task.get("unsafe_action_indices")
    if not isinstance(unsafe_action_indices, list):
        errors.append(f"{source}: unsafe_action_indices must be a list")
    else:
        for idx in unsafe_action_indices:
            if not isinstance(idx, int):
                errors.append(f"{source}: unsafe_action_indices entries must be integers")
                break

    proposed_action_index = task.get("proposed_action_index")
    if not isinstance(proposed_action_index, int):
        errors.append(f"{source}: proposed_action_index must be an integer")

    current_response = task.get("current_response")
    if not isinstance(current_response, str) or not current_response.strip():
        errors.append(f"{source}: current_response must be a non-empty string")

    tags = task.get("tags", [])
    if tags and (not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags)):
        errors.append(f"{source}: tags must be a list of non-empty strings when present")

    skills = task.get("skills", [])
    if skills and (not isinstance(skills, list) or not all(isinstance(skill, str) and skill.strip() for skill in skills)):
        errors.append(f"{source}: skills must be a list of non-empty strings when present")

    action_history = task.get("action_history", [])
    if action_history and (
        not isinstance(action_history, list)
        or not all(isinstance(entry, str) and entry.strip() for entry in action_history)
    ):
        errors.append(f"{source}: action_history must be a list of non-empty strings when present")

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


def validate_base_task_file(path: Path) -> List[str]:
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
                errors.append(f"{path}:{line_no}: base task must be a JSON object")
                continue
            task_errors = validate_base_task(task, source=f"{path}:{line_no}")
            errors.extend(task_errors)
            task_id = task.get("task_id")
            if isinstance(task_id, str):
                if task_id in seen_ids:
                    errors.append(f"{path}:{line_no}: duplicate task_id '{task_id}'")
                seen_ids.add(task_id)
    return errors
