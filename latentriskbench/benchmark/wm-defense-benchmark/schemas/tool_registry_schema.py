#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_KEYS = {
    "tool_id",
    "category",
    "action_prefixes",
    "env_suites",
    "risk_level",
    "description",
}
VALID_RISK_LEVELS = {"low", "medium", "high"}


def validate_tool_spec(spec: Dict[str, Any], source: str = "<memory>") -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_KEYS - set(spec.keys()))
    if missing:
        errors.append(f"{source}: missing required keys: {', '.join(missing)}")

    for key in ("tool_id", "category", "description"):
        value = spec.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{source}: {key} must be a non-empty string")

    for key in ("action_prefixes", "env_suites"):
        value = spec.get(key)
        if not isinstance(value, list):
            errors.append(f"{source}: {key} must be a list")
        elif not all(isinstance(item, str) and item.strip() for item in value):
            errors.append(f"{source}: {key} entries must be non-empty strings")

    risk_level = spec.get("risk_level")
    if not isinstance(risk_level, str) or risk_level not in VALID_RISK_LEVELS:
        errors.append(f"{source}: risk_level must be one of: {', '.join(sorted(VALID_RISK_LEVELS))}")
    return errors


def validate_tool_registry_file(path: Path) -> List[str]:
    errors: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return [f"{path}: invalid JSON: {exc}"]

    if not isinstance(payload, dict):
        return [f"{path}: registry file must be a JSON object"]

    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return [f"{path}: tools must be a non-empty list"]

    seen_ids = set()
    for idx, spec in enumerate(tools, start=1):
        source = f"{path}.tools[{idx}]"
        if not isinstance(spec, dict):
            errors.append(f"{source}: tool spec must be an object")
            continue
        errors.extend(validate_tool_spec(spec, source=source))
        tool_id = spec.get("tool_id")
        if isinstance(tool_id, str):
            if tool_id in seen_ids:
                errors.append(f"{source}: duplicate tool_id '{tool_id}'")
            seen_ids.add(tool_id)
    return errors
