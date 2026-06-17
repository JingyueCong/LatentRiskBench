#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


REQUIRED_KEYS = {
    "payload_id",
    "attack_id",
    "attack_family",
    "attack_depth",
    "overrides",
}

OPTIONAL_TARGET_KEYS = {
    "target_task_ids",
    "target_tags",
}

OPTIONAL_METADATA_KEYS = {
    # Populated by the WM-adaptive attack generator; records generator id,
    # WM/oracle configs, variant index, log path, and timing. Opaque to the
    # benchmark runner but required for reproducible attack auditing.
    "generation_metadata",
}


def validate_payload(payload: Dict[str, Any], source: str = "<memory>") -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_KEYS - set(payload.keys()))
    if missing:
        errors.append(f"{source}: missing required keys: {', '.join(missing)}")

    unknown = sorted(
        set(payload.keys()) - REQUIRED_KEYS - OPTIONAL_TARGET_KEYS - OPTIONAL_METADATA_KEYS
    )
    if unknown:
        errors.append(f"{source}: unknown top-level keys: {', '.join(unknown)}")

    payload_id = payload.get("payload_id")
    if not isinstance(payload_id, str) or not payload_id.strip():
        errors.append(f"{source}: payload_id must be a non-empty string")

    attack_id = payload.get("attack_id")
    if not isinstance(attack_id, str) or not attack_id.strip():
        errors.append(f"{source}: attack_id must be a non-empty string")

    attack_family = payload.get("attack_family")
    if not isinstance(attack_family, str) or not attack_family.strip():
        errors.append(f"{source}: attack_family must be a non-empty string")

    attack_depth = payload.get("attack_depth")
    if not isinstance(attack_depth, str) or not attack_depth.strip():
        errors.append(f"{source}: attack_depth must be a non-empty string")

    overrides = payload.get("overrides")
    if not isinstance(overrides, dict) or not overrides:
        errors.append(f"{source}: overrides must be a non-empty object")

    target_task_ids = payload.get("target_task_ids", [])
    target_tags = payload.get("target_tags", [])
    if target_task_ids and not isinstance(target_task_ids, list):
        errors.append(f"{source}: target_task_ids must be a list when present")
    if target_tags and not isinstance(target_tags, list):
        errors.append(f"{source}: target_tags must be a list when present")
    if not target_task_ids and not target_tags:
        errors.append(f"{source}: at least one of target_task_ids or target_tags must be provided")

    if isinstance(overrides, dict):
        if "attack_id" in overrides or "attack_family" in overrides or "attack_depth" in overrides:
            errors.append(f"{source}: overrides must not redefine attack_id/attack_family/attack_depth")

    generation_metadata = payload.get("generation_metadata")
    if generation_metadata is not None and not isinstance(generation_metadata, dict):
        errors.append(f"{source}: generation_metadata must be an object when present")

    return errors


def validate_payload_file(path: Path) -> List[str]:
    errors: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                import json

                payload = json.loads(line)
            except Exception as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"{path}:{line_no}: payload must be a JSON object")
                continue
            errors.extend(validate_payload(payload, source=f"{path}:{line_no}"))
    return errors
