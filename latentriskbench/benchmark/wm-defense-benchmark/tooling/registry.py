from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from .base import ToolSpec


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
TOOL_REGISTRY_PATH = BENCHMARK_ROOT / "data" / "tool_registry.json"


def _normalize_spec(raw: Dict[str, object]) -> ToolSpec:
    return ToolSpec(
        tool_id=str(raw["tool_id"]),
        category=str(raw["category"]),
        action_prefixes=[str(v) for v in raw.get("action_prefixes", [])],
        env_suites=[str(v) for v in raw.get("env_suites", [])],
        risk_level=str(raw["risk_level"]),
        description=str(raw["description"]),
    )


@lru_cache(maxsize=1)
def load_tool_registry() -> Dict[str, ToolSpec]:
    with TOOL_REGISTRY_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tools = payload.get("tools", [])
    registry: Dict[str, ToolSpec] = {}
    for item in tools:
        if not isinstance(item, dict):
            continue
        spec = _normalize_spec(item)
        registry[spec.tool_id] = spec
    return registry


def get_tool_spec(tool_id: str) -> ToolSpec:
    registry = load_tool_registry()
    if tool_id not in registry:
        raise KeyError(f"Unknown tool_id: {tool_id}")
    return registry[tool_id]


def list_tool_specs() -> List[ToolSpec]:
    return list(load_tool_registry().values())


def list_tool_metadata() -> List[Dict[str, object]]:
    return [tool.metadata() for tool in list_tool_specs()]
