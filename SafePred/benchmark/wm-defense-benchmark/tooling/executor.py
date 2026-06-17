from __future__ import annotations

from typing import Any, Dict

from .executors import get_tool_executor
from .parser import parse_action_to_tool_call


def simulate_tool_execution(
    action: Any,
    state: Dict[str, Any] | None,
    env_suite: str | None = None,
) -> Dict[str, Any]:
    tool_call = parse_action_to_tool_call(action, env_suite=env_suite)
    tool_id = str(tool_call["tool_id"])
    executor = get_tool_executor(tool_id)
    return executor(tool_call, state)
