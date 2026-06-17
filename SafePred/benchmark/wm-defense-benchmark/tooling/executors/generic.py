from __future__ import annotations

from typing import Any, Dict

from .base import build_execution_result, copy_state


def execute_generic_action(tool_call: Dict[str, Any], state: Dict[str, Any] | None) -> Dict[str, Any]:
    next_state = copy_state(state)
    next_state["last_tool_id"] = str(tool_call.get("tool_id", "generic_action"))
    next_state["last_tool_target"] = str(tool_call.get("target", ""))
    if not bool(tool_call.get("parsed")):
        return build_execution_result(
            tool_call,
            next_state,
            tool_success=False,
            failure_reason="unparseable_action",
            effect="parse_failed",
        )
    return build_execution_result(tool_call, next_state, effect="generic_action_executed")
