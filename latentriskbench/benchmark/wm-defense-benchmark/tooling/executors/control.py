from __future__ import annotations

from typing import Any, Dict

from .base import build_execution_result, copy_state


def execute_workflow_finish(tool_call: Dict[str, Any], state: Dict[str, Any] | None) -> Dict[str, Any]:
    next_state = copy_state(state)
    next_state["last_tool_id"] = "workflow_finish"
    next_state["last_tool_target"] = str(tool_call.get("target", ""))
    next_state["workflow_status"] = "completed"
    return build_execution_result(
        tool_call,
        next_state,
        tool_done=True,
        effect="workflow_completed",
    )
