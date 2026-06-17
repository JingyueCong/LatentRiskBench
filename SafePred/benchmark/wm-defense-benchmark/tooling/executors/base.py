from __future__ import annotations

from typing import Any, Dict


def copy_state(state: Dict[str, Any] | None) -> Dict[str, Any]:
    if isinstance(state, dict):
        return dict(state)
    return {}


def build_execution_result(
    tool_call: Dict[str, Any],
    next_state: Dict[str, Any],
    *,
    tool_success: bool = True,
    tool_done: bool = False,
    failure_reason: str | None = None,
    effect: str = "no_op",
) -> Dict[str, Any]:
    next_state["last_tool_success"] = tool_success
    if failure_reason is not None:
        next_state["last_tool_failure_reason"] = failure_reason
    return {
        "tool_call": tool_call,
        "tool_success": tool_success,
        "tool_done": tool_done,
        "failure_reason": failure_reason,
        "effect": effect,
        "simulated_next_state": next_state,
    }
