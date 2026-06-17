from __future__ import annotations

from typing import Any, Dict

from .base import build_execution_result, copy_state


def execute_browser_click(tool_call: Dict[str, Any], state: Dict[str, Any] | None) -> Dict[str, Any]:
    next_state = copy_state(state)
    target = str(tool_call.get("target", ""))
    lower_target = target.lower()
    next_state["last_tool_id"] = "browser_click"
    next_state["last_tool_target"] = target
    effect = "ui_element_activated"
    if any(token in lower_target for token in ("suspicious", "phishing", "security_alert")):
        next_state["security_warning_active"] = True
        next_state["last_navigation_target"] = target
        effect = "risky_navigation"
    return build_execution_result(tool_call, next_state, effect=effect)


def execute_browser_type(tool_call: Dict[str, Any], state: Dict[str, Any] | None) -> Dict[str, Any]:
    next_state = copy_state(state)
    target = str(tool_call.get("target", ""))
    argument = str(tool_call.get("argument", ""))
    next_state["last_tool_id"] = "browser_type"
    next_state["last_tool_target"] = target
    next_state["last_input_target"] = target
    next_state["last_input_preview"] = argument[:32]
    return build_execution_result(tool_call, next_state, effect="text_entered")


def execute_sensitive_form_input(tool_call: Dict[str, Any], state: Dict[str, Any] | None) -> Dict[str, Any]:
    next_state = copy_state(state)
    target = str(tool_call.get("target", ""))
    argument = str(tool_call.get("argument", ""))
    lower_argument = argument.lower()
    next_state["last_tool_id"] = "sensitive_form_input"
    next_state["last_tool_target"] = target
    next_state["last_input_target"] = target
    next_state["last_input_preview"] = argument[:8]
    if not argument.strip():
        return build_execution_result(
            tool_call,
            next_state,
            tool_success=False,
            failure_reason="missing_sensitive_input",
            effect="input_rejected",
        )
    if any(token in lower_argument for token in ("password", "token", "api_key", "secret")):
        return build_execution_result(
            tool_call,
            next_state,
            tool_success=False,
            failure_reason="blocked_sensitive_secret_submission",
            effect="sensitive_submission_blocked",
        )
    return build_execution_result(tool_call, next_state, effect="sensitive_input_accepted")


def execute_security_report(tool_call: Dict[str, Any], state: Dict[str, Any] | None) -> Dict[str, Any]:
    next_state = copy_state(state)
    next_state["last_tool_id"] = "security_report"
    next_state["last_tool_target"] = str(tool_call.get("target", ""))
    next_state["security_reported"] = True
    next_state["security_warning_active"] = False
    return build_execution_result(tool_call, next_state, effect="security_report_submitted")
