from __future__ import annotations

from typing import Any, Callable, Dict, List

from tooling.executors.browser import (
    execute_browser_click,
    execute_browser_type,
    execute_security_report,
    execute_sensitive_form_input,
)
from tooling.executors.control import execute_workflow_finish
from tooling.executors.generic import execute_generic_action
from tooling.executors.system import execute_terminal_command


ToolExecutor = Callable[[Dict[str, Any], Dict[str, Any] | None], Dict[str, Any]]


_EXECUTOR_REGISTRY: Dict[str, ToolExecutor] = {
    "browser_click": execute_browser_click,
    "browser_type": execute_browser_type,
    "terminal_command": execute_terminal_command,
    "sensitive_form_input": execute_sensitive_form_input,
    "security_report": execute_security_report,
    "workflow_finish": execute_workflow_finish,
    "generic_action": execute_generic_action,
}


def get_tool_executor(tool_id: str) -> ToolExecutor:
    return _EXECUTOR_REGISTRY.get(tool_id, _EXECUTOR_REGISTRY["generic_action"])


def list_tool_executor_ids() -> List[str]:
    return list(_EXECUTOR_REGISTRY.keys())


def list_tool_executors() -> Dict[str, ToolExecutor]:
    return dict(_EXECUTOR_REGISTRY)
