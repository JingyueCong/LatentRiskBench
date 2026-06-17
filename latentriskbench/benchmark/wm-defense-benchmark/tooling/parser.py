from __future__ import annotations

import re
from typing import Any, Dict

from .registry import get_tool_spec


ACTION_RE = re.compile(r'^(?P<verb>[a-zA-Z_]+)\s+\[(?P<target>[^\]]+)\](?:\s+"(?P<argument>.*)")?$')
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _infer_tool_id(verb: str, target: str, argument: str, env_suite: str | None) -> str:
    verb = verb.lower()
    target = target.lower()
    argument = argument.lower()
    env_suite = (env_suite or "").lower()

    if verb == "finish":
        return "workflow_finish"
    if verb == "click":
        if "report" in target or "phishing" in target:
            return "security_report"
        return "browser_click"
    if verb == "type":
        if target == "terminal" or env_suite == "os":
            return "terminal_command"
        if any(token in target for token in ("credential", "password", "api_key", "secret")):
            return "sensitive_form_input"
        if any(token in argument for token in ("password", "api_key", "token", "secret")):
            return "sensitive_form_input"
        return "browser_type"
    return "generic_action"


def parse_action_to_tool_call(action: Any, env_suite: str | None = None) -> Dict[str, Any]:
    raw_action = str(action).strip()
    match = ACTION_RE.match(raw_action)
    if match:
        verb = match.group("verb")
        target = match.group("target")
        argument = match.group("argument") or ""
    else:
        verb = "unknown"
        target = ""
        argument = ""

    tool_id = _infer_tool_id(verb, target, argument, env_suite)
    spec = get_tool_spec(tool_id)
    return {
        "raw_action": raw_action,
        "parsed": bool(match),
        "verb": verb,
        "target": target,
        "argument": argument,
        "tool_id": tool_id,
        "tool_category": spec.category,
        "tool_risk_level": spec.risk_level,
        "tool_description": spec.description,
    }


def action_risk_rank(action: Any, env_suite: str | None = None) -> int:
    parsed = parse_action_to_tool_call(action, env_suite=env_suite)
    return RISK_ORDER.get(str(parsed["tool_risk_level"]), 1)
