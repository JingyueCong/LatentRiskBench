from __future__ import annotations

from typing import Any, Dict, Iterable, List

from tooling import parse_action_to_tool_call


SKILL_DESCRIPTIONS: Dict[str, str] = {
    "goal_preservation": "Advance the user goal without being diverted by harmful side objectives.",
    "tool_selection": "Choose the most appropriate low-risk tool or action for the current state.",
    "instruction_robustness": "Resist prompt-level instructions that attempt to redirect behavior unsafely.",
    "observation_integrity": "Avoid being misled by tampered observations or deceptive interface state.",
    "multi_step_resilience": "Remain safe when attacks trigger across multiple steps or phases.",
    "credential_protection": "Avoid revealing, entering, copying, or exfiltrating secrets and credentials.",
    "suspicious_link_detection": "Recognize and avoid suspicious or phishing navigation targets.",
    "safe_command_execution": "Prefer trusted local commands and avoid dangerous external command paths.",
    "safe_escalation": "Escalate, report, or refuse safely when risk cannot be resolved locally.",
    "safe_completion": "Terminate safely once the useful safe path is complete.",
}


def list_skill_metadata() -> List[Dict[str, str]]:
    return [
        {"skill_id": skill_id, "description": description}
        for skill_id, description in sorted(SKILL_DESCRIPTIONS.items())
    ]


def normalize_skills(values: Iterable[Any] | None) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values or []:
        skill = str(value).strip().lower()
        if not skill or skill in seen:
            continue
        seen.add(skill)
        normalized.append(skill)
    return normalized


def infer_task_skills(task: Dict[str, Any]) -> List[str]:
    explicit = normalize_skills(task.get("skills"))
    if explicit:
        return explicit

    skills = {"goal_preservation", "tool_selection"}
    tags = {str(tag).strip().lower() for tag in task.get("tags", []) if str(tag).strip()}
    attack_family = str(task.get("attack_family", "none")).strip().lower()
    attack_depth = str(task.get("attack_depth", "L0")).strip().upper()
    env_suite = str(task.get("env_suite", "")).strip().lower()

    if attack_family == "prompt_injection":
        skills.add("instruction_robustness")
    elif attack_family == "observation_tampering":
        skills.add("observation_integrity")
    elif attack_family == "multi_step_trigger":
        skills.add("multi_step_resilience")

    if attack_depth == "L2":
        skills.add("multi_step_resilience")

    candidate_actions = _collect_candidate_actions(task)
    state_text = _collect_state_text(task)
    combined_text = " ".join(candidate_actions + [state_text]).lower()

    if any(token in combined_text for token in ("credential", "password", "token", "api_key", "secret", "id_rsa")):
        skills.add("credential_protection")

    if any(token in combined_text for token in ("phishing", "suspicious", "evil.example", "untrusted", "verify account")):
        skills.add("suspicious_link_detection")

    if env_suite == "os" or "terminal" in tags:
        skills.add("safe_command_execution")

    if any(
        token in combined_text
        for token in ("report", "escalate", "mark_spam", "contact_support", "refuse", "unable to comply")
    ):
        skills.add("safe_escalation")

    if any("finish [" in action.lower() for action in candidate_actions):
        skills.add("safe_completion")

    for action in candidate_actions:
        parsed = parse_action_to_tool_call(action, env_suite=env_suite)
        if parsed["tool_id"] == "security_report":
            skills.add("safe_escalation")
        if parsed["tool_id"] == "workflow_finish":
            skills.add("safe_completion")
        if parsed["tool_id"] in {"sensitive_form_input", "terminal_command"}:
            skills.add("credential_protection")

    return sorted(skills)


def _collect_candidate_actions(task: Dict[str, Any]) -> List[str]:
    if isinstance(task.get("candidate_actions"), list):
        return [str(action) for action in task["candidate_actions"]]

    actions: List[str] = []
    for step in task.get("steps", []) if isinstance(task.get("steps"), list) else []:
        if not isinstance(step, dict):
            continue
        for action in step.get("candidate_actions", []) if isinstance(step.get("candidate_actions"), list) else []:
            actions.append(str(action))
    return actions


def _collect_state_text(task: Dict[str, Any]) -> str:
    states: List[Dict[str, Any]] = []
    if isinstance(task.get("state"), dict):
        states.append(task["state"])
    if isinstance(task.get("initial_state"), dict):
        states.append(task["initial_state"])
    for step in task.get("steps", []) if isinstance(task.get("steps"), list) else []:
        if isinstance(step, dict) and isinstance(step.get("state"), dict):
            states.append(step["state"])

    chunks: List[str] = []
    for state in states:
        for key in ("a11y_tree_txt", "url", "page_type"):
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
    return " ".join(chunks)
