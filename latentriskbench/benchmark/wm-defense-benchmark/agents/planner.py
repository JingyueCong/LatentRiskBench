from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .base import AgentHandler
from tooling import action_risk_rank, parse_action_to_tool_call


class PlannerAgentHandler(AgentHandler):
    def __init__(self) -> None:
        super().__init__(agent_id="agent_planner")
        self._task_states: Dict[str, Dict[str, Any]] = {}
        self._unsafe_keywords = (
            "api_key",
            "token",
            "password",
            "credential",
            "id_rsa",
            "scp",
            "curl",
            "upload",
            "exfil",
            "evil.example",
        )
        self._inspect_keywords = ("review", "inspect", "check", "audit", "list", "open", "read", "look")
        self._execute_keywords = ("save", "apply", "update", "send", "submit", "patch", "run", "type")
        self._recover_keywords = ("report", "refuse", "cancel", "abort", "help", "finish", "done")

    def metadata(self) -> Dict[str, Any]:
        data = super().metadata()
        data["strategy"] = "stateful phase-based planning with readable plan steps and tool-feedback replanning"
        return data

    def reset_task(self, task_id: str, task: Optional[Dict[str, Any]] = None) -> None:
        task = task or {}
        plan_steps = self._generate_plan_steps(task)
        phases = self._build_plan(task, plan_steps)
        self._task_states[task_id] = {
            "task_id": task_id,
            "plan_steps": plan_steps,
            "phases": phases,
            "phase_index": 0,
            "plan_step_index": 0,
            "replan_count": 0,
            "history": [],
            "last_tool_success": None,
            "last_tool_id": None,
        }

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        state = self._task_states.get(task_id)
        if state is None:
            return None
        return deepcopy(state)

    def select_proposed_index(
        self,
        task: Dict[str, Any],
        candidate_actions: List[Any],
        fallback_index: int = 0,
    ) -> int:
        if not candidate_actions:
            return 0

        task_id = str(task.get("task_id", "__anonymous__"))
        if task_id not in self._task_states:
            self.reset_task(task_id, task)
        planner_state = self._task_states[task_id]
        current_phase = self._current_phase(planner_state)
        env_suite = str(task.get("env_suite", ""))

        best_idx = max(0, min(fallback_index, len(candidate_actions) - 1))
        best_score = self._score_action(candidate_actions[best_idx], current_phase, env_suite)
        for idx, action in enumerate(candidate_actions):
            score = self._score_action(action, current_phase, env_suite)
            if score < best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def observe_outcome(
        self,
        task_id: str,
        step_index: int,
        task: Dict[str, Any],
        selected_action: Any,
        tool_execution: Optional[Dict[str, Any]],
        step_result: Dict[str, Any],
        state_before: Optional[Dict[str, Any]] = None,
        state_after: Optional[Dict[str, Any]] = None,
    ) -> None:
        del state_before, state_after
        planner_state = self._task_states.get(task_id)
        if planner_state is None:
            return

        parsed = parse_action_to_tool_call(selected_action, env_suite=str(task.get("env_suite", "")))
        tool_id = parsed["tool_id"]
        tool_success = bool((tool_execution or {}).get("tool_success", step_result.get("step_success", False)))
        current_phase = self._current_phase(planner_state)

        if step_result.get("requires_regeneration") or step_result.get("unsafe_selected") or step_result.get("violation_selected"):
            self._set_phase(planner_state, "recover")
            planner_state["replan_count"] += 1
        elif not tool_success:
            self._set_phase(planner_state, "recover")
            planner_state["replan_count"] += 1
        elif tool_id == "security_report":
            self._set_phase(planner_state, "finish")
        elif tool_id == "workflow_finish":
            self._set_phase(planner_state, "finish")
        elif current_phase == "inspect":
            self._advance_phase(planner_state)
        elif current_phase == "verify":
            self._advance_phase(planner_state)
        elif current_phase == "execute":
            self._set_phase(planner_state, "finish")
        elif current_phase == "recover":
            self._set_phase(planner_state, "report")
        elif current_phase == "report":
            self._set_phase(planner_state, "finish")

        planner_state["last_tool_success"] = tool_success
        planner_state["last_tool_id"] = tool_id
        planner_state["history"].append(
            {
                "step_index": step_index,
                "phase_before": current_phase,
                "phase_after": self._current_phase(planner_state),
                "plan_step_before": self._plan_step_text_for_phase(current_phase),
                "plan_step_after": self._current_plan_step(planner_state),
                "selected_action": str(selected_action),
                "tool_id": tool_id,
                "tool_success": tool_success,
                "replanned": planner_state["replan_count"],
            }
        )

    def _generate_plan_steps(self, task: Dict[str, Any]) -> List[str]:
        return self._default_plan_steps(task)

    def _default_plan_steps(self, task: Dict[str, Any]) -> List[str]:
        tags = {str(tag).lower() for tag in task.get("tags", [])}
        plan_steps = ["Inspect the current state and identify the safest useful next move."]
        if tags & {"finance", "cloud", "config", "support", "backup"}:
            plan_steps.append("Verify the request and prefer safe escalation if anything looks suspicious.")
        plan_steps.append("Execute the lowest-risk action that still advances the user goal.")
        plan_steps.append("Finish safely once the task is complete or escalate if the task remains risky.")
        return plan_steps

    def _build_plan(self, task: Dict[str, Any], plan_steps: Optional[List[str]] = None) -> List[str]:
        del task
        source_steps = plan_steps or []
        phases: List[str] = []
        for step in source_steps:
            phase = self._classify_phase(str(step))
            if not phases or phases[-1] != phase:
                phases.append(phase)
        if not phases:
            phases = ["inspect", "execute", "finish"]
        if phases[-1] != "finish":
            phases.append("finish")
        return phases

    def _classify_phase(self, step_text: str) -> str:
        text = step_text.lower()
        if any(token in text for token in ("recover", "rollback", "undo")):
            return "recover"
        if any(token in text for token in ("report", "escalate", "notify", "alert")):
            return "report"
        if any(token in text for token in ("finish", "complete", "done", "close")):
            return "finish"
        if any(token in text for token in ("verify", "validate", "confirm", "cross-check")):
            return "verify"
        if any(token in text for token in ("execute", "apply", "update", "submit", "send", "save", "run")):
            return "execute"
        return "inspect"

    def _current_phase(self, planner_state: Dict[str, Any]) -> str:
        phases = planner_state["phases"]
        index = max(0, min(int(planner_state["phase_index"]), len(phases) - 1))
        return str(phases[index])

    def _current_plan_step(self, planner_state: Dict[str, Any]) -> Optional[str]:
        plan_steps = planner_state.get("plan_steps") or []
        if not plan_steps:
            return None
        index = max(0, min(int(planner_state.get("plan_step_index", 0)), len(plan_steps) - 1))
        return str(plan_steps[index])

    def _set_phase(self, planner_state: Dict[str, Any], phase_name: str) -> None:
        phases = planner_state["phases"]
        if phase_name not in phases:
            if phase_name == "recover":
                insert_at = max(0, len(phases) - 1)
                phases.insert(insert_at, "recover")
                if "report" not in phases:
                    phases.insert(min(insert_at + 1, len(phases)), "report")
            elif phase_name == "report":
                insert_at = max(0, len(phases) - 1)
                phases.insert(insert_at, "report")
            elif phase_name == "finish":
                phases.append("finish")
            self._sync_plan_steps_from_phases(planner_state)
        planner_state["phase_index"] = phases.index(phase_name)
        planner_state["plan_step_index"] = min(int(planner_state["phase_index"]), max(len(planner_state.get("plan_steps", [])) - 1, 0))

    def _advance_phase(self, planner_state: Dict[str, Any]) -> None:
        planner_state["phase_index"] = min(int(planner_state["phase_index"]) + 1, len(planner_state["phases"]) - 1)
        planner_state["plan_step_index"] = min(int(planner_state["phase_index"]), max(len(planner_state.get("plan_steps", [])) - 1, 0))

    def _sync_plan_steps_from_phases(self, planner_state: Dict[str, Any]) -> None:
        planner_state["plan_steps"] = [self._plan_step_text_for_phase(phase) for phase in planner_state["phases"]]
        planner_state["plan_step_index"] = min(int(planner_state.get("phase_index", 0)), max(len(planner_state["plan_steps"]) - 1, 0))

    def _plan_step_text_for_phase(self, phase: str) -> str:
        if phase == "inspect":
            return "Inspect the current state and gather context before acting."
        if phase == "verify":
            return "Verify the request and check for safety issues before proceeding."
        if phase == "execute":
            return "Execute the lowest-risk useful action for the current goal."
        if phase == "recover":
            return "Recover from a risky or failed action by switching to a safer branch."
        if phase == "report":
            return "Report or escalate the issue through a safe channel."
        if phase == "finish":
            return "Finish the task safely once no further action is needed."
        return "Proceed with the next safe step."

    def _score_action(self, action: Any, current_phase: str, env_suite: str) -> Tuple[int, int, int, int, int]:
        text = str(action).lower()
        parsed = parse_action_to_tool_call(action, env_suite=env_suite)
        tool_id = parsed["tool_id"]
        risk_rank = action_risk_rank(action, env_suite=env_suite)
        unsafe_hits = sum(1 for token in self._unsafe_keywords if token in text)
        phase_penalty = self._phase_penalty(current_phase, tool_id, text)
        safe_bias = -self._safe_bias(tool_id, text)
        return (phase_penalty, risk_rank, unsafe_hits, safe_bias, len(text))

    def _phase_penalty(self, phase: str, tool_id: str, text: str) -> int:
        if phase == "inspect":
            if tool_id == "security_report":
                return 2
            if tool_id in {"browser_click", "generic_action"} and any(token in text for token in self._inspect_keywords):
                return 0
            if tool_id == "workflow_finish":
                return 5
            return 2

        if phase == "verify":
            if tool_id == "security_report":
                return 0
            if any(token in text for token in self._inspect_keywords):
                return 1
            if tool_id == "workflow_finish":
                return 4
            return 2

        if phase == "execute":
            if tool_id == "security_report":
                return 2
            if any(token in text for token in self._execute_keywords):
                return 0
            if tool_id == "workflow_finish":
                return 3
            return 1

        if phase == "recover":
            if tool_id == "security_report":
                return 0
            if tool_id == "workflow_finish":
                return 1
            if any(token in text for token in self._recover_keywords):
                return 1
            return 3

        if phase == "report":
            if tool_id == "security_report":
                return 0
            if tool_id == "workflow_finish":
                return 1
            return 3

        if phase == "finish":
            if tool_id == "workflow_finish":
                return 0
            if tool_id == "security_report":
                return 1
            return 4

        return 2

    def _safe_bias(self, tool_id: str, text: str) -> int:
        score = 0
        if tool_id == "security_report":
            score += 3
        if tool_id == "workflow_finish":
            score += 2
        if any(token in text for token in self._recover_keywords):
            score += 1
        return score
