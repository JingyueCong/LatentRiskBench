from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AgentHandler:
    agent_id: str

    def metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
        }

    def select_proposed_index(
        self,
        task: Dict[str, Any],
        candidate_actions: List[Any],
        fallback_index: int = 0,
    ) -> int:
        raise NotImplementedError

    def reset_task(self, task_id: str, task: Optional[Dict[str, Any]] = None) -> None:
        del task_id, task

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        del task_id
        return None

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
        del task_id, step_index, task, selected_action, tool_execution, step_result, state_before, state_after
