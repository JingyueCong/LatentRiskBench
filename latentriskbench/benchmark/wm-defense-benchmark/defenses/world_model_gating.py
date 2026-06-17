from __future__ import annotations

from typing import Any, Dict, List

from .base import DefenseHandler
from .none import build_step_result


class WorldModelGatingDefenseHandler(DefenseHandler):
    def __init__(self) -> None:
        super().__init__(defense_id="defense_world_model_gating", uses_world_model=True)

    def evaluate_step(
        self,
        wrapper: Any,
        task_id: str,
        intent: str,
        state: Dict[str, Any],
        candidate_actions: List[Any],
        unsafe_action_indices: List[int],
        proposed_index: int,
        action_history: List[str],
        current_response: str,
    ) -> Dict[str, Any]:
        if wrapper is None:
            raise RuntimeError("WorldModelGatingDefenseHandler requires an initialized SafePredWrapper.")

        proposed_index = max(0, min(proposed_index, len(candidate_actions) - 1))
        proposed_action = candidate_actions[proposed_index]
        wrapper.config.tree_search_n_root = len(candidate_actions)
        wrapper.safe_agent.config.tree_search_n_root = len(candidate_actions)
        result = wrapper.evaluate_action_risk(
            state=state,
            action=proposed_action,
            candidate_actions=candidate_actions,
            intent=intent,
            metadata={
                "task_id": task_id,
                "intent": intent,
                "action_history": action_history,
                "current_response": current_response,
            },
        )
        return build_step_result(candidate_actions, unsafe_action_indices, proposed_index, result)
