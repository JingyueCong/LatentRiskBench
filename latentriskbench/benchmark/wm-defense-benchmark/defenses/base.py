from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DefenseHandler:
    defense_id: str
    uses_world_model: bool

    def metadata(self) -> Dict[str, Any]:
        return {
            "defense_id": self.defense_id,
            "uses_world_model": self.uses_world_model,
        }

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
        raise NotImplementedError
