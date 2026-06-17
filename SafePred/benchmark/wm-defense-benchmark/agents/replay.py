from __future__ import annotations

from typing import Any, Dict, List

from .base import AgentHandler


class ReplayAgentHandler(AgentHandler):
    def __init__(self) -> None:
        super().__init__(agent_id="agent_replay")

    def select_proposed_index(
        self,
        task: Dict[str, Any],
        candidate_actions: List[Any],
        fallback_index: int = 0,
    ) -> int:
        proposed_index = int(task.get("proposed_action_index", fallback_index))
        if not candidate_actions:
            return 0
        return max(0, min(proposed_index, len(candidate_actions) - 1))
