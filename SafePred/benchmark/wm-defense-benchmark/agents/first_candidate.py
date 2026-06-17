from __future__ import annotations

from typing import Any, Dict, List

from .base import AgentHandler


class FirstCandidateAgentHandler(AgentHandler):
    def __init__(self) -> None:
        super().__init__(agent_id="agent_first_candidate")

    def select_proposed_index(
        self,
        task: Dict[str, Any],
        candidate_actions: List[Any],
        fallback_index: int = 0,
    ) -> int:
        if not candidate_actions:
            return 0
        return 0
