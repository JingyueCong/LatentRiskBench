from __future__ import annotations

from typing import Any, Dict, List

from .base import AgentHandler
from tooling import action_risk_rank, parse_action_to_tool_call


class HeuristicRankerAgentHandler(AgentHandler):
    def __init__(self) -> None:
        super().__init__(agent_id="agent_heuristic_ranker")
        self._unsafe_keywords = (
            "api_key",
            "token",
            "password",
            "credential",
            "ssh",
            "id_rsa",
            "curl",
            "scp",
            "upload",
            "exfil",
            "phishing",
            "suspicious",
            "evil.example",
        )
        self._safe_keywords = (
            "report",
            "refuse",
            "finish",
            "done",
            "save",
            "help",
            "trusted",
        )

    def metadata(self) -> Dict[str, Any]:
        data = super().metadata()
        data["strategy"] = "tool-aware heuristic action ranking"
        return data

    def _score_action(self, action: Any, env_suite: str | None) -> tuple[int, int, int, int]:
        text = str(action).lower()
        tool_rank = action_risk_rank(action, env_suite=env_suite)
        parsed = parse_action_to_tool_call(action, env_suite=env_suite)
        unsafe_hits = sum(1 for token in self._unsafe_keywords if token in text)
        safe_hits = sum(1 for token in self._safe_keywords if token in text)
        if parsed["tool_id"] == "security_report":
            safe_hits += 1
        # Lower is better. Prefer lower tool risk, then fewer unsafe hits,
        # then more safe hits, then shorter actions.
        return (tool_rank, unsafe_hits, -safe_hits, len(text))

    def select_proposed_index(
        self,
        task: Dict[str, Any],
        candidate_actions: List[Any],
        fallback_index: int = 0,
    ) -> int:
        if not candidate_actions:
            return 0

        env_suite = str(task.get("env_suite", ""))
        best_idx = 0
        best_score = self._score_action(candidate_actions[0], env_suite)
        for idx, action in enumerate(candidate_actions[1:], start=1):
            score = self._score_action(action, env_suite)
            if score < best_score:
                best_idx = idx
                best_score = score
        return best_idx
