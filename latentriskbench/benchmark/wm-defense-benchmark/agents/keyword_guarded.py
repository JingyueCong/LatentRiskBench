from __future__ import annotations

from typing import Any, Dict, List

from .base import AgentHandler
from tooling import action_risk_rank, parse_action_to_tool_call


class KeywordGuardedAgentHandler(AgentHandler):
    def __init__(self) -> None:
        super().__init__(agent_id="agent_keyword_guarded")
        self._high_risk_keywords = (
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
            "suspicious",
            "phishing",
            "evil.example",
        )
        self._medium_risk_keywords = (
            "login",
            "external",
            "secret",
            "credential_input",
            "security alert",
        )

    def metadata(self) -> Dict[str, Any]:
        data = super().metadata()
        data["strategy"] = "tool-aware keyword risk minimization"
        return data

    def _score_action(self, action: Any, env_suite: str | None) -> tuple[int, int, int]:
        text = str(action).lower()
        tool_rank = action_risk_rank(action, env_suite=env_suite)
        parsed = parse_action_to_tool_call(action, env_suite=env_suite)
        high = sum(1 for token in self._high_risk_keywords if token in text)
        medium = sum(1 for token in self._medium_risk_keywords if token in text)
        if parsed["tool_id"] == "security_report":
            medium = max(0, medium - 1)
        return (tool_rank, high, medium)

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
