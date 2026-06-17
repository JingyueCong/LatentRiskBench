from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    category: str
    action_prefixes: List[str]
    env_suites: List[str]
    risk_level: str
    description: str

    def metadata(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "category": self.category,
            "action_prefixes": list(self.action_prefixes),
            "env_suites": list(self.env_suites),
            "risk_level": self.risk_level,
            "description": self.description,
        }
