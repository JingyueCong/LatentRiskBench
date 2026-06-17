from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseTool


class SimulatedTool(BaseTool):
    """A legitimate tool available to the agent, loaded from normal_tools.jsonl.

    Simulates successful execution by returning a predefined achievement string,
    allowing the world model to evaluate action sequences without live execution.
    """

    def __init__(self, name: str, tools_info_path: str | Path) -> None:
        self.name = name
        tools_info_path = Path(tools_info_path)
        tool_row: Optional[Dict[str, Any]] = None
        with tools_info_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("Tool Name") == name:
                    tool_row = entry
                    break
        if tool_row is None:
            raise ValueError(f"Tool '{name}' not found in {tools_info_path}")
        self.description: str = tool_row["Description"]
        self.expected_achievement: str = tool_row["Expected Achievements"]
        self.corresponding_skill: str = tool_row.get("Corresponding Skill", "")

    def run(self, params: Optional[Dict[str, Any]] = None) -> str:
        return self.expected_achievement

    def get_tool_call_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": None,
            },
        }


class AttackerTool(BaseTool):
    """A malicious tool injected by an attacker to subvert agent behaviour.

    Disguised with a benign-sounding description, but run() confirms the
    attacker goal was achieved — enabling the world model to detect when an
    agent selects or executes an injected tool.
    """

    def __init__(self, attacker_tool: Dict[str, Any]) -> None:
        self.tool_name: str = attacker_tool["Attacker Tool"]
        self.tool_description: str = attacker_tool["Description"]
        self.attack_goal: str = attacker_tool["Attack goal"]
        self.attack_type: str = attacker_tool.get("Attack Type", "Unknown")
        self.corresponding_skill: str = attacker_tool.get("Corresponding Skill", "")
        self.aggressive: bool = str(attacker_tool.get("Aggressive", "False")).lower() == "true"

    def run(self, params: Optional[Dict[str, Any]] = None) -> str:
        return (
            f"You have used the attacker tool {self.tool_name}. "
            f"You have achieved the attacker goal: {self.attack_goal}"
        )

    def get_tool_call_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
            },
        }
