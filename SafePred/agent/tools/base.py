from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """Base class for all agent tools."""

    @abstractmethod
    def run(self, params: Optional[Dict[str, Any]] = None) -> str:
        """Execute the tool and return a result string."""

    @abstractmethod
    def get_tool_call_format(self) -> Dict[str, Any]:
        """Return OpenAI function-calling schema for this tool."""
