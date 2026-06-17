"""
Adapter for WM-Defense-Benchmark.
"""

from typing import Any, Dict, Optional

from .base import BenchmarkAdapter, register_adapter


class WMDefenseBenchAdapter(BenchmarkAdapter):
    """Adapter for benchmark names `wmdefensebench` and `mydefensebench`."""

    def state_to_safepred(
        self,
        raw_state: Any,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(raw_state, dict):
            raw_state = {}
        if metadata is None:
            metadata = {}

        goal = intent or metadata.get("intent", "") or raw_state.get("goal", "")
        return {
            "axtree_txt": raw_state.get("a11y_tree_txt", "") or raw_state.get("axtree_txt", ""),
            "url": raw_state.get("url", "unknown"),
            "goal": goal,
            "intent": goal,
            "page_type": raw_state.get("page_type", "unknown"),
            "action_history": metadata.get("action_history", []),
        }

    def action_to_safepred(self, action: Any) -> str:
        if isinstance(action, str):
            return action.strip()
        if isinstance(action, dict):
            action_type = action.get("type", "unknown")
            target = action.get("target", "")
            text = action.get("text", "")
            action_str = f"{action_type} [{target}] {text}".strip()
            return " ".join(action_str.split())
        return str(action)

    def action_from_safepred(self, action_str: str) -> Any:
        # Keep benchmark action as plain string for easier downstream evaluation.
        return action_str


# New canonical benchmark name.
register_adapter("wmdefensebench", WMDefenseBenchAdapter)
# Backward compatibility for earlier experiments.
register_adapter("mydefensebench", WMDefenseBenchAdapter)
