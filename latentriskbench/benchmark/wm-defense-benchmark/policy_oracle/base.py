from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class OracleVerdict:
    """Verdict produced by a PolicyOracle for a single (state, action) pair.

    Attributes:
        unsafe: True if the oracle judges the action violates at least one policy.
        violated_policy_ids: Policies the oracle believes this action violates.
        confidence: Oracle's self-reported confidence in [0, 1].
        rationale: Short human-readable explanation.
        oracle_id: Unique identifier of the oracle (e.g. "rule_v1", "llm_claude_sonnet_46").
        matched_rules: Optional list of rule identifiers that fired (rule oracle).
        raw_response: Optional raw model output (LLM oracle).
    """

    unsafe: bool
    violated_policy_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    oracle_id: str = ""
    matched_rules: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PolicyOracle(Protocol):
    """Oracle interface: (intent, state, action, policies) -> OracleVerdict."""

    oracle_id: str

    def evaluate(
        self,
        intent: str,
        state: Dict[str, Any],
        action: str,
        policies: List[Dict[str, Any]],
    ) -> OracleVerdict:
        ...


def load_policies(policies_path: Path) -> List[Dict[str, Any]]:
    """Load policy definitions from the benchmark's policies.json.

    Oracle implementations read policy metadata (ids, descriptions) for reporting,
    but the matching logic is hard-coded in each oracle. Changing policy prose
    does not silently change oracle behavior.
    """
    path = Path(policies_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"policies.json must be a list, got {type(data)}")
    return data


def extract_state_text(state: Dict[str, Any]) -> str:
    """Concatenate state fields the oracle cares about into one searchable blob."""
    if not isinstance(state, dict):
        return str(state or "")
    parts = []
    for key in ("a11y_tree_txt", "url", "page_type"):
        value = state.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)
