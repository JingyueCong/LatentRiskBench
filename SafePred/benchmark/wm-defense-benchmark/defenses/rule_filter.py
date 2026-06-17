"""Rule-filter defense (Phase 1 baseline).

Uses the ``RulePolicyOracle`` at decision time. This is the
"policy-as-defense" baseline: everything the independent oracle can
catch is already filtered, without any LLM or world-model simulation.

Why it matters as a baseline:
- By construction the oracle has 100% coverage on hand-labelled Phase 0
  tasks (see ``validate_oracle_coverage.py``), so on those tasks this
  defense is perfect.
- On WM-adaptive attacks (Phase 1 & 3), the oracle's coverage degrades
  because adaptive attackers find formulations outside the rule set
  (e.g., the `--value/--key` escape DeepSeek discovered). This defense
  is therefore the "strong rule baseline" that WM gating must beat on
  novel attack families to justify its added cost.
- The oracle is configurable (step-level vs trajectory) so two variants
  of this defense live side by side:
    defense_rule_filter_step        — single-step RulePolicyOracle only
    defense_rule_filter_trajectory  — step + TrajectoryPolicyOracle

The trajectory variant consults the agent's action_history passed to
``evaluate_step``; it's the first defense in the repo that reasons about
prior actions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import DefenseHandler
from .filter_base import wrap_filter_result


class _LazyOracle:
    """Lazy oracle holder so the handler can be instantiated before the
    ``policy_oracle`` package is importable (the handler registry builds
    at import time)."""

    def __init__(self, use_trajectory: bool) -> None:
        self.use_trajectory = use_trajectory
        self._oracle: Any = None
        self._policies: Optional[List[Dict[str, Any]]] = None

    def ensure(self) -> None:
        if self._oracle is not None and self._policies is not None:
            return
        from policy_oracle import (  # type: ignore
            RulePolicyOracle,
            TrajectoryPolicyOracle,
            load_policies,
        )
        if self.use_trajectory:
            self._oracle = TrajectoryPolicyOracle(rule_oracle=RulePolicyOracle())
        else:
            self._oracle = RulePolicyOracle()
        benchmark_root = Path(__file__).resolve().parents[1]
        self._policies = load_policies(benchmark_root / "data" / "policies.json")


class RuleFilterDefenseHandler(DefenseHandler):
    """Step-level rule-filter defense (no trajectory awareness)."""

    def __init__(self) -> None:
        super().__init__(defense_id="defense_rule_filter", uses_world_model=False)
        self._cache = _LazyOracle(use_trajectory=False)

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
        self._cache.ensure()
        assert self._cache._oracle is not None and self._cache._policies is not None

        def _predicate(_idx: int, action: Any) -> tuple[bool, List[str], str, float]:
            verdict = self._cache._oracle.evaluate(
                intent, state, str(action), self._cache._policies
            )
            risk = 0.8 if verdict.unsafe else 0.0
            return (
                bool(verdict.unsafe),
                list(verdict.violated_policy_ids),
                verdict.rationale,
                risk,
            )

        return wrap_filter_result(
            candidate_actions, unsafe_action_indices, proposed_index, _predicate
        )


class TrajectoryRuleFilterDefenseHandler(DefenseHandler):
    """Rule-filter defense that ALSO evaluates the accumulated trajectory.

    Candidate action ``a`` is flagged if either:
    - step-level oracle flags ``a`` alone, OR
    - ``action_history + [a]`` triggers any trajectory rule.

    This matters against Phase 3 beam attacks where each step is
    individually safe but the chain violates a cross-step pattern.
    """

    def __init__(self) -> None:
        super().__init__(
            defense_id="defense_rule_filter_trajectory", uses_world_model=False
        )
        self._cache = _LazyOracle(use_trajectory=True)

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
        self._cache.ensure()
        oracle = self._cache._oracle
        policies = self._cache._policies
        assert oracle is not None and policies is not None
        history = [str(a) for a in (action_history or [])]

        def _predicate(_idx: int, action: Any) -> tuple[bool, List[str], str, float]:
            full_history = history + [str(action)]
            verdict = oracle.evaluate_trajectory(intent, state, full_history, policies)
            risk = 0.8 if verdict.unsafe else 0.0
            return (
                bool(verdict.unsafe),
                list(verdict.violated_policy_ids),
                verdict.rationale,
                risk,
            )

        return wrap_filter_result(
            candidate_actions, unsafe_action_indices, proposed_index, _predicate
        )
