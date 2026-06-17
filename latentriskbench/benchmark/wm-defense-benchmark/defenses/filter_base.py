"""Shared plumbing for filter-style defenses.

A filter-style defense walks the candidate_actions list in order,
consulting some ``is_unsafe(action)`` predicate, and:
- picks the proposed action if the predicate says safe;
- otherwise picks the first other safe candidate;
- otherwise requests regeneration.

All Phase 1 non-WM defenses (keyword list, rule filter, LLM judge) share
this selection logic. This helper keeps the logic in one place so that
later behavioural changes (e.g. adding a confidence threshold) can be
made once.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .none import build_step_result


def filter_then_select(
    candidate_actions: List[Any],
    unsafe_action_indices: List[int],
    proposed_index: int,
    is_unsafe: Callable[[int, Any], tuple[bool, List[str], str, float]],
) -> Dict[str, Any]:
    """Select a safe action via a predicate; fall back to regeneration.

    ``is_unsafe`` returns (is_flagged, violated_policy_ids, explanation,
    risk_score) for a single (index, action) pair.

    Selection order:
    1. the proposed action, if the predicate says safe;
    2. the lowest-index candidate for which the predicate says safe;
    3. no candidate -> ``requires_regeneration=True``, ``is_safe=False``.

    The returned dict is shaped for ``build_step_result`` so the caller
    only has to wrap this output.
    """
    proposed_index = max(0, min(proposed_index, len(candidate_actions) - 1))
    verdicts: List[tuple[bool, List[str], str, float]] = [
        is_unsafe(idx, action) for idx, action in enumerate(candidate_actions)
    ]

    # Per-candidate risk scores aligned to candidate_actions, so build_step_result
    # can expose candidate_risk_scores for a per-candidate violation-prediction
    # AUROC. Even the binary-scoring baselines (keyword/rule) yield a meaningful
    # short-horizon AUROC this way, matching the world-model defense's column.
    root_action_risks = [
        {"action": str(action), "risk_score": float(risk), "violated_policy_ids": list(vids)}
        for action, (flagged, vids, exp, risk) in zip(candidate_actions, verdicts)
    ]

    # 1. Proposed action first.
    proposed_flagged, proposed_vids, proposed_exp, proposed_risk = verdicts[proposed_index]
    if not proposed_flagged:
        return {
            "selected_action": candidate_actions[proposed_index],
            "risk_score": proposed_risk,
            "risk_explanation": proposed_exp or "proposed action passed filter",
            "requires_regeneration": False,
            "violated_policy_ids": list(proposed_vids),
            "is_safe": True,
            "root_action_risks": root_action_risks,
        }

    # 2. First other safe candidate.
    for idx, (flagged, vids, exp, risk) in enumerate(verdicts):
        if idx == proposed_index:
            continue
        if not flagged:
            return {
                "selected_action": candidate_actions[idx],
                "risk_score": risk,
                "risk_explanation": exp or f"fell back from flagged proposed to idx {idx}",
                "requires_regeneration": False,
                "violated_policy_ids": list(vids),
                "is_safe": True,
                "root_action_risks": root_action_risks,
            }

    # 3. Everything flagged -> regenerate.
    combined_vids: List[str] = []
    seen = set()
    for _, vids, _, _ in verdicts:
        for pid in vids:
            if pid not in seen:
                seen.add(pid)
                combined_vids.append(pid)
    max_risk = max((v[3] for v in verdicts), default=1.0)
    reason = verdicts[proposed_index][2] or "all candidates flagged by filter"
    return {
        "selected_action": None,
        "risk_score": max_risk,
        "risk_explanation": reason,
        "requires_regeneration": True,
        "violated_policy_ids": combined_vids,
        "is_safe": False,
        "root_action_risks": root_action_risks,
    }


def wrap_filter_result(
    candidate_actions: List[Any],
    unsafe_action_indices: List[int],
    proposed_index: int,
    is_unsafe: Callable[[int, Any], tuple[bool, List[str], str, float]],
) -> Dict[str, Any]:
    """Convenience: run filter_then_select and wrap in build_step_result."""
    result = filter_then_select(
        candidate_actions, unsafe_action_indices, proposed_index, is_unsafe
    )
    return build_step_result(candidate_actions, unsafe_action_indices, proposed_index, result)
