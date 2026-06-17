"""Fold a defender world model's predicted delta into a state dict.

The defender WM ``raw_delta`` (produced by ``LLMBasedWorldModel.simulate`` and
carried through ``DefenderVerdict.raw_delta``) describes the WM's prediction
of how the state changes in response to an action. Its structured fields are:

    {
        "semantic_delta": "<free-form text>",
        "element_changes": {
            "new_elements": ["<label>", ...],
            "removed_elements": ["<label>", ...],
        },
        "risk_score": 0.8,
        "violated_policy_ids": [...],
        "risk_explanation": "...",
        "optimization_guidance": "...",
        ...  # other scoring fields not relevant to dynamics
    }

``fold_delta_into_state`` consumes the dynamics-relevant subset
(``semantic_delta`` + ``element_changes``) and returns a new state dict whose
``a11y_tree_txt`` reflects the predicted UI state. This is the single point
of coupling between "WM-as-classifier" (today) and "WM-as-forward-simulator"
(the target use case for multi-step attack search); both generators consume
this helper rather than reimplementing state synthesis.

The implementation is intentionally conservative:

- If ``raw_delta`` is empty, missing, or lacks dynamics fields, the input
  state is returned unchanged (as a copy). This keeps call sites safe when
  the defender is a stub or a coarsened scorer that never populated the
  dynamics fields.
- Only ``a11y_tree_txt`` is mutated. Other state keys (``url``,
  ``page_type``, etc.) are preserved verbatim because the WM does not emit
  structured updates for them today; folding free-form URL edits out of
  ``semantic_delta`` is unreliable and would silently diverge from ground
  truth during evaluation.
- String-level element removal is best-effort: a listed removed element is
  stripped from ``a11y_tree_txt`` if it appears verbatim; otherwise the
  removal is recorded in the ``[predicted] ...`` trailer so downstream
  consumers can still audit the WM's intent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _as_str_list(value: Any) -> List[str]:
    """Coerce ``value`` to a list of strings. Non-list/None inputs yield []."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
    return out


def _get_element_changes(raw_delta: Dict[str, Any]) -> tuple[List[str], List[str]]:
    changes = raw_delta.get("element_changes")
    if not isinstance(changes, dict):
        return [], []
    return (
        _as_str_list(changes.get("new_elements")),
        _as_str_list(changes.get("removed_elements")),
    )


def _get_semantic_delta(raw_delta: Dict[str, Any]) -> str:
    val = raw_delta.get("semantic_delta")
    return val.strip() if isinstance(val, str) else ""


def fold_delta_into_state(
    current_state: Optional[Dict[str, Any]],
    raw_delta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a new state dict with the WM's predicted delta applied.

    Args:
        current_state: State dict as seen by the generator (keys like
            ``a11y_tree_txt``, ``url``, ``page_type``). May be ``None`` or
            empty, in which case an empty dict is the starting point.
        raw_delta: ``DefenderVerdict.raw_delta``. May be ``None`` or empty,
            in which case ``current_state`` is returned unchanged (as a
            shallow copy).

    Returns:
        A shallow copy of ``current_state`` with ``a11y_tree_txt`` updated
        to reflect predicted element additions/removals and a semantic
        trailer. All other keys are preserved verbatim.
    """
    state: Dict[str, Any] = dict(current_state or {})
    if not isinstance(raw_delta, dict) or not raw_delta:
        return state

    new_elements, removed_elements = _get_element_changes(raw_delta)
    semantic_delta = _get_semantic_delta(raw_delta)

    if not new_elements and not removed_elements and not semantic_delta:
        return state

    axtree = state.get("a11y_tree_txt")
    axtree = axtree if isinstance(axtree, str) else ""

    # Best-effort removal: strip verbatim occurrences; unrecognised removals
    # fall through to the predicted trailer so nothing is silently lost.
    unrecognised_removals: List[str] = []
    for elt in removed_elements:
        if elt and elt in axtree:
            axtree = axtree.replace(elt, "")
        elif elt:
            unrecognised_removals.append(elt)

    trailer_parts: List[str] = []
    if new_elements:
        trailer_parts.append("new: " + "; ".join(new_elements))
    if unrecognised_removals:
        trailer_parts.append("removed: " + "; ".join(unrecognised_removals))
    if semantic_delta:
        trailer_parts.append(semantic_delta)

    if trailer_parts:
        trailer = "[predicted] " + " | ".join(trailer_parts)
        axtree = (axtree + ("\n" if axtree else "") + trailer).strip()
    else:
        axtree = axtree.strip()

    state["a11y_tree_txt"] = axtree
    return state
