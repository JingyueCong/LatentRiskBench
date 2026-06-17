"""Task-loading helpers that enforce the dev/test label-sequestration discipline.

``tasks_test.jsonl`` is published without ``unsafe_action_indices`` or
``proposed_action_index``. Those labels live in ``tasks_test_labels.jsonl``
and must be merged back in by the scoring driver at evaluation time. The
discipline is enforced by convention: any code path that iterates raw
``tasks_test.jsonl`` rows will see label-free tasks and fail loudly when
it tries to score them, which surfaces accidental leakage.

Public API:
    load_tasks_jsonl(path)             - raw JSONL loader, no mutation.
    merge_test_labels(tasks, labels)   - reattach labels by task_id.
    load_tasks_with_labels(path, labels_path=None)
                                       - loader + optional merge, the
                                         one-call helper for runners.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_tasks_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file of task rows. Empty lines are skipped."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def merge_test_labels(
    tasks: List[Dict[str, Any]],
    labels_path: Optional[Path],
) -> List[Dict[str, Any]]:
    """Re-attach sequestered labels from ``labels_path`` onto ``tasks``.

    Returns a new list; the input ``tasks`` is not mutated. Rows without a
    matching label are returned unchanged — this keeps dev runs, where
    labels are inline, working without any changes.

    Raises ValueError if a label file exists but cannot be parsed, so
    malformed label files fail loudly rather than silently under-scoring
    test tasks.
    """
    if labels_path is None:
        return [dict(t) for t in tasks]
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return [dict(t) for t in tasks]
    label_map: Dict[str, Dict[str, Any]] = {}
    for row in load_tasks_jsonl(labels_path):
        tid = row.get("task_id")
        if not isinstance(tid, str) or not tid:
            raise ValueError(
                f"{labels_path}: label row missing string task_id: {row!r}"
            )
        label_map[tid] = row

    merged: List[Dict[str, Any]] = []
    for t in tasks:
        out = dict(t)
        tid = str(out.get("task_id", ""))
        labels = label_map.get(tid)
        if labels is not None:
            if "unsafe_action_indices" in labels:
                out["unsafe_action_indices"] = list(labels["unsafe_action_indices"])
            pi = labels.get("proposed_action_index")
            if isinstance(pi, int):
                out["proposed_action_index"] = pi
        merged.append(out)
    return merged


def load_tasks_with_labels(
    tasks_path: Path,
    *,
    test_labels_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load ``tasks_path`` and optionally merge in ``test_labels_path``.

    Use this in runners/validators whenever they need to evaluate the
    test split. If ``test_labels_path`` is None or missing, behaves
    identically to ``load_tasks_jsonl``.
    """
    rows = load_tasks_jsonl(tasks_path)
    if test_labels_path is not None:
        rows = merge_test_labels(rows, test_labels_path)
    return rows
