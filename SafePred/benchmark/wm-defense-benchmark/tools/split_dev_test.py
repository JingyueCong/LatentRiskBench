#!/usr/bin/env python3
"""Split ``data/tasks.jsonl`` into deterministic dev / test sets.

Rationale (from docs/wm_0417.md §B.1 "Label leakage"):
    All 36 offline tasks currently publish ``unsafe_action_indices``
    inline. A defense can trivially be tuned against those labels during
    development, producing optimistic evaluation numbers. To fix this we:

    1. split the task set into ``tasks_dev.jsonl`` (labels kept inline,
       used for defense iteration) and ``tasks_test.jsonl`` (labels
       stripped, used only at final evaluation),
    2. write the test labels to ``tasks_test_labels.jsonl`` so a scoring
       driver can re-attach them. In a real benchmark deployment this
       file is server-side-only; here it lives in the repo but is
       loaded lazily by the runner so development iteration cannot
       accidentally look at it.

The split is:
- seeded (default seed=0) so the same split is reproducible forever;
- stratified by ``attack_family`` so every family appears in both
  halves in rough proportion;
- 70/30 by default (25 dev / 11 test for the current 36-task corpus).

Idempotent: running twice with the same seed produces bit-identical
output files.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BENCHMARK_ROOT / "data"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(
    tasks: List[Dict[str, Any]],
    *,
    test_fraction: float = 0.30,
    seed: int = 0,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (dev, test) split, stratified by attack_family.

    Determinism:
    - sort tasks by task_id inside each family before sampling so the
      order is independent of input iteration order,
    - use a dedicated ``random.Random(seed)`` instance (never the global
      RNG) so tests remain reproducible.
    """
    rng = random.Random(seed)
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        by_family[str(t.get("attack_family", "unknown"))].append(t)

    dev: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for family, items in sorted(by_family.items()):
        items_sorted = sorted(items, key=lambda t: str(t.get("task_id", "")))
        rng.shuffle(items_sorted)
        n_test = max(1, round(len(items_sorted) * test_fraction))
        # Ensure dev is non-empty too when family has >= 2 items.
        if n_test >= len(items_sorted) and len(items_sorted) > 1:
            n_test = len(items_sorted) - 1
        test.extend(items_sorted[:n_test])
        dev.extend(items_sorted[n_test:])

    # Sort outputs by task_id for stable diffs.
    dev.sort(key=lambda t: str(t.get("task_id", "")))
    test.sort(key=lambda t: str(t.get("task_id", "")))
    return dev, test


def strip_labels(task: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a label-free copy of ``task`` for the public test file.

    Removes:
    - ``unsafe_action_indices`` (the primary label)
    - ``proposed_action_index`` (meta-label that leaks the expected agent
      behaviour; a scoring-time convenience)
    """
    out = dict(task)
    out.pop("unsafe_action_indices", None)
    out.pop("proposed_action_index", None)
    return out


def label_record(task: Dict[str, Any]) -> Dict[str, Any]:
    """One row of ``tasks_test_labels.jsonl``: task_id + sequestered labels."""
    return {
        "task_id": str(task.get("task_id", "")),
        "unsafe_action_indices": list(task.get("unsafe_action_indices", []) or []),
        "proposed_action_index": task.get("proposed_action_index"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split tasks.jsonl into dev/test.")
    parser.add_argument("--tasks", type=Path, default=DATA_ROOT / "tasks.jsonl")
    parser.add_argument("--dev-out", type=Path, default=DATA_ROOT / "tasks_dev.jsonl")
    parser.add_argument("--test-out", type=Path, default=DATA_ROOT / "tasks_test.jsonl")
    parser.add_argument(
        "--test-labels-out",
        type=Path,
        default=DATA_ROOT / "tasks_test_labels.jsonl",
    )
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split summary but do not write files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = _read_jsonl(args.tasks)
    dev, test = stratified_split(tasks, test_fraction=args.test_fraction, seed=args.seed)

    stripped_test = [strip_labels(t) for t in test]
    labels = [label_record(t) for t in test]

    # Summary.
    summary = {
        "source": str(args.tasks),
        "total": len(tasks),
        "dev": len(dev),
        "test": len(test),
        "test_fraction_actual": round(len(test) / max(1, len(tasks)), 3),
        "seed": args.seed,
        "by_family": {},
    }
    # Per-family breakdown.
    by_family: Dict[str, Dict[str, int]] = defaultdict(lambda: {"dev": 0, "test": 0})
    for t in dev:
        by_family[str(t.get("attack_family", "unknown"))]["dev"] += 1
    for t in test:
        by_family[str(t.get("attack_family", "unknown"))]["test"] += 1
    summary["by_family"] = dict(by_family)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    _write_jsonl(args.dev_out, dev)
    _write_jsonl(args.test_out, stripped_test)
    _write_jsonl(args.test_labels_out, labels)


if __name__ == "__main__":
    main()
