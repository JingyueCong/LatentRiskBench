#!/usr/bin/env python3
"""Validate that the policy oracle flags every labelled unsafe action.

Reads a task JSONL file, runs the rule oracle over every action listed in
each task's ``unsafe_action_indices``, and reports coverage.

This doubles as the Phase 0 smoke test: any hand-authored attack where the
oracle fails to flag the labelled unsafe action is a false-negative and
must be fixed (either by strengthening the rule set or by re-examining the
label itself).

Exit code is 0 iff every labelled unsafe action is flagged as unsafe by the
rule oracle AND no labelled safe action is flagged as unsafe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from policy_oracle import RulePolicyOracle, load_policies  # noqa: E402
from runners.task_loader import merge_test_labels  # noqa: E402


def _iter_tasks(path: Path, test_labels: Path = None):
    """Stream tasks from ``path``; if ``test_labels`` is given, merge them in.

    The merge runs in memory once; streaming is preserved afterward. This
    keeps the validator usable against both the full ``tasks.jsonl`` (no
    merge) and ``tasks_test.jsonl`` (labels must be merged or every
    action will look "unlabelled" and the coverage check becomes trivial).
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
    if test_labels is not None:
        tasks_only = [t for _, t in rows]
        merged = merge_test_labels(tasks_only, test_labels)
        rows = list(zip((ln for ln, _ in rows), merged))
    for line_no, task in rows:
        yield line_no, task


def _eval_offline(task: Dict[str, Any], oracle: RulePolicyOracle, policies: List[Dict[str, Any]]):
    intent = str(task.get("intent", ""))
    state = task.get("state", {}) or {}
    candidates = task.get("candidate_actions", []) or []
    unsafe_idx = set(int(i) for i in task.get("unsafe_action_indices", []) or [])
    results = []
    for idx, action in enumerate(candidates):
        verdict = oracle.evaluate(intent, state, str(action), policies)
        results.append(
            {
                "index": idx,
                "action": str(action),
                "labelled_unsafe": idx in unsafe_idx,
                "oracle_unsafe": verdict.unsafe,
                "violated_policy_ids": verdict.violated_policy_ids,
                "matched_rules": verdict.matched_rules,
                "rationale": verdict.rationale,
            }
        )
    return results


def _eval_online(task: Dict[str, Any], oracle: RulePolicyOracle, policies: List[Dict[str, Any]]):
    intent = str(task.get("intent", ""))
    results = []
    for step_idx, step in enumerate(task.get("steps", []) or []):
        state = step.get("state") or task.get("initial_state", {}) or {}
        candidates = step.get("candidate_actions", []) or []
        unsafe_idx = set(int(i) for i in step.get("unsafe_action_indices", []) or [])
        for idx, action in enumerate(candidates):
            verdict = oracle.evaluate(intent, state, str(action), policies)
            results.append(
                {
                    "step": step_idx,
                    "index": idx,
                    "action": str(action),
                    "labelled_unsafe": idx in unsafe_idx,
                    "oracle_unsafe": verdict.unsafe,
                    "violated_policy_ids": verdict.violated_policy_ids,
                    "matched_rules": verdict.matched_rules,
                    "rationale": verdict.rationale,
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    data_root = BENCHMARK_ROOT / "data"
    parser = argparse.ArgumentParser(
        description="Validate rule-oracle coverage of labelled unsafe actions."
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=data_root / "tasks.jsonl",
        help="Path to a task JSONL file (offline or online).",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=data_root / "policies.json",
        help="Path to policies.json.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "online", "auto"),
        default="auto",
        help="Task format. 'auto' detects by file extension and first record.",
    )
    parser.add_argument(
        "--strict-over-flag",
        action="store_true",
        help="Also fail if a labelled-safe action is flagged unsafe.",
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        default=None,
        help="Optional path to tasks_test_labels.jsonl. When provided, "
        "labels are merged into the loaded tasks before coverage is "
        "computed. Use this to validate coverage on the sequestered test "
        "split without editing the test file itself.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a full JSON coverage report.",
    )
    return parser.parse_args()


def _detect_mode(path: Path, hint: str) -> str:
    if hint in {"offline", "online"}:
        return hint
    name = path.name.lower()
    if "online" in name:
        return "online"
    # Peek first non-empty record.
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            return "online" if "steps" in record else "offline"
    return "offline"


def main() -> None:
    args = parse_args()
    mode = _detect_mode(args.tasks, args.mode)
    oracle = RulePolicyOracle()
    policies = load_policies(args.policies)

    per_task_reports: List[Dict[str, Any]] = []
    false_negatives: List[Dict[str, Any]] = []
    false_positives: List[Dict[str, Any]] = []
    total_unsafe = 0
    total_safe = 0

    for line_no, task in _iter_tasks(args.tasks, test_labels=args.test_labels):
        task_id = str(task.get("task_id", f"line{line_no}"))
        evals = _eval_online(task, oracle, policies) if mode == "online" else _eval_offline(task, oracle, policies)
        per_task_reports.append({"task_id": task_id, "line": line_no, "results": evals})
        for row in evals:
            if row["labelled_unsafe"]:
                total_unsafe += 1
                if not row["oracle_unsafe"]:
                    false_negatives.append({"task_id": task_id, **row})
            else:
                total_safe += 1
                if row["oracle_unsafe"]:
                    false_positives.append({"task_id": task_id, **row})

    coverage = 1.0 - (len(false_negatives) / max(1, total_unsafe))
    summary = {
        "tasks_file": str(args.tasks),
        "mode": mode,
        "policies_file": str(args.policies),
        "total_unsafe_labelled": total_unsafe,
        "total_safe_labelled": total_safe,
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "coverage": round(coverage, 4),
        "first_false_negatives": false_negatives[:10],
        "first_false_positives": false_positives[:10],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.report:
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "per_task": per_task_reports,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    failed = bool(false_negatives) or (args.strict_over_flag and bool(false_positives))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
