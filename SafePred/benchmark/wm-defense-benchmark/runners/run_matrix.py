#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from attacks import get_attack_handler, list_attack_metadata
from agents import list_agent_metadata
from schemas.base_task_schema import validate_base_task_file
from defenses import get_defense_handler, list_defense_metadata
from schemas.explicit_task_schema import validate_offline_task_file, validate_online_task_file
from metrics import aggregate_reports
from schemas.payload_schema import validate_payload_file
from skills import list_skill_metadata
from schemas.tool_registry_schema import validate_tool_registry_file
from tooling import list_tool_executor_ids, list_tool_metadata


def parse_args() -> argparse.Namespace:
    root = BENCHMARK_ROOT
    data_root = root / "data"
    parser = argparse.ArgumentParser(description="Run attack x defense x seed matrix and aggregate report.")
    parser.add_argument("--protocol", type=Path, default=root / "eval_protocol.yaml")
    parser.add_argument("--tasks", type=Path, default=data_root / "tasks.jsonl")
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    parser.add_argument("--policy", type=Path, default=data_root / "policies.json")
    parser.add_argument("--mode", choices=["offline", "online"], default=None)
    parser.add_argument(
        "--agent-mode",
        choices=["from_task", "replay", "first_candidate", "keyword_guarded", "heuristic_ranker", "planner", "llm_planner"],
        default="from_task",
        help="Agent proposal mode passed through to the single-run runner.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=root / "matrix_results")
    parser.add_argument(
        "--list-registry",
        action="store_true",
        help="Print registered attacks and defenses, then exit.",
    )
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Validate benchmark input files before running the matrix.",
    )
    return parser.parse_args()


def load_protocol(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Protocol must be a mapping.")
    return data


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_payload_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flatten_tool_metrics_for_csv(cell_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell in cell_summary.values():
        tool_metrics = cell.get("tool_metrics_by_tool", {})
        if not isinstance(tool_metrics, dict):
            continue
        for tool_id, metrics in tool_metrics.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "attack_id": cell.get("attack_id"),
                    "defense_id": cell.get("defense_id"),
                    "tool_id": tool_id,
                    "category": metrics.get("category"),
                    "risk_level": metrics.get("risk_level"),
                    "selected_count": metrics.get("selected_count", 0),
                    "execution_count": metrics.get("execution_count", 0),
                    "success_count": metrics.get("success_count", 0),
                    "failure_count": metrics.get("failure_count", 0),
                    "violation_count": metrics.get("violation_count", 0),
                    "success_rate": metrics.get("success_rate", 0.0),
                    "failure_rate": metrics.get("failure_rate", 0.0),
                    "violation_rate": metrics.get("violation_rate", 0.0),
                }
            )
    return rows


def _flatten_skill_metrics_for_csv(cell_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell in cell_summary.values():
        skill_metrics = cell.get("skill_metrics_by_skill", {})
        if not isinstance(skill_metrics, dict):
            continue
        for skill_id, metrics in skill_metrics.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "attack_id": cell.get("attack_id"),
                    "defense_id": cell.get("defense_id"),
                    "skill_id": skill_id,
                    "task_count": metrics.get("task_count", 0),
                    "step_count": metrics.get("step_count", 0),
                    "attacked_task_count": metrics.get("attacked_task_count", 0),
                    "task_success_rate": metrics.get("task_success_rate", 0.0),
                    "original_goal_success_rate": metrics.get("original_goal_success_rate", 0.0),
                    "attack_success_rate": metrics.get("attack_success_rate", 0.0),
                    "violation_rate": metrics.get("violation_rate", 0.0),
                    "over_refusal_rate": metrics.get("over_refusal_rate", 0.0),
                    "necessary_refusal_rate": metrics.get("necessary_refusal_rate", 0.0),
                    "avg_risk_score": metrics.get("avg_risk_score", 0.0),
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_inputs_for_matrix(args: argparse.Namespace, mode: str) -> None:
    errors: List[str] = []
    if mode == "online":
        errors.extend(validate_online_task_file(args.tasks))
    else:
        # Matrix can consume either explicit tasks or base tasks. Try explicit validation first,
        # then fall back to base-task validation if explicit validation fails.
        explicit_errors = validate_offline_task_file(args.tasks)
        if explicit_errors:
            base_errors = validate_base_task_file(args.tasks)
            if base_errors:
                errors.extend(explicit_errors)
                errors.extend(base_errors)

    benchmark_root = BENCHMARK_ROOT
    data_root = benchmark_root / "data"
    base_tasks_path = data_root / "base_tasks.jsonl"
    if base_tasks_path.exists():
        errors.extend(validate_base_task_file(base_tasks_path))

    payload_root = data_root / "attack_payloads"
    for payload_file in iter_payload_files(payload_root):
        errors.extend(validate_payload_file(payload_file))
    tool_registry_path = data_root / "tool_registry.json"
    if tool_registry_path.exists():
        errors.extend(validate_tool_registry_file(tool_registry_path))

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))


def canonical_defense_mode(defense_id: str) -> str:
    if not get_defense_handler(defense_id).uses_world_model:
        return "none"
    return "world_model"


def main() -> None:
    args = parse_args()
    if args.list_registry:
        print(
            json.dumps(
                {
                    "attacks": list_attack_metadata(),
                    "agents": list_agent_metadata(),
                    "defenses": list_defense_metadata(),
                    "skills": list_skill_metadata(),
                    "tools": list_tool_metadata(),
                    "tool_executors": list_tool_executor_ids(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    protocol = load_protocol(args.protocol)
    tasks = load_jsonl(args.tasks)
    attacks = protocol.get("matrix", {}).get("attacks", ["attack_none"])
    defenses = protocol.get("matrix", {}).get("defenses", ["defense_world_model_gating"])
    runs_per_cell = int(protocol.get("execution", {}).get("runs_per_cell", 1))
    mode = args.mode or protocol.get("execution", {}).get("mode", "offline")
    max_steps = args.max_steps if args.max_steps is not None else int(protocol.get("execution", {}).get("max_steps", 20))
    if args.validate_inputs:
        validate_inputs_for_matrix(args, mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_reports: List[Dict[str, Any]] = []

    for attack_id in attacks:
        attack_handler = get_attack_handler({"attack_id": attack_id})
        for defense_id in defenses:
            defense_handler = get_defense_handler(defense_id)
            defense_mode = canonical_defense_mode(defense_id)
            for run_idx in range(runs_per_cell):
                seed = run_idx
                cell_tasks = []
                for t in tasks:
                    transformed = attack_handler.apply_to_task(dict(t))
                    if transformed is None:
                        continue
                    row = dict(transformed)
                    row["attack_id"] = attack_handler.attack_id
                    row["attack_family"] = attack_handler.attack_family
                    row["defense_id"] = defense_handler.defense_id
                    row["seed"] = seed
                    cell_tasks.append(row)
                if args.max_tasks > 0:
                    cell_tasks = cell_tasks[: args.max_tasks]
                if not cell_tasks:
                    continue

                task_file = args.output_dir / f"tasks_{attack_id}_{defense_id}_seed{seed}.jsonl"
                out_file = args.output_dir / f"result_{attack_id}_{defense_id}_seed{seed}.json"
                save_jsonl(task_file, cell_tasks)

                cmd = [
                    sys.executable,
                    str(BENCHMARK_ROOT / "runners" / "run.py"),
                    "--mode",
                    mode,
                    "--agent-mode",
                    args.agent_mode,
                    "--tasks",
                    str(task_file),
                    "--config",
                    str(args.config),
                    "--policy",
                    str(args.policy),
                    "--output",
                    str(out_file),
                    "--defense-mode",
                    defense_mode,
                ]
                if mode == "online":
                    cmd.extend(["--max-steps", str(max_steps)])
                subprocess.run(cmd, check=True)

                with out_file.open("r", encoding="utf-8") as f:
                    rep = json.load(f)
                cell_reports.append(
                    {
                        "attack_id": attack_id,
                        "defense_id": defense_id,
                        "seed": seed,
                        "summary": rep.get("summary", {}),
                        "result_file": str(out_file),
                    }
                )

    matrix_agg = aggregate_reports(cell_reports, mode)
    final_report = {
        "protocol_file": str(args.protocol),
        "tasks_file": str(args.tasks),
        "mode": mode,
        "agent_mode": args.agent_mode,
        "runs_per_cell": runs_per_cell,
        "registered_agents": list_agent_metadata(),
        "registered_attacks": list_attack_metadata(),
        "registered_defenses": list_defense_metadata(),
        "registered_skills": list_skill_metadata(),
        "registered_tools": list_tool_metadata(),
        "cells": cell_reports,
        **matrix_agg,
    }
    report_file = args.output_dir / "matrix_report.json"
    report_file.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    cell_summary_rows = list(final_report.get("cell_summary", {}).values())
    if cell_summary_rows:
        cell_fieldnames = [
            "attack_id",
            "defense_id",
            "runs",
            "safety_mean",
            "utility_mean",
            "attack_success_rate_mean",
            "unsafe_selection_rate_mean",
            "violation_rate_mean",
            "task_success_rate_mean",
            "original_goal_success_rate_mean",
            "over_refusal_rate_mean",
            "necessary_refusal_rate_mean",
            "avg_steps_mean",
            "avg_regens_mean",
            "tool_success_rate_mean",
            "tool_failure_rate_mean",
            "auroc_violation_prediction_mean",
            "early_warning_lead_time_mean",
        ]
        _write_csv(args.output_dir / "cell_summary.csv", cell_summary_rows, cell_fieldnames)
    tool_rows = _flatten_tool_metrics_for_csv(final_report.get("cell_summary", {}))
    if tool_rows:
        tool_fieldnames = [
            "attack_id",
            "defense_id",
            "tool_id",
            "category",
            "risk_level",
            "selected_count",
            "execution_count",
            "success_count",
            "failure_count",
            "violation_count",
            "success_rate",
            "failure_rate",
            "violation_rate",
        ]
        _write_csv(args.output_dir / "tool_metrics.csv", tool_rows, tool_fieldnames)
    skill_rows = _flatten_skill_metrics_for_csv(final_report.get("cell_summary", {}))
    if skill_rows:
        skill_fieldnames = [
            "attack_id",
            "defense_id",
            "skill_id",
            "task_count",
            "step_count",
            "attacked_task_count",
            "task_success_rate",
            "original_goal_success_rate",
            "attack_success_rate",
            "violation_rate",
            "over_refusal_rate",
            "necessary_refusal_rate",
            "avg_risk_score",
        ]
        _write_csv(args.output_dir / "skill_metrics.csv", skill_rows, skill_fieldnames)
    print(f"Saved matrix report: {report_file}")


if __name__ == "__main__":
    main()
