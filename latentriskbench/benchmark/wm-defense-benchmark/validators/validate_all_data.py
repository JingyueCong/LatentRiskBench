#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from schemas.base_task_schema import validate_base_task_file
from schemas.explicit_task_schema import validate_offline_task_file, validate_online_task_file
from schemas.payload_schema import validate_payload_file
from schemas.tool_registry_schema import validate_tool_registry_file
from policy_oracle import RulePolicyOracle, load_policies


def parse_args() -> argparse.Namespace:
    data_root = BENCHMARK_ROOT / "data"
    parser = argparse.ArgumentParser(description="Validate all benchmark input data files.")
    parser.add_argument("--base-tasks", type=Path, default=data_root / "base_tasks.jsonl")
    parser.add_argument("--tasks", type=Path, default=data_root / "tasks.jsonl")
    parser.add_argument("--tasks-online", type=Path, default=data_root / "tasks_online.jsonl")
    parser.add_argument("--attack-payload-dir", type=Path, default=data_root / "attack_payloads")
    parser.add_argument("--tool-registry", type=Path, default=data_root / "tool_registry.json")
    parser.add_argument("--policies", type=Path, default=data_root / "policies.json")
    parser.add_argument(
        "--skip-oracle-coverage",
        action="store_true",
        help="Skip the policy-oracle coverage check on tasks.jsonl and tasks_online.jsonl.",
    )
    return parser.parse_args()


def _oracle_coverage_errors(
    oracle: RulePolicyOracle,
    policies: list[dict],
    tasks_path: Path,
    mode: str,
) -> list[str]:
    """Run the rule oracle over labelled unsafe/safe actions and report mismatches."""
    if not tasks_path.exists():
        return [f"Missing file: {tasks_path}"]
    errors: list[str] = []
    with tasks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{tasks_path}:{line_no} invalid JSON: {exc}")
                continue
            intent = str(task.get("intent", ""))
            task_id = str(task.get("task_id", f"line{line_no}"))
            if mode == "online":
                for step_idx, step in enumerate(task.get("steps", []) or []):
                    state = step.get("state") or task.get("initial_state", {}) or {}
                    labelled = set(int(i) for i in step.get("unsafe_action_indices", []) or [])
                    for idx, action in enumerate(step.get("candidate_actions", []) or []):
                        verdict = oracle.evaluate(intent, state, str(action), policies)
                        if (idx in labelled) and not verdict.unsafe:
                            errors.append(
                                f"{task_id} step{step_idx} [{idx}]: labelled unsafe but oracle missed: {action}"
                            )
                        elif (idx not in labelled) and verdict.unsafe:
                            errors.append(
                                f"{task_id} step{step_idx} [{idx}]: labelled safe but oracle flagged: {action}"
                            )
            else:
                state = task.get("state", {}) or {}
                labelled = set(int(i) for i in task.get("unsafe_action_indices", []) or [])
                for idx, action in enumerate(task.get("candidate_actions", []) or []):
                    verdict = oracle.evaluate(intent, state, str(action), policies)
                    if (idx in labelled) and not verdict.unsafe:
                        errors.append(f"{task_id}[{idx}]: labelled unsafe but oracle missed: {action}")
                    elif (idx not in labelled) and verdict.unsafe:
                        errors.append(f"{task_id}[{idx}]: labelled safe but oracle flagged: {action}")
    return errors


def iter_payload_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def main() -> None:
    args = parse_args()
    payload_files = iter_payload_files(args.attack_payload_dir)

    sections = []

    base_errors = validate_base_task_file(args.base_tasks) if args.base_tasks.exists() else [f"Missing file: {args.base_tasks}"]
    sections.append(
        {
            "kind": "base_tasks",
            "path": str(args.base_tasks),
            "valid": not base_errors,
            "error_count": len(base_errors),
            "errors": base_errors,
        }
    )

    offline_errors = validate_offline_task_file(args.tasks) if args.tasks.exists() else [f"Missing file: {args.tasks}"]
    sections.append(
        {
            "kind": "tasks_offline",
            "path": str(args.tasks),
            "valid": not offline_errors,
            "error_count": len(offline_errors),
            "errors": offline_errors,
        }
    )

    online_errors = validate_online_task_file(args.tasks_online) if args.tasks_online.exists() else [f"Missing file: {args.tasks_online}"]
    sections.append(
        {
            "kind": "tasks_online",
            "path": str(args.tasks_online),
            "valid": not online_errors,
            "error_count": len(online_errors),
            "errors": online_errors,
        }
    )

    payload_errors = []
    if not args.attack_payload_dir.exists():
        payload_errors.append(f"Missing directory: {args.attack_payload_dir}")
    else:
        for path in payload_files:
            payload_errors.extend(validate_payload_file(path))
    sections.append(
        {
            "kind": "attack_payloads",
            "path": str(args.attack_payload_dir),
            "files_checked": [str(path) for path in payload_files],
            "valid": not payload_errors,
            "error_count": len(payload_errors),
            "errors": payload_errors,
        }
    )

    tool_registry_errors = (
        validate_tool_registry_file(args.tool_registry)
        if args.tool_registry.exists()
        else [f"Missing file: {args.tool_registry}"]
    )
    sections.append(
        {
            "kind": "tool_registry",
            "path": str(args.tool_registry),
            "valid": not tool_registry_errors,
            "error_count": len(tool_registry_errors),
            "errors": tool_registry_errors,
        }
    )

    if not args.skip_oracle_coverage:
        if not args.policies.exists():
            oracle_errors = [f"Missing file: {args.policies}"]
            oracle_sources: list[str] = []
        else:
            oracle = RulePolicyOracle()
            policies = load_policies(args.policies)
            oracle_errors = []
            oracle_sources = []
            if args.tasks.exists():
                oracle_sources.append(str(args.tasks))
                oracle_errors.extend(
                    _oracle_coverage_errors(oracle, policies, args.tasks, "offline")
                )
            if args.tasks_online.exists():
                oracle_sources.append(str(args.tasks_online))
                oracle_errors.extend(
                    _oracle_coverage_errors(oracle, policies, args.tasks_online, "online")
                )
        sections.append(
            {
                "kind": "oracle_coverage",
                "policies": str(args.policies),
                "tasks_checked": oracle_sources,
                "valid": not oracle_errors,
                "error_count": len(oracle_errors),
                "errors": oracle_errors,
            }
        )

    total_errors = sum(section["error_count"] for section in sections)
    output = {
        "valid": total_errors == 0,
        "total_error_count": total_errors,
        "sections": sections,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
