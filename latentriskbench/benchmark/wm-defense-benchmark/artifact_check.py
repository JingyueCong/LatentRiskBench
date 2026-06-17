#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from agents import list_agent_metadata
from attacks import list_attack_metadata
from defenses import list_defense_metadata
from schemas.base_task_schema import validate_base_task_file
from schemas.explicit_task_schema import validate_offline_task_file, validate_online_task_file
from schemas.payload_schema import validate_payload_file
from schemas.tool_registry_schema import validate_tool_registry_file
from tooling import list_tool_executor_ids, list_tool_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-submission artifact self-checks.")
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="Skip running paper_supplement/package_submission.sh.",
    )
    parser.add_argument(
        "--skip-smoke-run",
        action="store_true",
        help="Skip the single-task benchmark smoke run.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["first_candidate", "keyword_guarded", "heuristic_ranker", "replay"],
        default="first_candidate",
        help="Agent mode used for the smoke run.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=1,
        help="Number of tasks used in the smoke run.",
    )
    return parser.parse_args()


def _iter_payload_files(payload_root: Path) -> List[Path]:
    if not payload_root.exists():
        return []
    return sorted(path for path in payload_root.rglob("*.jsonl") if path.is_file())


def check_structure() -> Dict[str, Any]:
    required_paths = {
        "data": BENCHMARK_ROOT / "data",
        "schemas": BENCHMARK_ROOT / "schemas",
        "validators": BENCHMARK_ROOT / "validators",
        "runners": BENCHMARK_ROOT / "runners",
        "tools": BENCHMARK_ROOT / "tools",
        "tooling": BENCHMARK_ROOT / "tooling",
        "artifact_check": BENCHMARK_ROOT / "artifact_check.py",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    return {
        "kind": "structure",
        "valid": not missing,
        "checked": {name: str(path) for name, path in required_paths.items()},
        "missing": missing,
    }


def check_registry() -> Dict[str, Any]:
    attacks = list_attack_metadata()
    agents = list_agent_metadata()
    defenses = list_defense_metadata()
    tools = list_tool_metadata()
    tool_executor_ids = set(list_tool_executor_ids())
    issues: List[str] = []
    if not attacks:
        issues.append("No registered attacks found.")
    if not agents:
        issues.append("No registered agents found.")
    if not defenses:
        issues.append("No registered defenses found.")
    if not tools:
        issues.append("No registered tools found.")
    missing_executors = [
        str(tool.get("tool_id"))
        for tool in tools
        if str(tool.get("tool_id")) not in tool_executor_ids
    ]
    if missing_executors:
        issues.append("Missing tool executors for: " + ", ".join(sorted(missing_executors)))
    return {
        "kind": "registry",
        "valid": not issues,
        "counts": {
            "attacks": len(attacks),
            "agents": len(agents),
            "defenses": len(defenses),
            "tools": len(tools),
            "tool_executors": len(tool_executor_ids),
        },
        "issues": issues,
    }


def check_data() -> Dict[str, Any]:
    data_root = BENCHMARK_ROOT / "data"
    payload_root = data_root / "attack_payloads"
    tool_registry_path = data_root / "tool_registry.json"
    payload_files = _iter_payload_files(payload_root)

    sections = []
    base_errors = validate_base_task_file(data_root / "base_tasks.jsonl")
    sections.append(
        {
            "kind": "base_tasks",
            "path": str(data_root / "base_tasks.jsonl"),
            "valid": not base_errors,
            "error_count": len(base_errors),
        }
    )

    offline_errors = validate_offline_task_file(data_root / "tasks.jsonl")
    sections.append(
        {
            "kind": "tasks_offline",
            "path": str(data_root / "tasks.jsonl"),
            "valid": not offline_errors,
            "error_count": len(offline_errors),
        }
    )

    online_errors = validate_online_task_file(data_root / "tasks_online.jsonl")
    sections.append(
        {
            "kind": "tasks_online",
            "path": str(data_root / "tasks_online.jsonl"),
            "valid": not online_errors,
            "error_count": len(online_errors),
        }
    )

    payload_errors: List[str] = []
    for payload_file in payload_files:
        payload_errors.extend(validate_payload_file(payload_file))
    sections.append(
        {
            "kind": "attack_payloads",
            "path": str(payload_root),
            "files_checked": [str(path) for path in payload_files],
            "valid": not payload_errors,
            "error_count": len(payload_errors),
        }
    )

    tool_registry_errors = validate_tool_registry_file(tool_registry_path)
    sections.append(
        {
            "kind": "tool_registry",
            "path": str(tool_registry_path),
            "valid": not tool_registry_errors,
            "error_count": len(tool_registry_errors),
        }
    )

    all_errors = base_errors + offline_errors + online_errors + payload_errors + tool_registry_errors
    return {
        "kind": "data_validation",
        "valid": not all_errors,
        "total_error_count": len(all_errors),
        "sections": sections,
        "errors": all_errors,
    }


def check_smoke_run(agent_mode: str, max_tasks: int) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="wmdefense_artifact_check_") as tmpdir:
        output_path = Path(tmpdir) / "smoke_result.json"
        cmd = [
            sys.executable,
            str(BENCHMARK_ROOT / "runners" / "run.py"),
            "--validate-inputs",
            "--defense-mode",
            "none",
            "--agent-mode",
            agent_mode,
            "--max-tasks",
            str(max_tasks),
            "--output",
            str(output_path),
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result: Dict[str, Any] = {
            "kind": "smoke_run",
            "valid": proc.returncode == 0 and output_path.exists(),
            "command": cmd,
            "returncode": proc.returncode,
        }
        if proc.stdout:
            result["stdout_tail"] = proc.stdout.strip().splitlines()[-10:]
        if proc.stderr:
            result["stderr_tail"] = proc.stderr.strip().splitlines()[-10:]
        if proc.returncode == 0 and output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            result["summary"] = payload.get("summary", {})
        return result


def check_package() -> Dict[str, Any]:
    package_script = REPO_ROOT / "paper_supplement" / "package_submission.sh"
    tarball = REPO_ROOT / "paper_supplement" / "wm-defense-benchmark-supplement.tar.gz"
    proc = subprocess.run(
        ["bash", str(package_script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    result: Dict[str, Any] = {
        "kind": "package_submission",
        "valid": proc.returncode == 0 and tarball.exists(),
        "script": str(package_script),
        "tarball": str(tarball),
        "returncode": proc.returncode,
    }
    if proc.stdout:
        result["stdout_tail"] = proc.stdout.strip().splitlines()[-10:]
    if proc.stderr:
        result["stderr_tail"] = proc.stderr.strip().splitlines()[-10:]
    if tarball.exists():
        result["tarball_size_bytes"] = tarball.stat().st_size
    return result


def main() -> None:
    args = parse_args()
    checks = [
        check_structure(),
        check_registry(),
        check_data(),
    ]
    if args.skip_smoke_run:
        checks.append({"kind": "smoke_run", "valid": True, "skipped": True})
    else:
        checks.append(check_smoke_run(args.agent_mode, args.max_tasks))

    if args.skip_package:
        checks.append({"kind": "package_submission", "valid": True, "skipped": True})
    else:
        checks.append(check_package())

    valid = all(check.get("valid", False) for check in checks)
    output = {
        "valid": valid,
        "benchmark_root": str(BENCHMARK_ROOT),
        "checks": checks,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
