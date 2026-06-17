#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from schemas.base_task_schema import validate_base_task_file


def parse_args() -> argparse.Namespace:
    data_root = BENCHMARK_ROOT / "data"
    parser = argparse.ArgumentParser(description="Validate base task files against the benchmark schema.")
    parser.add_argument(
        "--base-tasks",
        type=Path,
        default=data_root / "base_tasks.jsonl",
        help="Path to the base task JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_base_task_file(args.base_tasks)
    output = {
        "base_tasks_file": str(args.base_tasks),
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
