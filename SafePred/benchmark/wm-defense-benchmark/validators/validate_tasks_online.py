#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from schemas.explicit_task_schema import validate_online_task_file


def parse_args() -> argparse.Namespace:
    data_root = BENCHMARK_ROOT / "data"
    parser = argparse.ArgumentParser(description="Validate explicit online task files.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=data_root / "tasks_online.jsonl",
        help="Path to the explicit online task JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_online_task_file(args.tasks)
    output = {
        "tasks_file": str(args.tasks),
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
