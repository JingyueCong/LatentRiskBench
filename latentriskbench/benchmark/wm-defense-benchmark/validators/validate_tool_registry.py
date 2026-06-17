#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from schemas.tool_registry_schema import validate_tool_registry_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the tool registry file.")
    parser.add_argument(
        "--tool-registry",
        type=Path,
        default=BENCHMARK_ROOT / "data" / "tool_registry.json",
        help="Path to tool_registry.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_tool_registry_file(args.tool_registry)
    output = {
        "tool_registry_file": str(args.tool_registry),
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
