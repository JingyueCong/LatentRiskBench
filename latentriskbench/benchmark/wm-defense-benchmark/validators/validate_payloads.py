#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from schemas.payload_schema import validate_payload_file


def parse_args() -> argparse.Namespace:
    data_root = BENCHMARK_ROOT / "data"
    parser = argparse.ArgumentParser(description="Validate attack payload files against the benchmark schema.")
    parser.add_argument(
        "--attack-payload-dir",
        type=Path,
        default=data_root / "attack_payloads",
        help="Root directory containing attack-family payload subdirectories.",
    )
    return parser.parse_args()


def iter_payload_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def main() -> None:
    args = parse_args()
    payload_files = iter_payload_files(args.attack_payload_dir)
    all_errors = []
    for path in payload_files:
        all_errors.extend(validate_payload_file(path))

    output = {
        "payload_root": str(args.attack_payload_dir),
        "files_checked": [str(path) for path in payload_files],
        "valid": not all_errors,
        "error_count": len(all_errors),
        "errors": all_errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
