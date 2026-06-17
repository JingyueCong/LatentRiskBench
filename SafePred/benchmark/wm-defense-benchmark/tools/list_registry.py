#!/usr/bin/env python3
import argparse
import json
import sys

from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from agents import list_agent_metadata
from attacks import list_attack_metadata
from defenses import list_defense_metadata
from skills import list_skill_metadata
from tooling import list_tool_executor_ids, list_tool_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List registered attacks and defenses.")
    parser.add_argument(
        "--kind",
        choices=["attacks", "agents", "defenses", "skills", "tools", "tool_executors", "all"],
        default="all",
        help="Which registry to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = {}
    if args.kind in {"attacks", "all"}:
        output["attacks"] = list_attack_metadata()
    if args.kind in {"agents", "all"}:
        output["agents"] = list_agent_metadata()
    if args.kind in {"defenses", "all"}:
        output["defenses"] = list_defense_metadata()
    if args.kind in {"skills", "all"}:
        output["skills"] = list_skill_metadata()
    if args.kind in {"tools", "all"}:
        output["tools"] = list_tool_metadata()
    if args.kind in {"tool_executors", "all"}:
        output["tool_executors"] = list_tool_executor_ids()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
