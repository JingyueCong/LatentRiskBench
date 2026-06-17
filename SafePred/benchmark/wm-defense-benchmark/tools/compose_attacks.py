#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from attacks import get_attack_handler, list_attack_metadata
from schemas.base_task_schema import validate_base_task
from schemas.payload_schema import validate_payload


def parse_args() -> argparse.Namespace:
    root = BENCHMARK_ROOT
    data_root = root / "data"
    parser = argparse.ArgumentParser(description="Compose attacked tasks from benign/base tasks.")
    parser.add_argument("--base-tasks", type=Path, default=data_root / "base_tasks.jsonl")
    parser.add_argument("--attack-payload-dir", type=Path, default=data_root / "attack_payloads")
    parser.add_argument("--protocol", type=Path, default=root / "eval_protocol.yaml")
    parser.add_argument("--output", type=Path, default=data_root / "tasks_composed.jsonl")
    parser.add_argument(
        "--attacks",
        nargs="*",
        default=None,
        help="Optional explicit attack ids. Defaults to protocol matrix.attacks.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def load_protocol(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Protocol file must be a YAML mapping.")
    return data


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_attack_payloads(payload_dir: Path) -> List[Dict[str, Any]]:
    if not payload_dir.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for path in sorted(p for p in payload_dir.rglob("*.jsonl") if p.is_file()):
        for idx, payload in enumerate(load_jsonl(path), start=1):
            errors = validate_payload(payload, source=f"{path}:{idx}")
            if errors:
                raise ValueError("Invalid payload schema:\n" + "\n".join(errors))
            rows.append(payload)
    return rows


def payload_matches_attack(payload: Dict[str, Any], attack_id: str, attack_family: str) -> bool:
    payload_attack_id = str(payload.get("attack_id", ""))
    payload_attack_family = str(payload.get("attack_family", ""))
    return payload_attack_id == attack_id or (payload_attack_family and payload_attack_family == attack_family)


def main() -> None:
    args = parse_args()
    base_tasks = load_jsonl(args.base_tasks)
    for idx, task in enumerate(base_tasks, start=1):
        errors = validate_base_task(task, source=f"{args.base_tasks}:{idx}")
        if errors:
            raise ValueError("Invalid base task schema:\n" + "\n".join(errors))
    payloads = load_attack_payloads(args.attack_payload_dir)
    protocol = load_protocol(args.protocol)
    attack_ids = args.attacks or protocol.get("matrix", {}).get("attacks", ["attack_none"])

    composed: List[Dict[str, Any]] = []
    for task in base_tasks:
        for attack_id in attack_ids:
            handler = get_attack_handler({"attack_id": attack_id})
            if attack_id in {"attack_none", "none"}:
                transformed = handler.apply_to_task(task)
                if transformed is not None:
                    composed.append(transformed)
                continue

            matched_payloads = [
                payload
                for payload in payloads
                if payload_matches_attack(payload, handler.attack_id, handler.attack_family)
            ]
            if matched_payloads:
                for payload in matched_payloads:
                    transformed = handler.apply_to_task(task, payload=payload)
                    if transformed is not None:
                        composed.append(transformed)
                continue

            transformed = handler.apply_to_task(task)
            if transformed is not None:
                composed.append(transformed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output, composed)
    print(json.dumps(
        {
            "saved_tasks": len(composed),
            "output": str(args.output),
            "registered_attacks": list_attack_metadata(),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
