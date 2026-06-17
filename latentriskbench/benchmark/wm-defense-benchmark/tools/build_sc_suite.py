#!/usr/bin/env python3
"""Build the state-conditioned (SC) suite from benign seeds via the top-down injector.

For every (seed, target) it constructs a risky item and its constant-action benign twin,
certifies them by-construction with the struct_v1 oracle (risky flags exactly the target
policy; twin is clean; terminal action identical; the leverage field is the only STATE-ONLY
difference), and writes paired audit items to data/tasks_sc.jsonl.

Usage:
  python tools/build_sc_suite.py
  python tools/build_sc_suite.py --seeds data/sc_seeds.jsonl --out data/tasks_sc.jsonl
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

from attack_generator.sc_injector import build_pair, load_seeds, SCInjectionError
from schemas.state_schema import canonical_state, STATE_ONLY_PATHS


def diff_paths(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """STATE-ONLY paths whose canonical value differs between two states."""
    ca, cb = dict(canonical_state(a)), dict(canonical_state(b))
    return [p for p in STATE_ONLY_PATHS if ca.get(p) != cb.get(p)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=BENCHMARK_ROOT / "data" / "sc_seeds.jsonl")
    ap.add_argument("--out", type=Path, default=BENCHMARK_ROOT / "data" / "tasks_sc.jsonl")
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    items: List[Dict[str, Any]] = []
    n_pairs = 0
    print(f"{'pair':40} {'policy':6} {'mechanism':24} leverage-diff")
    print("-" * 100)
    for seed in seeds:
        for target in seed.get("targets", []):
            try:
                risky, twin = build_pair(seed, target)
            except SCInjectionError as exc:
                print(f"  [FAIL] {seed['seed_id']}/{target['id']}: {exc}")
                raise
            d = diff_paths(risky["state_struct"], twin["state_struct"])
            leverage_ok = target["path"] in d  # leverage field must differ
            action_ok = risky["terminal_action"] == twin["terminal_action"]
            flag = "OK " if (leverage_ok and action_ok) else "WARN"
            print(f"[{flag}] {seed['seed_id']+'/'+target['id']:34} {target['policy']:6} {target['mechanism']:24} {d}")
            items.extend([risky, twin])
            n_pairs += 1

    args.out.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")

    risky_n = sum(1 for it in items if it["label"]["risky"])
    print("-" * 100)
    print(f"Pairs: {n_pairs}  | items: {len(items)} ({risky_n} risky + {len(items)-risky_n} benign twins)")
    by_mech: Dict[str, int] = {}
    for it in items:
        if it["label"]["risky"]:
            by_mech[it["label"]["mechanism"]] = by_mech.get(it["label"]["mechanism"], 0) + 1
    print("Mechanism balance (risky):", by_mech)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
