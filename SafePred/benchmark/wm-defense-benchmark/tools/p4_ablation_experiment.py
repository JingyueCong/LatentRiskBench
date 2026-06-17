#!/usr/bin/env python3
"""P4: per-axis ablation experiment for the 5-axis attacker taxonomy.

Runs five factorial cells that vary attacker knowledge along the axes
defined in docs/wm_0417.md §C.6, keeping defender and judge fixed
(DeepSeek WMG as the defender-under-test, rule+trajectory oracle as
ground truth). The experiment quantifies **per-axis contribution** to
adversarial success — the first real empirical support for the claim
that the 5-axis taxonomy isn't just a relabelling of white/grey/black
but a decomposition with measurable impact.

Design: same attacker model (deepseek-chat) and same defender model
(deepseek-chat via SafePredWrapper) across all cells; the only thing
varying between cells is the attacker's knowledge profile. This
isolates axis effects from model-identity effects.

Cells (each differs from baseline by exactly the specified axes):
    baseline   all-off: A1=none  A2=binary  A3=none    A4=hidden  A5=unknown
    +A1        A1=wrapper  (others same as baseline)
    +A3        A3=full     (others same as baseline)
    +A1+A3     A1=wrapper AND A3=full
    all_on     A1=wrapper A2=scalar A3=full A4=visible A5=known (≡ white_box)

The baseline deliberately matches the axis-derived black_box preset
(no defender knowledge at all) while keeping the SAME model identity.
This is only possible because Phase P4's axis-derived black_box path
relaxes the model-mismatch enforcement (see attacker_wm.py).

Per cell we:
  1. Generate up to K attack payloads under that axis configuration,
  2. Evaluate each payload against defense_world_model_gating
     (DeepSeek WMG via SafePredWrapper) — the real defender,
  3. Record fooling rate (defender picked unsafe action / payloads).

Output: JSON + CSV table per cell, plus a headline summary with
per-axis attribution (baseline→+A1 delta = A1 contribution, etc.).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


CELLS: List[Dict[str, Any]] = [
    {
        "name": "baseline",
        "description": "all axes off (A1=none, A3=none, A5=unknown, A2=binary, A4=hidden)",
        "axis_flags": [
            "--axis-rollout", "none",
            "--axis-policy", "none",
            "--axis-budget", "unknown",
            "--axis-risk-score", "binary",
            "--axis-pre-filter", "hidden",
        ],
        "use_llm_defender": False,  # rollout=none → no LLM defender scorer
    },
    {
        "name": "a1_rollout",
        "description": "+A1 rollout=wrapper (attacker sees eval-time path)",
        "axis_flags": [
            "--axis-rollout", "wrapper",
            "--axis-policy", "none",
            "--axis-budget", "unknown",
            "--axis-risk-score", "binary",
            "--axis-pre-filter", "hidden",
        ],
        "use_llm_defender": True,
    },
    {
        "name": "a3_policy",
        "description": "+A3 policy=full (attacker sees policy text in prompt)",
        "axis_flags": [
            "--axis-rollout", "none",
            "--axis-policy", "full",
            "--axis-budget", "unknown",
            "--axis-risk-score", "binary",
            "--axis-pre-filter", "hidden",
        ],
        "use_llm_defender": False,
    },
    {
        "name": "a1_plus_a3",
        "description": "+A1+A3 (pipeline + policy together)",
        "axis_flags": [
            "--axis-rollout", "wrapper",
            "--axis-policy", "full",
            "--axis-budget", "unknown",
            "--axis-risk-score", "binary",
            "--axis-pre-filter", "hidden",
        ],
        "use_llm_defender": True,
    },
    {
        "name": "all_on",
        "description": "all axes on ≡ white_box",
        "axis_flags": [
            "--axis-rollout", "wrapper",
            "--axis-policy", "full",
            "--axis-budget", "known",
            "--axis-risk-score", "scalar",
            "--axis-pre-filter", "visible",
        ],
        "use_llm_defender": True,
    },
]


def _run(cmd: List[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a subprocess with explicit error surfacing."""
    print(f"\n> {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(BENCHMARK_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _generate_cell(
    cell: Dict[str, Any],
    *,
    base_tasks: List[str],
    n_variants: int,
    max_payloads: int,
    max_llm_calls: int,
) -> Dict[str, Any]:
    """Run attack_generator.cli for one cell. Archive payloads + return summary."""
    payloads_root = BENCHMARK_ROOT / "data" / "attack_payloads"
    wm_adaptive_dir = payloads_root / "wm_adaptive"
    logs_dir = BENCHMARK_ROOT / "data" / "attack_generation_logs"
    cell_dir = payloads_root / f"p4_cell_{cell['name']}"

    # Clear per-cell state (wm_adaptive is mutable scratch space).
    for path in (wm_adaptive_dir, logs_dir):
        if path.exists():
            shutil.rmtree(path)

    base_args = []
    for b in base_tasks:
        base_args.extend(["--base-task-id", b])

    cmd = [
        sys.executable, "-m", "attack_generator.cli",
        *base_args,
        "--rule-oracle-only",
        "--threat-mode", "white_box",  # starts from white_box; axes override selectively
        "--algorithm", "rephrase",
        "--use-llm-attacker", "--attacker-backend", "deepseek",
        "--n-variants", str(n_variants),
        "--max-payloads", str(max_payloads),
        "--max-llm-calls", str(max_llm_calls),
        "--wall-time-sec", "600",
        *cell["axis_flags"],
    ]
    if cell["use_llm_defender"]:
        cmd.extend(["--use-llm-defender", "--defender-backend", "deepseek",
                    "--defender-scorer-mode", "wrapper"])

    result = _run(cmd, timeout=1800)
    generated = []
    if wm_adaptive_dir.exists():
        generated = sorted(p.name for p in wm_adaptive_dir.glob("*.jsonl"))
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True)
        for jsonl in wm_adaptive_dir.glob("*.jsonl"):
            shutil.copy(jsonl, cell_dir / jsonl.name)

    return {
        "cell": cell["name"],
        "description": cell["description"],
        "cmd_exit_code": result.returncode,
        "n_payload_files": len(generated),
        "archive_dir": str(cell_dir) if cell_dir.exists() else None,
        "stderr_tail": result.stderr.splitlines()[-5:] if result.stderr else [],
    }


def _eval_cell_vs_wmg(
    cell: Dict[str, Any], *, output_dir: Path
) -> Dict[str, Any]:
    """Run tools/evaluate_transfer_matrix.py on a cell's archived payloads,
    with WM-gating defender enabled."""
    cell_dir = BENCHMARK_ROOT / "data" / "attack_payloads" / f"p4_cell_{cell['name']}"
    if not cell_dir.exists() or not any(cell_dir.glob("*.jsonl")):
        return {"cell": cell["name"], "status": "skipped_empty", "fooling_rates": {}}
    out_json = output_dir / f"p4_{cell['name']}_matrix.json"
    out_csv = output_dir / f"p4_{cell['name']}_matrix.csv"
    cmd = [
        sys.executable, "tools/evaluate_transfer_matrix.py",
        "--payloads-root", str(cell_dir),
        "--include-world-model", "--defender-backend", "deepseek",
        "--output", str(out_json),
        "--csv", str(out_csv),
    ]
    result = _run(cmd, timeout=1800)
    rates: Dict[str, float] = {}
    if out_json.exists():
        with out_json.open("r", encoding="utf-8") as f:
            report = json.load(f)
        by_mode = report.get("summary", {}).get("by_cell", {}) or {}
        for key, cell_summary in by_mode.items():
            defense_id = cell_summary.get("defense_id", key)
            rates[defense_id] = float(cell_summary.get("fooling_rate", 0.0))
    return {
        "cell": cell["name"],
        "status": "ok" if result.returncode == 0 else "error",
        "fooling_rates": rates,
        "stderr_tail": result.stderr.splitlines()[-5:] if result.stderr else [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P4 ablation runner.")
    p.add_argument(
        "--base-tasks",
        nargs="+",
        default=["base_doc_share", "base_support_triage", "base_log_archive"],
    )
    p.add_argument("--n-variants", type=int, default=3)
    p.add_argument("--max-payloads", type=int, default=2)
    p.add_argument("--max-llm-calls", type=int, default=30)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_ROOT / "data" / "results_p4_ablation",
    )
    p.add_argument(
        "--only-cells",
        nargs="+",
        default=None,
        help="Subset of cell names to run (default: all 5). Useful for resume.",
    )
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation and only run evaluation on previously archived cells.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation and only generate attacks per cell.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells_to_run = CELLS
    if args.only_cells:
        cells_to_run = [c for c in CELLS if c["name"] in set(args.only_cells)]

    gen_summaries: List[Dict[str, Any]] = []
    eval_summaries: List[Dict[str, Any]] = []

    for cell in cells_to_run:
        print(f"\n{'='*72}\n[P4] cell={cell['name']}  desc={cell['description']}\n{'='*72}", flush=True)
        if not args.skip_generation:
            gen_summaries.append(
                _generate_cell(
                    cell,
                    base_tasks=args.base_tasks,
                    n_variants=args.n_variants,
                    max_payloads=args.max_payloads,
                    max_llm_calls=args.max_llm_calls,
                )
            )
        if not args.skip_eval:
            eval_summaries.append(_eval_cell_vs_wmg(cell, output_dir=args.output_dir))

    # Aggregate: per-axis attribution table.
    rates_by_cell: Dict[str, Dict[str, float]] = {}
    for es in eval_summaries:
        rates_by_cell[es["cell"]] = es.get("fooling_rates") or {}

    def _rate(cell_name: str, defense_id: str = "defense_world_model_gating") -> Optional[float]:
        v = rates_by_cell.get(cell_name, {}).get(defense_id)
        return None if v is None else round(float(v), 4)

    baseline_asr = _rate("baseline")
    per_axis_attribution = {
        "baseline_asr_vs_wmg": baseline_asr,
        "delta_a1_rollout":    None if (baseline_asr is None or _rate("a1_rollout") is None)
                               else round(_rate("a1_rollout") - baseline_asr, 4),
        "delta_a3_policy":     None if (baseline_asr is None or _rate("a3_policy") is None)
                               else round(_rate("a3_policy") - baseline_asr, 4),
        "delta_a1_plus_a3":    None if (baseline_asr is None or _rate("a1_plus_a3") is None)
                               else round(_rate("a1_plus_a3") - baseline_asr, 4),
        "delta_all_on":        None if (baseline_asr is None or _rate("all_on") is None)
                               else round(_rate("all_on") - baseline_asr, 4),
        # Interaction check: A1+A3 vs A1-only and A3-only.
        "nonadditive_a1_a3": None,
    }
    if all(_rate(n) is not None for n in ("baseline", "a1_rollout", "a3_policy", "a1_plus_a3")):
        base = _rate("baseline")
        da1 = _rate("a1_rollout") - base
        da3 = _rate("a3_policy") - base
        combined = _rate("a1_plus_a3") - base
        per_axis_attribution["nonadditive_a1_a3"] = round(combined - (da1 + da3), 4)

    report = {
        "base_tasks": args.base_tasks,
        "cells_run": [c["name"] for c in cells_to_run],
        "n_variants": args.n_variants,
        "max_payloads": args.max_payloads,
        "generation_summaries": gen_summaries,
        "eval_summaries": eval_summaries,
        "asr_by_cell_and_defense": rates_by_cell,
        "per_axis_attribution": per_axis_attribution,
    }

    report_path = args.output_dir / "p4_ablation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72, flush=True)
    print("P4 ablation summary:", flush=True)
    print(json.dumps(
        {
            "asr_by_cell_vs_wmg": {
                name: rates_by_cell.get(name, {}).get("defense_world_model_gating")
                for name in (c["name"] for c in cells_to_run)
            },
            "per_axis_attribution": per_axis_attribution,
            "report_path": str(report_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
