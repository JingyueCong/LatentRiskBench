"""Unified ablation matrix across Phase-C (beam rollout_mode) and
Phase-D (rephrase state_override) configurations.

Replaces ``tools/compare_rollout_modes.py`` and ``tools/compare_composition.py``.

A "cell" is one attacker configuration. Each cell runs on the same base
tasks + seeds against the same defender and emits a normalised metric
schema, so cells can be compared side-by-side and pairwise action-set
Jaccards establish which cells cover distinct attack spaces.

Cells:
    beam_score_only     Beam generator, rollout_mode=False (pre-Phase-C).
    beam_rollout        Beam generator, rollout_mode=True  (Phase C).
    rephrase_vanilla    Rephrase generator, no state_override (pre-Phase-D).
    rephrase_composed   Rephrase generator, state_override=BeamNode.predicted_state
                        from a precursor rollout=on beam (Phase D).

Presets:
    --preset a1_use_mode        = beam_score_only, beam_rollout
    --preset a1_context_state   = rephrase_vanilla, rephrase_composed
    --preset full_a1            = all four

Usage (Template, $0 sanity):
    python tools/ablation_matrix.py --preset full_a1 --max-base-tasks 10 \
        --output /tmp/ablation_template.json

Usage (DeepSeek LLM attacker, small scale):
    python tools/ablation_matrix.py --preset full_a1 \
        --attacker llm --attacker-backend deepseek \
        --max-base-tasks 10 \
        --output data/results_ablation_llm.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCHMARK_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from attack_generator import (  # noqa: E402
    BeamGenerator,
    Budget,
    DefenderVerdict,
    GenerationRequest,
    LLMActionProposer,
    LLMAttacker,
    RephraseGenerator,
    StubDefenderWMScorer,
    TemplateActionProposer,
    TemplateAttacker,
)
from attack_generator.action_proposer import infer_attack_intent  # noqa: E402
from attack_generator.cli import (  # noqa: E402
    _BACKEND_REGISTRY,
    _collect_seed_actions,
    _load_base_tasks,
    _resolve_backend,
)
from policy_oracle import (  # noqa: E402
    RulePolicyOracle,
    TrajectoryPolicyOracle,
    load_policies,
)

POLICIES_PATH = BENCHMARK_ROOT / "data" / "policies.json"


# ---------------------------------------------------------------------------
# Cell registry
# ---------------------------------------------------------------------------

ALL_CELLS = (
    "beam_score_only",
    "beam_rollout",
    "rephrase_vanilla",
    "rephrase_composed",
)

PRESETS: Dict[str, Tuple[str, ...]] = {
    "a1_use_mode": ("beam_score_only", "beam_rollout"),
    "a1_context_state": ("rephrase_vanilla", "rephrase_composed"),
    "full_a1": ALL_CELLS,
}


@dataclass
class CellConfig:
    cell_id: str
    generator: str  # "beam" | "rephrase"
    rollout_mode: bool  # beam only; ignored for rephrase
    use_precursor_override: bool  # rephrase only; ignored for beam

    @classmethod
    def from_id(cls, cell_id: str) -> "CellConfig":
        table = {
            "beam_score_only": cls(cell_id, "beam", False, False),
            "beam_rollout": cls(cell_id, "beam", True, False),
            "rephrase_vanilla": cls(cell_id, "rephrase", False, False),
            "rephrase_composed": cls(cell_id, "rephrase", False, True),
        }
        if cell_id not in table:
            raise ValueError(
                f"Unknown cell_id {cell_id!r}; valid: {sorted(table)}"
            )
        return table[cell_id]


@dataclass
class CellResult:
    """Normalised metrics for one cell on one (base_task, seed)."""

    source: str  # "beam_nodes" | "variant_scores"
    n_units_scored: int = 0
    n_adversarial: int = 0
    n_payloads: int = 0
    mean_oracle_confidence_adv: float = 0.0
    mean_wm_risk_score_adv: float = 0.0
    predicted_state_diversity: int = 0
    action_set: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def adversarial_hit_rate(self) -> float:
        if self.n_units_scored <= 0:
            return 0.0
        return self.n_adversarial / self.n_units_scored


# ---------------------------------------------------------------------------
# Shared defender: zero-$ synthetic delta so both rollout and composition
# exercise the fold_delta_into_state pipeline without LLM defender cost.
# ---------------------------------------------------------------------------


class SyntheticDeltaStubDefender:
    def __init__(self) -> None:
        self._call_count = 0
        self.defender_id = "stub_synthetic_delta"
        self.config: Dict[str, Any] = {
            "algorithm": "stub_synthetic_delta",
            "mode": "synthetic",
        }

    def score(
        self,
        intent: str,
        state: Dict[str, Any],
        action: str,
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> DefenderVerdict:
        budget.start()
        n = self._call_count
        self._call_count += 1
        raw_delta = {
            "semantic_delta": f"Synthetic WM step #{n} for action {action[:40]!r}",
            "element_changes": {
                "new_elements": [f"pred_marker_step_{n}"],
                "removed_elements": [],
            },
            "risk_score": 0.0,
            "violated_policy_ids": [],
        }
        return DefenderVerdict(raw_delta=raw_delta)


def _build_defender(args: argparse.Namespace) -> Any:
    if args.defender == "stub-lenient":
        return StubDefenderWMScorer(mode="lenient")
    if args.defender == "stub-strict":
        return StubDefenderWMScorer(mode="strict")
    if args.defender == "stub-synthetic":
        return SyntheticDeltaStubDefender()
    raise ValueError(f"Unknown defender {args.defender!r}")


# ---------------------------------------------------------------------------
# Attacker builders
# ---------------------------------------------------------------------------


def _build_beam_proposer(args: argparse.Namespace) -> Any:
    if args.attacker == "template":
        return TemplateActionProposer()
    backend = _resolve_backend(args.attacker_backend)
    model = args.attacker_model or backend["default_model"]
    return LLMActionProposer(
        api_key=backend["api_key"],
        api_url=backend["api_url"],
        model_name=model,
        provider=backend["sdk_provider"],
        temperature=args.attacker_temperature,
        seed=args.attacker_seed,
    )


def _build_rephrase_attacker(args: argparse.Namespace) -> Any:
    if args.attacker == "template":
        return TemplateAttacker()
    backend = _resolve_backend(args.attacker_backend)
    model = args.attacker_model or backend["default_model"]
    return LLMAttacker(
        api_key=backend["api_key"],
        api_url=backend["api_url"],
        model_name=model,
        provider=backend["sdk_provider"],
        temperature=args.attacker_temperature,
        seed=args.attacker_seed,
    )


# ---------------------------------------------------------------------------
# Per-cell runners
# ---------------------------------------------------------------------------


def _run_beam_cell(
    args: argparse.Namespace,
    cfg: CellConfig,
    defender: Any,
    policies: List[Dict[str, Any]],
    base: Dict[str, Any],
    seed: str,
    log_dir: Path,
) -> CellResult:
    proposer = _build_beam_proposer(args)
    gen = BeamGenerator(
        attacker=proposer,
        defender=defender,
        oracle=TrajectoryPolicyOracle(rule_oracle=RulePolicyOracle()),
        policies=policies,
        log_dir=log_dir,
        beam_width=args.beam_width,
        max_depth=args.beam_depth,
        branch_factor=args.branch_factor,
        default_budget=Budget(
            max_llm_calls=args.max_llm_calls,
            wall_time_sec=args.wall_time_sec,
        ),
        rollout_mode=cfg.rollout_mode,
    )
    budget = Budget(
        max_llm_calls=args.max_llm_calls,
        wall_time_sec=args.wall_time_sec,
    )
    try:
        terminal_nodes, search_log = gen._run_beam(  # type: ignore[attr-defined]
            intent=str(base.get("intent", "")),
            state=dict(base.get("state") or {}),
            attack_intent=infer_attack_intent(seed),
            budget=budget,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return CellResult(source="beam_nodes", error=str(exc))

    adversarial = [n for n in terminal_nodes if getattr(n, "trajectory_unsafe", False)]
    oracle_confs = [float(n.oracle_confidence) for n in adversarial]
    unique_states: Set[str] = set()
    action_set: Set[str] = set()
    for n in search_log:
        ps = getattr(n, "predicted_state", None)
        axtree = (
            ps.get("a11y_tree_txt", "") if isinstance(ps, dict) else ""
        )
        unique_states.add(axtree)
        for a in getattr(n, "history", []) or []:
            action_set.add(str(a))
    return CellResult(
        source="beam_nodes",
        n_units_scored=len(search_log),
        n_adversarial=len(adversarial),
        n_payloads=len(adversarial),
        mean_oracle_confidence_adv=(
            sum(oracle_confs) / len(oracle_confs) if oracle_confs else 0.0
        ),
        mean_wm_risk_score_adv=0.0,  # beam nodes record step-level risks; adv here refers to whole trajectory
        predicted_state_diversity=len(unique_states),
        action_set=sorted(action_set),
    )


def _precursor_predicted_state(
    args: argparse.Namespace,
    defender: Any,
    policies: List[Dict[str, Any]],
    base: Dict[str, Any],
    seed: str,
    log_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Always uses rollout_mode=on (composition only makes sense then)."""
    proposer = _build_beam_proposer(args)
    gen = BeamGenerator(
        attacker=proposer,
        defender=defender,
        oracle=TrajectoryPolicyOracle(rule_oracle=RulePolicyOracle()),
        policies=policies,
        log_dir=log_dir,
        beam_width=args.beam_width,
        max_depth=args.beam_depth,
        branch_factor=args.branch_factor,
        default_budget=Budget(
            max_llm_calls=args.max_llm_calls,
            wall_time_sec=args.wall_time_sec,
        ),
        rollout_mode=True,
    )
    budget = Budget(
        max_llm_calls=args.max_llm_calls,
        wall_time_sec=args.wall_time_sec,
    )
    try:
        _terminals, search_log = gen._run_beam(  # type: ignore[attr-defined]
            intent=str(base.get("intent", "")),
            state=dict(base.get("state") or {}),
            attack_intent=infer_attack_intent(seed),
            budget=budget,
        )
    except Exception:
        return None
    if not search_log:
        return None
    deepest = max(search_log, key=lambda n: getattr(n, "depth", 0))
    ps = getattr(deepest, "predicted_state", None)
    if isinstance(ps, dict) and ps:
        return dict(ps)
    return None


def _run_rephrase_cell(
    args: argparse.Namespace,
    cfg: CellConfig,
    defender: Any,
    policies: List[Dict[str, Any]],
    base: Dict[str, Any],
    seed: str,
    log_dir: Path,
    precursor_defender: Any,
    precursor_log_dir: Path,
) -> CellResult:
    state_override: Optional[Dict[str, Any]] = None
    if cfg.use_precursor_override:
        state_override = _precursor_predicted_state(
            args, precursor_defender, policies, base, seed, precursor_log_dir
        )

    attacker = _build_rephrase_attacker(args)
    gen = RephraseGenerator(
        attacker=attacker,
        defender=defender,
        oracle=RulePolicyOracle(),
        policies=policies,
        log_dir=log_dir,
        default_budget=Budget(
            max_llm_calls=args.max_llm_calls,
            wall_time_sec=args.wall_time_sec,
        ),
    )
    request = GenerationRequest(
        base_task=base,
        seed_actions=[seed],
        n_variants=args.n_variants,
        max_payloads=args.max_payloads,
        state_override=state_override,
    )
    try:
        result = gen.generate(request)
    except Exception as exc:
        return CellResult(source="variant_scores", error=str(exc))

    # Read persisted GenerationRecord(s) to recover all scored variants
    # (adversarial and rejected both). ``result.payloads`` only has
    # adversarial-selected ones.
    variants: List[Dict[str, Any]] = []
    for f in sorted(log_dir.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        variants.extend(rec.get("variants") or [])

    adv = [v for v in variants if v.get("adversarial")]
    oracle_confs = [float(v.get("oracle_confidence", 0.0)) for v in adv]
    wm_risks = [float(v.get("wm_risk_score", 0.0)) for v in adv]
    action_set = sorted({str(v.get("action", "")) for v in variants if v.get("action")})
    return CellResult(
        source="variant_scores",
        n_units_scored=len(variants),
        n_adversarial=len(adv),
        n_payloads=len(result.payloads),
        mean_oracle_confidence_adv=(
            sum(oracle_confs) / len(oracle_confs) if oracle_confs else 0.0
        ),
        mean_wm_risk_score_adv=(
            sum(wm_risks) / len(wm_risks) if wm_risks else 0.0
        ),
        predicted_state_diversity=1,  # rephrase sees exactly one state per request
        action_set=action_set,
    )


def _run_cell(
    args: argparse.Namespace,
    cfg: CellConfig,
    defender: Any,
    precursor_defender: Any,
    policies: List[Dict[str, Any]],
    base: Dict[str, Any],
    seed: str,
) -> CellResult:
    with tempfile.TemporaryDirectory() as td_main, \
         tempfile.TemporaryDirectory() as td_precursor:
        main_log = Path(td_main)
        precursor_log = Path(td_precursor)
        if cfg.generator == "beam":
            return _run_beam_cell(
                args, cfg, defender, policies, base, seed, main_log
            )
        if cfg.generator == "rephrase":
            return _run_rephrase_cell(
                args, cfg, defender, policies, base, seed, main_log,
                precursor_defender, precursor_log,
            )
        raise ValueError(f"Unknown generator {cfg.generator!r}")


# ---------------------------------------------------------------------------
# Aggregation + pairwise overlap
# ---------------------------------------------------------------------------


def _aggregate_cell(results: List[CellResult]) -> Dict[str, Any]:
    ok = [r for r in results if r.error is None]
    if not ok:
        return {
            "n_runs": 0,
            "n_errors": len(results),
        }
    n = len(ok)
    units = sum(r.n_units_scored for r in ok)
    adv = sum(r.n_adversarial for r in ok)
    payloads = sum(r.n_payloads for r in ok)
    hit_rates = [r.adversarial_hit_rate for r in ok]
    oracle_confs = [r.mean_oracle_confidence_adv for r in ok]
    diversities = [r.predicted_state_diversity for r in ok]
    return {
        "n_runs": n,
        "n_errors": len(results) - n,
        "sum_n_units_scored": units,
        "sum_n_adversarial": adv,
        "sum_n_payloads": payloads,
        "mean_adversarial_hit_rate": sum(hit_rates) / n,
        "mean_oracle_confidence_adv": sum(oracle_confs) / n,
        "mean_predicted_state_diversity": sum(diversities) / n,
    }


def _pairwise_overlap(
    cell_results: Dict[str, List[CellResult]],
) -> List[Dict[str, Any]]:
    """One row per (cell_a, cell_b) unordered pair. Jaccard averaged over
    per-run pairs of action sets."""
    cells = sorted(cell_results.keys())
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            per_run_j: List[float] = []
            only_a_total = only_b_total = shared_total = 0
            ras = cell_results[a]
            rbs = cell_results[b]
            for ra, rb in zip(ras, rbs):
                if ra.error or rb.error:
                    continue
                sa = set(ra.action_set)
                sb = set(rb.action_set)
                union = sa | sb
                inter = sa & sb
                per_run_j.append(len(inter) / len(union) if union else 1.0)
                only_a_total += len(sa - sb)
                only_b_total += len(sb - sa)
                shared_total += len(inter)
            rows.append({
                "cell_a": a,
                "cell_b": b,
                "n_pairs": len(per_run_j),
                "mean_jaccard": sum(per_run_j) / len(per_run_j) if per_run_j else 1.0,
                "min_jaccard": min(per_run_j) if per_run_j else 1.0,
                "sum_only_a": only_a_total,
                "sum_only_b": only_b_total,
                "sum_shared": shared_total,
            })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    data_root = BENCHMARK_ROOT / "data"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-tasks", type=Path, default=data_root / "base_tasks.jsonl")
    p.add_argument("--attack-payloads-root", type=Path, default=data_root / "attack_payloads")
    p.add_argument("--output", type=Path, default=data_root / "results_ablation.json")
    p.add_argument("--max-base-tasks", type=int, default=10)
    p.add_argument("--max-seeds-per-task", type=int, default=1)

    # Cell selection
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Preset cell selection. If both --preset and --cells are set, --cells wins.",
    )
    p.add_argument(
        "--cells",
        nargs="+",
        choices=ALL_CELLS,
        default=None,
        help="Explicit list of cell_ids to run.",
    )

    # Beam (also used as precursor for rephrase_composed)
    p.add_argument("--beam-width", type=int, default=2)
    p.add_argument("--beam-depth", type=int, default=2)
    p.add_argument("--branch-factor", type=int, default=2)

    # Rephrase
    p.add_argument("--n-variants", type=int, default=4)
    p.add_argument("--max-payloads", type=int, default=3)
    p.add_argument("--max-llm-calls", type=int, default=64)
    p.add_argument("--wall-time-sec", type=float, default=180.0)

    # Attacker / defender
    p.add_argument("--attacker", choices=["template", "llm"], default="template")
    p.add_argument("--attacker-backend", default="deepseek", choices=tuple(_BACKEND_REGISTRY.keys()))
    p.add_argument("--attacker-model", default=None)
    p.add_argument("--attacker-temperature", type=float, default=0.9)
    p.add_argument("--attacker-seed", type=int, default=0)
    p.add_argument(
        "--defender",
        choices=["stub-lenient", "stub-strict", "stub-synthetic"],
        default="stub-synthetic",
        help="stub-synthetic (default) emits deterministic non-empty raw_delta "
             "so Phase-C/D folding is exercised without LLM cost.",
    )
    return p.parse_args()


def _resolve_cells(args: argparse.Namespace) -> List[str]:
    if args.cells:
        return list(args.cells)
    if args.preset:
        return list(PRESETS[args.preset])
    return list(ALL_CELLS)


def main() -> None:
    args = parse_args()
    cell_ids = _resolve_cells(args)
    cell_configs = [CellConfig.from_id(cid) for cid in cell_ids]

    policies = load_policies(POLICIES_PATH)
    base_tasks = _load_base_tasks(args.base_tasks)
    if args.max_base_tasks is not None:
        base_tasks = base_tasks[: args.max_base_tasks]

    # cell_results[cell_id] = list of CellResult, one per (base, seed) run.
    cell_results: Dict[str, List[CellResult]] = {c.cell_id: [] for c in cell_configs}
    per_task_rows: List[Dict[str, Any]] = []

    for base in base_tasks:
        seeds = _collect_seed_actions(base, args.attack_payloads_root)
        if not seeds:
            continue
        seeds = seeds[: args.max_seeds_per_task]
        for seed in seeds:
            # One defender per (base, seed) shared across cells for parity:
            # all cells score on the same synthetic-delta counter stream.
            # Precursor defender is separate so its call counter starts at 0.
            defender = _build_defender(args)
            precursor_defender = _build_defender(args)
            row: Dict[str, Any] = {
                "task_id": base.get("task_id"),
                "intent": base.get("intent"),
                "seed": seed,
                "cells": {},
            }
            for cfg in cell_configs:
                res = _run_cell(
                    args, cfg, defender, precursor_defender,
                    policies, base, seed,
                )
                cell_results[cfg.cell_id].append(res)
                row["cells"][cfg.cell_id] = {
                    "source": res.source,
                    "n_units_scored": res.n_units_scored,
                    "n_adversarial": res.n_adversarial,
                    "adversarial_hit_rate": res.adversarial_hit_rate,
                    "mean_oracle_confidence_adv": res.mean_oracle_confidence_adv,
                    "n_payloads": res.n_payloads,
                    "predicted_state_diversity": res.predicted_state_diversity,
                    "action_set_size": len(res.action_set),
                    "error": res.error,
                }
            per_task_rows.append(row)

    overlap_rows = _pairwise_overlap(cell_results)

    summary = {
        "config": {
            "attacker": args.attacker,
            "attacker_backend": args.attacker_backend if args.attacker == "llm" else None,
            "attacker_model": args.attacker_model,
            "defender": args.defender,
            "beam_width": args.beam_width,
            "beam_depth": args.beam_depth,
            "branch_factor": args.branch_factor,
            "n_variants": args.n_variants,
            "cells": cell_ids,
            "max_base_tasks": args.max_base_tasks,
            "max_seeds_per_task": args.max_seeds_per_task,
        },
        "n_seed_runs": len(per_task_rows),
        "cells": {
            cid: _aggregate_cell(cell_results[cid]) for cid in cell_ids
        },
        "pairwise_overlap": overlap_rows,
    }
    out = {"summary": summary, "per_task": per_task_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Human-readable print
    print(f"\nWrote {args.output}\n")
    print(f"Seed runs: {summary['n_seed_runs']} | attacker={args.attacker} | defender={args.defender}")
    print(f"\n{'cell_id':<22} {'src':<15} {'runs':<5} {'units':<6} {'adv':<5} "
          f"{'hit_rate':<9} {'oracle_adv':<10} {'div':<5} {'payloads':<9}")
    for cid in cell_ids:
        a = summary["cells"][cid]
        if a.get("n_runs", 0) == 0:
            print(f"{cid:<22} (no runs)")
            continue
        # pick source from first non-error result
        src = next(
            (r.source for r in cell_results[cid] if r.error is None),
            "?",
        )
        print(
            f"{cid:<22} {src:<15} {a['n_runs']:<5} {a['sum_n_units_scored']:<6} "
            f"{a['sum_n_adversarial']:<5} {a['mean_adversarial_hit_rate']:<9.3f} "
            f"{a['mean_oracle_confidence_adv']:<10.3f} "
            f"{a['mean_predicted_state_diversity']:<5.1f} "
            f"{a['sum_n_payloads']:<9}"
        )
    print("\nPairwise action-set Jaccard (off-diagonal):")
    print(f"  {'cell_a':<22} {'cell_b':<22} {'pairs':<6} {'mean_jacc':<10} {'min_jacc':<10}")
    for row in overlap_rows:
        print(
            f"  {row['cell_a']:<22} {row['cell_b']:<22} {row['n_pairs']:<6} "
            f"{row['mean_jaccard']:<10.3f} {row['min_jaccard']:<10.3f}"
        )


if __name__ == "__main__":
    main()
