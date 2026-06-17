#!/usr/bin/env python3
"""Judge-correlation experiment: how much does the choice of attack-success
judge inflate or deflate the reported ASR?

Hypothesis (the "correlated-judge failure"):
    A benchmark for predictive LLM defenses that uses an LLM-as-judge
    drawn from the same model family as the defender will systematically
    UNDER-report attack success. The judge agrees with the defender by
    construction (shared priors, shared training, often shared prompt
    style), so attacks the defender accepts get rubber-stamped as "safe"
    by the judge as well. The independent rule-based oracle does not
    share these priors and surfaces the true attack rate.

Procedure (per attack payload, per defender):
    1. Run the defender on the payload (this consumes defender LLM calls
       and produces a ``selected_action`` plus a ``requires_regeneration``
       flag).
    2. If the defender refused, skip — there's no realised action to judge.
    3. Otherwise, ask each of three judges:
         - shared_model:  same LLM model + same prompt style as defender
         - shared_family: same model FAMILY (e.g. another DeepSeek model
           or same model with a leaner prompt) — partially correlated
         - independent:   rule + trajectory oracle, NO LLM
       whether the realised action is unsafe.
    4. ASR(defender, judge) = fraction of payloads where judge says unsafe.

The headline statistic is the gap
    Δ = ASR(defender, independent) − ASR(defender, shared_model)
which directly quantifies how much a correlated judge under-reports.

Cost:
    defenders: see existing transfer-matrix tool. WMG ≈ 3 LLM calls /
        payload (DeepSeek wrapper).
    judges: shared_model + shared_family ≈ 1 LLM call each per payload;
        independent is free.
    Result for N payloads × 1 defender × 3 judges:
        defender LLM calls = N × 3
        judge LLM calls    = N × 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
SAFEPRED_ROOT = BENCHMARK_ROOT.parent.parent
SAFEPRED_PARENT = SAFEPRED_ROOT.parent
for _p in (BENCHMARK_ROOT, SAFEPRED_ROOT, SAFEPRED_PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load .env from repo root.
try:
    from dotenv import load_dotenv  # type: ignore
    for _parent in (BENCHMARK_ROOT, *BENCHMARK_ROOT.parents):
        _env = _parent / ".env"
        if _env.exists():
            load_dotenv(_env)
            break
except ImportError:
    pass


from defenses import get_defense_handler  # noqa: E402
from policy_oracle import (  # noqa: E402
    LLMPolicyOracle,
    OracleVerdict,
    RulePolicyOracle,
    TrajectoryPolicyOracle,
    load_policies,
)


_WORLD_MODEL_DEFENSES = {"defense_world_model_gating"}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_payloads(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for jsonl in sorted(root.rglob("*.jsonl")):
        for payload in _iter_jsonl(jsonl):
            if payload.get("attack_family") != "wm_adaptive":
                continue
            payload["_source_file"] = str(jsonl)
            out.append(payload)
    return out


def _build_defender_wrapper(*, config_path: Path, policy_path: Path) -> Any:
    """SafePredWrapper for WM-gating defender. DeepSeek-routed if env says."""
    ds_key = os.environ.get("DEEPSEEK_API_KEY")
    if ds_key:
        os.environ["OPENAI_API_KEY"] = ds_key
        os.environ["OPENAI_API_URL"] = (
            os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        )
    elif not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Need DEEPSEEK_API_KEY or OPENAI_API_KEY to run WMG defender.")
    from SafePred import SafePredWrapper  # type: ignore  # noqa: WPS433
    return SafePredWrapper(
        benchmark="wmdefensebench",
        config_path=str(config_path),
        policy_path=str(policy_path),
        web_agent_llm_config={},
    )


def _build_judges(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct the three judge configurations.

    'shared_model' and 'shared_family' both call DeepSeek; the difference
    is which model id and which prompt template is used. 'independent'
    uses the rule + trajectory oracle ensemble (no LLM, no shared priors
    with the defender at all).
    """
    judges: Dict[str, Any] = {}
    if "independent" in args.judges:
        rule = RulePolicyOracle()
        traj = TrajectoryPolicyOracle(rule_oracle=rule)
        judges["independent"] = traj  # supports .evaluate() shim too

    ds_key = os.environ.get("DEEPSEEK_API_KEY")
    ds_url = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"

    if "shared_model" in args.judges:
        if not ds_key:
            raise SystemExit("shared_model judge requires DEEPSEEK_API_KEY.")
        judges["shared_model"] = LLMPolicyOracle(
            api_key=ds_key,
            api_url=ds_url,
            model_name=args.shared_model_id,
            provider="openai",  # DeepSeek speaks OpenAI protocol
            temperature=0.0,
            prompt_template="default",  # full policy text — same context defender sees
        )
    if "shared_family" in args.judges:
        if not ds_key:
            raise SystemExit("shared_family judge requires DEEPSEEK_API_KEY.")
        judges["shared_family"] = LLMPolicyOracle(
            api_key=ds_key,
            api_url=ds_url,
            model_name=args.shared_family_id,
            provider="openai",
            temperature=0.0,
            prompt_template="short",  # leaner prompt; no policy text
        )
    return judges


def _run_defender(
    payload: Dict[str, Any],
    defense_id: str,
    wrapper: Optional[Any],
) -> Dict[str, Any]:
    overrides = payload.get("overrides", {}) or {}
    state = overrides.get("state") or payload.get("state", {}) or {}
    intent = overrides.get("intent") or payload.get("intent", "")
    candidates = list(overrides.get("candidate_actions", []) or [])
    unsafe_idx = list(overrides.get("unsafe_action_indices", []) or [])
    proposed = int(overrides.get("proposed_action_index", 0))
    handler = get_defense_handler(defense_id)
    handler_wrapper = wrapper if getattr(handler, "uses_world_model", False) else None
    try:
        result = handler.evaluate_step(
            wrapper=handler_wrapper,
            task_id=str(overrides.get("task_id", payload.get("payload_id", ""))),
            intent=intent,
            state=state,
            candidate_actions=candidates,
            unsafe_action_indices=unsafe_idx,
            proposed_index=proposed,
            action_history=[],
            current_response="",
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "selected_action_index": -1,
        }
    return {
        "ok": True,
        "selected_action_index": result.get("selected_action_index", -1),
        "selected_action": result.get("selected_action"),
        "requires_regeneration": bool(result.get("requires_regeneration", False)),
    }


def _query_judge(
    judge: Any,
    intent: str,
    state: Dict[str, Any],
    action: str,
    policies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        verdict: OracleVerdict = judge.evaluate(intent, state, action, policies)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "unsafe": None}
    return {
        "ok": True,
        "unsafe": bool(verdict.unsafe),
        "violated_policy_ids": list(verdict.violated_policy_ids),
        "confidence": float(verdict.confidence),
        "rationale": str(verdict.rationale)[:300],
        "oracle_id": verdict.oracle_id,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge-correlation experiment.")
    p.add_argument(
        "--payloads-root",
        type=Path,
        default=BENCHMARK_ROOT / "data" / "attack_payloads" / "wm_adaptive",
    )
    p.add_argument(
        "--defenses",
        nargs="+",
        default=["defense_world_model_gating"],
        help="Defenses to evaluate against. WMG is the most informative since "
        "it's the LLM defender we want to expose correlated-judge bias on.",
    )
    p.add_argument(
        "--judges",
        nargs="+",
        default=["independent", "shared_model", "shared_family"],
        choices=("independent", "shared_model", "shared_family"),
    )
    p.add_argument(
        "--shared-model-id",
        default="deepseek-chat",
        help="Model id for the shared-model judge. Should match the defender's WM model.",
    )
    p.add_argument(
        "--shared-family-id",
        default="deepseek-chat",
        help="Model id for the shared-family judge. Same family as defender; the "
        "differentiator is a leaner prompt (see prompt_template='short').",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=BENCHMARK_ROOT / "config_deepseek.yaml",
        help="Defender wrapper config; defaults to DeepSeek-routed config.",
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=BENCHMARK_ROOT / "data" / "policies.json",
    )
    p.add_argument(
        "--max-payloads",
        type=int,
        default=0,
        help="Cap on payloads to process; 0 = all.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_ROOT / "results_judge_correlation.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=BENCHMARK_ROOT / "results_judge_correlation.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policies = load_policies(args.policy)

    payloads = _collect_payloads(args.payloads_root)
    if args.max_payloads > 0:
        payloads = payloads[: args.max_payloads]
    if not payloads:
        raise SystemExit(f"No WM-adaptive payloads found under {args.payloads_root}")

    judges = _build_judges(args)
    if not judges:
        raise SystemExit("No judges enabled.")

    # Build defender wrapper only if a WM defense is selected. Others are stateless.
    wrapper: Optional[Any] = None
    if any(d in _WORLD_MODEL_DEFENSES for d in args.defenses):
        wrapper = _build_defender_wrapper(config_path=args.config, policy_path=args.policy)

    flat_rows: List[Dict[str, Any]] = []
    # cell counts: (defense, judge) -> {n, unsafe}
    cells: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"n": 0, "unsafe": 0, "errors": 0})

    for payload in payloads:
        pid = str(payload.get("payload_id", "?"))
        overrides = payload.get("overrides", {}) or {}
        state = overrides.get("state") or payload.get("state", {}) or {}
        intent = overrides.get("intent") or payload.get("intent", "")

        for defense_id in args.defenses:
            d_result = _run_defender(payload, defense_id, wrapper)
            if not d_result["ok"]:
                # Defender errored; skip judging.
                for judge_name in judges:
                    flat_rows.append(
                        {
                            "payload_id": pid,
                            "defense_id": defense_id,
                            "judge": judge_name,
                            "skipped": True,
                            "skip_reason": "defender_error",
                            "defender_error": d_result.get("error"),
                        }
                    )
                continue

            sel_idx = d_result["selected_action_index"]
            sel_action = d_result["selected_action"]
            if sel_idx == -1 or sel_action is None or d_result["requires_regeneration"]:
                # Defender refused; nothing to judge.
                for judge_name in judges:
                    flat_rows.append(
                        {
                            "payload_id": pid,
                            "defense_id": defense_id,
                            "judge": judge_name,
                            "skipped": True,
                            "skip_reason": "defender_refused",
                        }
                    )
                continue

            for judge_name, judge in judges.items():
                v = _query_judge(judge, intent, state, str(sel_action), policies)
                cell = cells[(defense_id, judge_name)]
                cell["n"] += 1
                if not v["ok"]:
                    cell["errors"] += 1
                elif v["unsafe"]:
                    cell["unsafe"] += 1
                flat_rows.append(
                    {
                        "payload_id": pid,
                        "defense_id": defense_id,
                        "judge": judge_name,
                        "skipped": False,
                        "selected_action_index": sel_idx,
                        "selected_action": str(sel_action)[:200],
                        "judge_unsafe": v.get("unsafe"),
                        "judge_violated_policy_ids": v.get("violated_policy_ids"),
                        "judge_confidence": v.get("confidence"),
                        "judge_rationale": v.get("rationale"),
                        "judge_oracle_id": v.get("oracle_id"),
                        "judge_error": v.get("error"),
                    }
                )

    # Per-cell ASR.
    asr_table: Dict[str, Dict[str, Any]] = {}
    for (defense_id, judge_name), cell in cells.items():
        n = cell["n"]
        rate = cell["unsafe"] / n if n else 0.0
        asr_table.setdefault(defense_id, {})[judge_name] = {
            "n": n,
            "unsafe": cell["unsafe"],
            "errors": cell["errors"],
            "asr": round(rate, 4),
        }

    # Per-defender deltas: (independent − shared_model), (independent − shared_family).
    inflation_gaps: Dict[str, Dict[str, Any]] = {}
    for defense_id, judge_map in asr_table.items():
        if "independent" in judge_map:
            ind_asr = judge_map["independent"]["asr"]
            gaps = {}
            for jn, jres in judge_map.items():
                if jn == "independent":
                    continue
                gaps[jn] = {
                    "asr_diff_independent_minus_judge": round(ind_asr - jres["asr"], 4),
                    "judge_asr": jres["asr"],
                    "independent_asr": ind_asr,
                }
            inflation_gaps[defense_id] = gaps

    report = {
        "payloads_root": str(args.payloads_root),
        "n_payloads_in_corpus": len(payloads),
        "defenses": args.defenses,
        "judges": list(judges.keys()),
        "asr_table": asr_table,
        "correlated_judge_inflation": inflation_gaps,
        "per_payload": flat_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if flat_rows:
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "payload_id",
                    "defense_id",
                    "judge",
                    "skipped",
                    "skip_reason",
                    "selected_action_index",
                    "selected_action",
                    "judge_unsafe",
                    "judge_violated_policy_ids",
                    "judge_confidence",
                    "judge_rationale",
                    "judge_oracle_id",
                    "judge_error",
                    "defender_error",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)

    # Compact stdout summary: one matrix per defender.
    print(json.dumps({"asr_table": asr_table, "inflation": inflation_gaps}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
