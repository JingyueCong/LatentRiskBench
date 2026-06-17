#!/usr/bin/env python3
"""Command-line driver for the WM-adaptive attack generator.

Pilot usage (Phase 1; TemplateAttacker + StubDefenderWMScorer(lenient) +
RulePolicyOracle, no LLM credentials required):

    python -m attack_generator.cli \
        --base-tasks data/base_tasks.jsonl \
        --attack-payloads-root data/attack_payloads \
        --output-dir data/attack_payloads/wm_adaptive \
        --log-dir data/attack_generation_logs \
        --pilot 5 --threat-mode white_box --algorithm rephrase

Real-LLM usage (requires OPENAI_API_KEY for the attacker/defender and the
configured ``POLICY_ORACLE_API_KEY`` for the LLM oracle, if enabled):

    python -m attack_generator.cli --use-llm-attacker --use-llm-defender ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
# SafePred package root (three levels up). Its PARENT must be on sys.path
# so ``from latentriskbench.utils.llm_client import LLMClient`` works; some of
# SafePred's utilities use ``from ..utils.X`` (two-level relative) and
# therefore need ``SafePred`` to be importable as a package.
SAFEPRED_ROOT = Path(__file__).resolve().parents[3]
SAFEPRED_PARENT = SAFEPRED_ROOT.parent
for _p in (BENCHMARK_ROOT, SAFEPRED_ROOT, SAFEPRED_PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_env_from_walk_up() -> Optional[Path]:
    """Look for a ``.env`` file at BENCHMARK_ROOT and each parent up to the
    filesystem root; load the first one found. Returns the loaded path for
    logging. No-op if ``python-dotenv`` is not installed or no file found."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return None
    for parent in (BENCHMARK_ROOT, *BENCHMARK_ROOT.parents):
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            return candidate
    return None


_LOADED_ENV_PATH = _load_env_from_walk_up()

from attack_generator import (  # noqa: E402
    AttackerAxes,
    BUDGET_OPTIONS,
    BeamGenerator,
    Budget,
    CoarseningDefenderScorer,
    GenerationRequest,
    LLMActionProposer,
    LLMAttacker,
    LLMDefenderWMScorer,
    POLICY_OPTIONS,
    PRE_FILTER_OPTIONS,
    RISK_SCORE_OPTIONS,
    ROLLOUT_OPTIONS,
    RephraseGenerator,
    StubDefenderWMScorer,
    TemplateActionProposer,
    TemplateAttacker,
    WrapperDefenderWMScorer,
    build_attacker,
    valid_threat_modes,
)
from policy_oracle import (  # noqa: E402
    RulePolicyOracle,
    TrajectoryPolicyOracle,
    build_oracle_from_config,
    load_policies,
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_base_tasks(path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(path))


def _collect_seed_actions(
    base_task: Dict[str, Any], payloads_root: Path
) -> List[str]:
    """For a benign base_task, scan existing L1-L3 payloads that target it
    and extract their unsafe candidate actions to use as seeds."""
    base_id = str(base_task.get("task_id", ""))
    seeds: List[str] = []
    if not payloads_root.exists():
        return seeds
    for payload_file in sorted(payloads_root.rglob("*.jsonl")):
        if payload_file.parent.name == "wm_adaptive":
            # Do not seed the WM-adaptive generator from its own output.
            continue
        for payload in _iter_jsonl(payload_file):
            target_ids = payload.get("target_task_ids") or []
            if base_id not in (str(t) for t in target_ids):
                continue
            overrides = payload.get("overrides") or {}
            candidates = overrides.get("candidate_actions") or []
            unsafe_idx = overrides.get("unsafe_action_indices") or []
            for idx in unsafe_idx:
                try:
                    action = str(candidates[int(idx)])
                except (IndexError, ValueError, TypeError):
                    continue
                if action and action not in seeds:
                    seeds.append(action)
    return seeds


# Backend registry: logical name -> (api_key_env, default_api_url, default_model, sdk_provider).
# ``sdk_provider`` is what LLMClient sees; DeepSeek speaks OpenAI protocol so
# we pass ``openai`` and override ``base_url``. This keeps one code path.
_BACKEND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "default_api_url_env": "OPENAI_BASE_URL",
        "default_api_url": None,  # OpenAI SDK defaults to api.openai.com/v1
        "default_model": "gpt-4o-mini",
        "sdk_provider": "openai",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_api_url_env": "DEEPSEEK_BASE_URL",
        "default_api_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "sdk_provider": "openai",  # DeepSeek is OpenAI-compatible
    },
    "openrouter": {
        # OpenRouter is a unified gateway to many models (OpenAI, Anthropic,
        # DeepSeek, Llama, etc.) via an OpenAI-compatible API. Useful for
        # cross-model transfer studies without managing per-provider keys.
        # Model names follow OpenRouter's ``<org>/<model>`` convention,
        # e.g. ``openai/gpt-4o-mini``, ``deepseek/deepseek-chat``,
        # ``anthropic/claude-haiku-4-5``. See https://openrouter.ai/models.
        "api_key_env": "OPEN_ROUTER_API_KEY",
        "default_api_url_env": "OPEN_ROUTER_BASE_URL",
        "default_api_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "sdk_provider": "openai",
    },
}


def _resolve_backend(name: str) -> Dict[str, Any]:
    """Look up (api_key, api_url, default_model, sdk_provider) for a named backend.

    Raises SystemExit with a helpful message if the backend's API key env
    var is not set, so that pilot runs fail loudly at startup rather than
    mid-budget.
    """
    import os as _os
    key = name.strip().lower()
    if key not in _BACKEND_REGISTRY:
        raise SystemExit(
            f"Unknown backend {name!r}. Registered: {sorted(_BACKEND_REGISTRY)}"
        )
    cfg = _BACKEND_REGISTRY[key]
    api_key = _os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise SystemExit(
            f"Backend {name!r} requires {cfg['api_key_env']} to be set (via env or .env)."
        )
    api_url = _os.environ.get(cfg["default_api_url_env"]) or cfg["default_api_url"]
    return {
        "api_key": api_key,
        "api_url": api_url,
        "default_model": cfg["default_model"],
        "sdk_provider": cfg["sdk_provider"],
        "name": key,
    }


def _build_wrapper_for_defender(
    args: argparse.Namespace,
    defender_backend: Dict[str, Any],
) -> Any:
    """Spin up a SafePredWrapper for use by WrapperDefenderWMScorer.

    Mirrors the env-var trick in tools/evaluate_transfer_matrix.py: when
    the chosen backend is DeepSeek (or any OpenAI-compatible third party),
    we keep ``provider: openai`` in the yaml config (so the OpenAI SDK
    code path is used) and overwrite ``OPENAI_API_KEY`` /
    ``OPENAI_API_URL`` in-process so the SDK is directed at the right
    endpoint.

    Raises SystemExit if dependencies aren't available so failure is
    obvious at startup rather than mid-generation.
    """
    import os as _os
    backend_name = defender_backend["name"]
    if backend_name == "deepseek":
        # SafetyConfig.from_yaml reads OPENAI_API_KEY / OPENAI_API_URL by
        # provider name; remap them to DeepSeek values for this process.
        _os.environ["OPENAI_API_KEY"] = defender_backend["api_key"]
        if defender_backend.get("api_url"):
            _os.environ["OPENAI_API_URL"] = defender_backend["api_url"]
    elif backend_name == "openai":
        if not defender_backend.get("api_key"):
            raise SystemExit(
                "--defender-scorer-mode wrapper with --defender-backend openai "
                "requires OPENAI_API_KEY."
            )
    config_path = args.defender_config or (
        BENCHMARK_ROOT / (
            "config_deepseek.yaml" if backend_name == "deepseek" else "config.yaml"
        )
    )
    policy_path = args.policies
    try:
        from latentriskbench import SafePredWrapper  # type: ignore  # noqa: WPS433
    except Exception as exc:
        raise SystemExit(f"Failed to import SafePredWrapper: {exc}") from exc
    return SafePredWrapper(
        benchmark="wmdefensebench",
        config_path=str(config_path),
        policy_path=str(policy_path),
        web_agent_llm_config={},
    )


def _load_protocol(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    # Avoid adding a yaml dep: do a minimal parse of the one block we need.
    # The file is project-controlled and the only block we read is
    # ``policy_oracle``. Fallback to empty dict if PyYAML is unavailable.
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_axes(
    args: argparse.Namespace,
    threat_mode: str,
) -> AttackerAxes:
    """Resolve the effective 5-axis config for this run.

    Starts from the preset attached to ``threat_mode`` (conventional
    white/grey/black_box), then overrides any axis whose CLI flag was
    set. If any override triggers, ``AttackerAxes.threat_mode`` flips
    to ``"custom"`` so logs don't misleadingly label the run with a
    canonical tier name.
    """
    base = AttackerAxes.from_threat_mode(threat_mode)
    return base.override(
        rollout=args.axis_rollout,
        policy=args.axis_policy,
        budget=args.axis_budget,
        risk_score=args.axis_risk_score,
        pre_filter=args.axis_pre_filter,
    )


def _build_generator(
    args: argparse.Namespace,
    policies: List[Dict[str, Any]],
    threat_mode: str,
) -> Any:
    # Phase P2: resolve the full 5-axis config once and use it both for
    # attacker construction (via axes.policy → prompt template) and
    # defender scoring (via axes.rollout → scorer mode; axes.budget →
    # wrapper n_root). The resolved AttackerAxes is also stashed on the
    # generator so each GenerationRecord can record its provenance.
    axes = _resolve_axes(args, threat_mode)

    # Back-compat: if the user set --defender-scorer-mode explicitly, it
    # wins over the axis-derived default. Otherwise, axes.rollout maps to
    # the scorer mode: wrapper→wrapper, simulate→simulate, none→stub.
    if args.defender_scorer_mode == "simulate" and axes.rollout == "wrapper":
        effective_scorer_mode = "wrapper"
    elif args.defender_scorer_mode == "wrapper" and axes.rollout == "simulate":
        effective_scorer_mode = "simulate"
    else:
        effective_scorer_mode = args.defender_scorer_mode

    # Resolve attacker + defender backends (credentials, base URL, default
    # model, SDK provider). ``_resolve_backend`` raises SystemExit when the
    # required env var is missing so failures surface at startup.
    attacker_backend = (
        _resolve_backend(args.attacker_backend) if args.use_llm_attacker else None
    )
    defender_backend = (
        _resolve_backend(args.defender_backend) if args.use_llm_defender else None
    )

    defender_model = (
        args.defender_model
        or (defender_backend["default_model"] if defender_backend else "gpt-4o-mini")
    )
    defender_provider = (
        args.defender_provider
        or (defender_backend["sdk_provider"] if defender_backend else "openai")
    )
    attacker_model = (
        args.attacker_model
        or (attacker_backend["default_model"] if attacker_backend else "gpt-4o-mini")
    )
    attacker_provider = (
        args.attacker_provider
        or (attacker_backend["sdk_provider"] if attacker_backend else "openai")
    )

    defender_config: Dict[str, Any] = {
        "model_name": defender_model,
        "provider": defender_provider,
    }
    attacker_overrides: Dict[str, Any] = {
        "model_name": attacker_model,
        "provider": attacker_provider,
        "temperature": args.attacker_temperature,
        "seed": args.attacker_seed,
        "max_variants": max(args.n_variants, 6),
    }
    if attacker_backend:
        attacker_overrides["api_key"] = attacker_backend["api_key"]
        attacker_overrides["api_url"] = attacker_backend["api_url"]

    attacker = build_attacker(
        threat_mode=threat_mode,
        use_llm=args.use_llm_attacker,
        defender_config=defender_config,
        attacker_overrides=attacker_overrides,
        axes=axes,
    )

    if args.use_llm_defender:
        assert defender_backend is not None
        if effective_scorer_mode == "wrapper":
            # Phase 4 path: scorer routes through SafePredWrapper.evaluate_action_risk
            # so attacker selection optimises against the same path the
            # WM-gating defense uses at eval time.
            # Phase P2: axes.budget="known" tells the wrapper scorer to
            # probe with the same tree_search width the defender will use
            # at eval time (len(candidate_actions) ≈ 3). "unknown" stays
            # at n_root=1 (single-action probe, cheapest).
            tsr = 3 if getattr(axes, "budget", "unknown") == "known" else 1
            wrapper_obj = _build_wrapper_for_defender(
                args=args,
                defender_backend=defender_backend,
            )
            defender = WrapperDefenderWMScorer(
                wrapper_obj,
                config={
                    "model_name": defender_model,
                    "provider": defender_provider,
                    "scorer_mode": "wrapper",
                    "axis_budget": axes.budget,
                },
                tree_search_n_root=tsr,
            )
        else:
            # Phase 1-3 path: simulate() direct call.
            try:
                from latentriskbench.models.world_model import LLMBasedWorldModel  # type: ignore
            except Exception:
                from models.world_model import LLMBasedWorldModel  # type: ignore
            llm_config: Dict[str, Any] = {
                "model_name": defender_model,
                "provider": defender_provider,
                "api_key": defender_backend["api_key"],
                "api_url": defender_backend["api_url"],
                "temperature": 0.2,
                "max_tokens": 2000,
                "timeout": 90,
            }
            wm = LLMBasedWorldModel(llm_config=llm_config)
            defender = LLMDefenderWMScorer(wm)
    else:
        # Phase P2: axes.rollout="none" explicitly requests no defender
        # scoring; we map that to the stub (any scorer-based selection
        # filter is then effectively a pure oracle filter).
        defender = StubDefenderWMScorer(mode=args.stub_defender_mode)

    # Phase P3: A2 (risk_score granularity) + A4 (pre_filter visibility)
    # are applied uniformly as a post-hoc view on whatever defender the
    # above block produced. We only allocate a coarsening wrapper when it
    # would actually change behaviour; the scalar+visible default is
    # equivalent to a passthrough and would needlessly lengthen
    # defender_id in the record.
    if axes.risk_score != "scalar" or axes.pre_filter != "visible":
        defender = CoarseningDefenderScorer(
            defender,
            risk_score=axes.risk_score,
            pre_filter=axes.pre_filter,
        )

    protocol = _load_protocol(args.protocol)
    oracle_cfg = protocol.get("policy_oracle", {}) if isinstance(protocol, dict) else {}
    if args.rule_oracle_only or not oracle_cfg:
        oracle: Any = RulePolicyOracle()
    else:
        # Forbid the oracle from using the defender's own model.
        forbidden = [defender_model] if args.use_llm_defender else []
        oracle = build_oracle_from_config(oracle_cfg, forbidden_model_names=forbidden)

    budget = Budget(
        max_llm_calls=args.max_llm_calls,
        wall_time_sec=args.wall_time_sec,
    )

    if args.algorithm == "beam":
        # Beam search uses an ActionProposer, not the AttackerWM variant
        # interface. The RephraseGenerator's attacker is unused here.
        proposer: Any
        if args.use_llm_attacker:
            assert attacker_backend is not None
            proposer = LLMActionProposer(
                api_key=attacker_backend["api_key"],
                api_url=attacker_backend["api_url"],
                model_name=attacker_model,
                provider=attacker_provider,
                temperature=args.attacker_temperature,
                seed=args.attacker_seed,
            )
        else:
            proposer = TemplateActionProposer()
        trajectory_oracle = TrajectoryPolicyOracle(rule_oracle=RulePolicyOracle())
        return BeamGenerator(
            attacker=proposer,
            defender=defender,
            oracle=trajectory_oracle,
            policies=policies,
            log_dir=args.log_dir,
            beam_width=args.beam_width,
            max_depth=args.beam_depth,
            branch_factor=args.beam_branch_factor,
            default_budget=budget,
            attacker_axes=axes,
            rollout_mode=(args.beam_rollout_mode == "on"),
        )

    return RephraseGenerator(
        attacker=attacker,
        defender=defender,
        oracle=oracle,
        policies=policies,
        log_dir=args.log_dir,
        default_budget=budget,
        attacker_axes=axes,
    )


def _select_base_tasks(
    all_base: List[Dict[str, Any]], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    if args.base_task_id:
        want = set(args.base_task_id)
        selected = [t for t in all_base if str(t.get("task_id")) in want]
        missing = want - {str(t.get("task_id")) for t in selected}
        if missing:
            raise SystemExit(f"Unknown base_task_id(s): {sorted(missing)}")
        return selected
    if args.pilot and args.pilot > 0:
        return all_base[: args.pilot]
    return all_base


def _write_payloads(output_path: Path, payloads: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    data_root = BENCHMARK_ROOT / "data"
    parser = argparse.ArgumentParser(description="WM-adaptive attack generator (Phase 1).")
    parser.add_argument("--base-tasks", type=Path, default=data_root / "base_tasks.jsonl")
    parser.add_argument(
        "--attack-payloads-root",
        type=Path,
        default=data_root / "attack_payloads",
        help="Root of existing attack payloads; used to extract seed unsafe actions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "attack_payloads" / "wm_adaptive",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=data_root / "attack_generation_logs",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=data_root / "policies.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=BENCHMARK_ROOT / "eval_protocol.yaml",
    )
    parser.add_argument("--base-task-id", action="append", default=[], help="Restrict to listed base task ids (repeatable).")
    parser.add_argument("--pilot", type=int, default=0, help="Process only the first N base tasks.")
    parser.add_argument(
        "--threat-mode",
        default="white_box",
        choices=("white_box", "grey_box", "black_box"),
        help="Threat mode to generate. Ignored when --all-threat-modes is set.",
    )
    parser.add_argument(
        "--all-threat-modes",
        action="store_true",
        help="Generate under all three threat modes. Output files are suffixed with __<threat_mode>.jsonl.",
    )
    # 5-axis threat-model flags (Phase P2; see docs/wm_0417.md §C.6).
    # When any of these is set, it overrides the corresponding slot in the
    # preset selected by --threat-mode. If you override every axis the
    # resulting config is labelled "custom" in generation_metadata.
    parser.add_argument(
        "--axis-rollout",
        choices=ROLLOUT_OPTIONS,
        default=None,
        help="A1: attacker's rollout access. 'wrapper' routes scoring "
        "through SafePredWrapper.evaluate_action_risk (Phase 4 behaviour); "
        "'simulate' uses LLMBasedWorldModel.simulate (Phase 1-3 behaviour); "
        "'none' disables defender-scoring entirely (Template-only).",
    )
    parser.add_argument(
        "--axis-policy",
        choices=POLICY_OPTIONS,
        default=None,
        help="A3: attacker's policy-text access. 'full' includes complete "
        "policy spec in the attacker LLM prompt; 'summary' uses a stripped-"
        "down prompt; 'none' gives the attacker no policy information.",
    )
    parser.add_argument(
        "--axis-budget",
        choices=BUDGET_OPTIONS,
        default=None,
        help="A5: attacker's search-budget knowledge. 'known' probes with "
        "the defender's actual tree_search_n_root = N candidates (matched "
        "to eval-time); 'unknown' uses n_root=1.",
    )
    parser.add_argument(
        "--axis-risk-score",
        choices=RISK_SCORE_OPTIONS,
        default=None,
        help="A2: attacker's view of the defender's verdict. 'scalar' "
        "preserves the scalar risk and full violated_policy_ids list; "
        "'binary' coarsens to {safe, unsafe} with an opaque single-label "
        "violation so the attacker can't rank borderline variants.",
    )
    parser.add_argument(
        "--axis-pre-filter",
        choices=PRE_FILTER_OPTIONS,
        default=None,
        help="A4: attacker's pre-filter visibility. 'visible' surfaces "
        "whether the wrapper's deterministic secret-pattern guard blocked "
        "a candidate (distinct from LLM-based rejection); 'hidden' "
        "suppresses this signal so the attacker can't route around known "
        "regex patterns.",
    )
    parser.add_argument(
        "--algorithm",
        default="rephrase",
        choices=("rephrase", "beam"),
        help="Attack generator algorithm. 'rephrase' = Phase 1 single-action search; "
        "'beam' = Phase 3 multi-step beam search over trajectories.",
    )
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--beam-depth", type=int, default=3)
    parser.add_argument("--beam-branch-factor", type=int, default=4)
    parser.add_argument(
        "--beam-rollout-mode",
        choices=["off", "on"],
        default="off",
        help=(
            "Beam-only. 'on' = attacker + defender see the WM-predicted "
            "next state (forward-simulator rollout); 'off' (default) = "
            "both always see the initial state (score-only, pre-Phase-C "
            "behaviour). Recorded on each payload under "
            "attacker_config.rollout_mode."
        ),
    )
    parser.add_argument(
        "--rephrase-state-override",
        choices=["none", "beam"],
        default="none",
        help=(
            "Phase D. Rephrase-only. 'none' (default) = rephrase runs at "
            "the base task's initial state. 'beam' = run a precursor beam "
            "(reusing --beam-width/--beam-depth/--beam-branch-factor and "
            "forcing rollout_mode=on so predicted_state is folded) to "
            "produce an intermediate WM-predicted state, then feed that "
            "state as state_override into the rephrase. Composition: "
            "rephrase-at-depth-k. Recorded on each payload under "
            "attacker_config.state_override_used/state_signature."
        ),
    )
    parser.add_argument("--n-variants", type=int, default=6)
    parser.add_argument("--max-payloads", type=int, default=3)
    parser.add_argument("--max-llm-calls", type=int, default=8)
    parser.add_argument("--wall-time-sec", type=float, default=60.0)
    parser.add_argument("--attacker-seed", type=int, default=0)
    parser.add_argument("--attacker-temperature", type=float, default=0.9)
    parser.add_argument("--use-llm-attacker", action="store_true")
    parser.add_argument(
        "--attacker-backend",
        default="openai",
        choices=tuple(_BACKEND_REGISTRY.keys()),
        help="LLM backend for the attacker. Controls which API key env var is "
        "read and which base URL is used. ``deepseek`` uses DEEPSEEK_API_KEY "
        "and api.deepseek.com; ``openai`` uses OPENAI_API_KEY.",
    )
    parser.add_argument("--attacker-model", default=None, help="Model name; defaults to the backend's default.")
    parser.add_argument("--attacker-provider", default=None, help="SDK provider override (usually auto from --attacker-backend).")
    parser.add_argument("--use-llm-defender", action="store_true")
    parser.add_argument(
        "--defender-backend",
        default="openai",
        choices=tuple(_BACKEND_REGISTRY.keys()),
        help="LLM backend for the defender WM. Same semantics as --attacker-backend.",
    )
    parser.add_argument("--defender-model", default=None)
    parser.add_argument("--defender-provider", default=None)
    parser.add_argument(
        "--defender-scorer-mode",
        default="simulate",
        choices=("simulate", "wrapper"),
        help="Phase 4: defender scoring path used by the attack generator. "
        "'simulate' calls LLMBasedWorldModel.simulate() directly (Phase 1-3 "
        "default; cheap; produces attacks that may not transfer to "
        "defense_world_model_gating because the eval-time defender uses a "
        "different prompt). 'wrapper' calls SafePredWrapper.evaluate_action_risk() "
        "-- the EXACT eval-time path -- so attacks selected here can actually "
        "fool the WM-gating defense by construction. 'wrapper' implies a live "
        "SafePredWrapper and is more expensive (one wrapper call per variant).",
    )
    parser.add_argument(
        "--defender-config",
        type=Path,
        default=None,
        help="SafePred config yaml for the wrapper-mode defender scorer. "
        "Defaults to config.yaml (openai backend) or config_deepseek.yaml "
        "(deepseek backend).",
    )
    parser.add_argument("--stub-defender-mode", default="lenient", choices=("lenient", "strict"))
    parser.add_argument("--rule-oracle-only", action="store_true", help="Force rule-only oracle, ignoring eval_protocol.yaml config.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write payloads or logs; only print the summary.")
    return parser.parse_args()


def _state_override_from_precursor_beam(
    args: argparse.Namespace,
    policies: List[Dict[str, Any]],
    threat_mode: str,
    base: Dict[str, Any],
    seed_action: str,
) -> Optional[Dict[str, Any]]:
    """Phase D: run a precursor BeamGenerator to the same threat_mode to
    produce a WM-predicted intermediate state, then return it for use as
    ``GenerationRequest.state_override`` on the main rephrase pass.

    The precursor shares defender configuration with the main rephrase
    (same --axis-* / --defender-* flags) so the state_override reflects a
    trajectory the same defender would actually see. ``rollout_mode`` is
    forced ``on`` regardless of --beam-rollout-mode: an override built
    from a score-only beam would be identical to the initial state, so
    composition is meaningful only when rollout is enabled.

    Returns ``None`` if the beam produced no visited nodes past the root.
    """
    precursor_args = argparse.Namespace(**vars(args))
    precursor_args.algorithm = "beam"
    precursor_args.beam_rollout_mode = "on"
    beam_generator = _build_generator(precursor_args, policies, threat_mode)

    from .action_proposer import infer_attack_intent

    budget = Budget(
        max_llm_calls=args.max_llm_calls,
        wall_time_sec=args.wall_time_sec,
    )
    attack_intent = infer_attack_intent(seed_action)
    try:
        _terminals, search_log = beam_generator._run_beam(  # type: ignore[attr-defined]
            intent=str(base.get("intent", "")),
            state=dict(base.get("state") or {}),
            attack_intent=attack_intent,
            budget=budget,
        )
    except Exception:
        return None
    # Pick the deepest visited node's predicted_state; fall back to the
    # first node's state if (pathologically) only root was visited, in
    # which case state_override is effectively a no-op (signature ==
    # base state signature) but the provenance flag still records that
    # composition was requested.
    if not search_log:
        return None
    deepest = max(search_log, key=lambda n: getattr(n, "depth", 0))
    predicted = getattr(deepest, "predicted_state", None)
    if isinstance(predicted, dict) and predicted:
        return dict(predicted)
    return None


def _run_for_threat_mode(
    args: argparse.Namespace,
    policies: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    threat_mode: str,
) -> List[Dict[str, Any]]:
    generator = _build_generator(args, policies, threat_mode)
    compose_via_beam = (
        getattr(args, "rephrase_state_override", "none") == "beam"
        and args.algorithm == "rephrase"
    )
    summaries: List[Dict[str, Any]] = []
    for base in selected:
        seeds = _collect_seed_actions(base, args.attack_payloads_root)
        if not seeds:
            summaries.append(
                {
                    "base_task_id": base.get("task_id"),
                    "threat_mode": threat_mode,
                    "status": "skipped",
                    "reason": "no_seed_actions_found",
                    "payloads_generated": 0,
                }
            )
            continue
        state_override = None
        if compose_via_beam:
            # Precursor beam uses the first seed as the adversarial anchor;
            # the resulting predicted_state is shared across this base
            # task's seeds (all seeds target the same initial env, so the
            # WM-predicted trajectory prefix is equally representative).
            state_override = _state_override_from_precursor_beam(
                args, policies, threat_mode, base, seeds[0]
            )
        request = GenerationRequest(
            base_task=base,
            seed_actions=seeds,
            n_variants=args.n_variants,
            max_payloads=args.max_payloads,
            threat_mode=threat_mode,
            attacker_seed=args.attacker_seed,
            state_override=state_override,
        )
        result = generator.generate(request)
        summary: Dict[str, Any] = {
            "base_task_id": base.get("task_id"),
            "threat_mode": threat_mode,
            "status": result.status,
            "reason": result.reason,
            "payloads_generated": len(result.payloads),
            "state_override_used": state_override is not None,
        }
        if result.payloads and not args.dry_run:
            if args.algorithm == "rephrase" and compose_via_beam:
                # Phase D composition deserves its own filename suffix so
                # it does not collide with vanilla rephrase output.
                suffix = f"__rephrase_beamcompose__{threat_mode}"
            elif args.algorithm != "rephrase":
                suffix = f"__{args.algorithm}__{threat_mode}"
            else:
                suffix = f"__{threat_mode}"
            out = args.output_dir / f"{base['task_id']}{suffix}.jsonl"
            _write_payloads(out, result.payloads)
            summary["output_path"] = str(out)
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    policies = load_policies(args.policies)
    all_base = _load_base_tasks(args.base_tasks)
    selected = _select_base_tasks(all_base, args)

    log_dir = None if args.dry_run else args.log_dir
    args.log_dir = log_dir

    modes = list(valid_threat_modes()) if args.all_threat_modes else [args.threat_mode]
    per_mode: List[Dict[str, Any]] = []
    for mode in modes:
        per_mode.extend(_run_for_threat_mode(args, policies, selected, mode))

    # Aggregate a quick fooling-rate-by-mode summary for the CLI output.
    by_mode: Dict[str, Dict[str, int]] = {}
    for entry in per_mode:
        slot = by_mode.setdefault(
            entry["threat_mode"],
            {"base_tasks": 0, "payloads": 0, "null_results": 0, "skipped": 0},
        )
        slot["base_tasks"] += 1
        slot["payloads"] += int(entry.get("payloads_generated", 0))
        if entry.get("status") == "null":
            slot["null_results"] += 1
        if entry.get("status") == "skipped":
            slot["skipped"] += 1

    print(
        json.dumps(
            {
                "threat_modes": modes,
                "per_mode_summary": by_mode,
                "results": per_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
