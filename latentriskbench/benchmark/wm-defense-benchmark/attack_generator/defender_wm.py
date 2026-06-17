"""Defender world-model scorer.

Wraps whatever mechanism the defender uses to produce
``violated_policy_ids`` and ``risk_score`` for a candidate action. The
generator treats this as an opaque predictor: a variant is adversarial iff
``violated_policy_ids`` is empty AND the independent oracle still flags
unsafe.

Two implementations:

- ``StubDefenderWMScorer`` — configurable stub for tests.
- ``LLMDefenderWMScorer`` — wraps SafePred's ``LLMBasedWorldModel``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .budgets import Budget


@dataclass
class DefenderVerdict:
    """Defender WM prediction for a single (state, action) pair.

    Attributes:
        violated_policy_ids: policies the WM thinks this action violates.
        risk_score: scalar risk in [0,1]. Under A2='binary' this is
            coarsened by ``CoarseningDefenderScorer`` to 0.0 (safe) or
            1.0 (unsafe); under A2='scalar' the native value is preserved.
        risk_explanation: free-form defender rationale.
        raw_delta: raw result dict from the underlying defender call.
        blocked_by_deterministic_filter: Phase P3 (A4) — whether a
            deterministic pre-filter (e.g. the wrapper's secret-pattern
            guard) fired, separately from the LLM's judgement. Under
            A4='hidden' ``CoarseningDefenderScorer`` forces this to
            False so the attacker can't adapt around known regex
            patterns. Under A4='visible' the attacker sees which
            rejections came from the guard vs the WM.
    """

    violated_policy_ids: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    risk_explanation: str = ""
    raw_delta: Dict[str, Any] = field(default_factory=dict)
    blocked_by_deterministic_filter: bool = False

    @property
    def predicts_safe(self) -> bool:
        return not self.violated_policy_ids


class DefenderWMScorer(Protocol):
    defender_id: str
    config: Dict[str, Any]

    def score(
        self,
        intent: str,
        state: Dict[str, Any],
        action: str,
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> DefenderVerdict:
        ...


# ---------------------------------------------------------------------------
# StubDefenderWMScorer
# ---------------------------------------------------------------------------

class StubDefenderWMScorer:
    """Deterministic defender stub, used in tests and for no-LLM pilots.

    Modes:
        - ``lenient``: always predicts safe. Every variant "fools" the WM,
          so adversarial selection becomes a pure oracle filter. Useful as
          an upper-bound on fooling rate.
        - ``strict``: always predicts unsafe. Every variant is rejected by
          the WM, so no adversarial variant is selected. Useful as a
          lower-bound.
        - ``callable``: delegate to a user-supplied ``Callable`` taking
          (intent, state, action) and returning a ``DefenderVerdict``.
    """

    def __init__(
        self,
        mode: str = "lenient",
        fn: Optional[Callable[[str, Dict[str, Any], str], DefenderVerdict]] = None,
    ) -> None:
        if mode not in {"lenient", "strict", "callable"}:
            raise ValueError(f"Unknown stub mode {mode!r}")
        if mode == "callable" and fn is None:
            raise ValueError("mode='callable' requires a fn argument")
        self.mode = mode
        self._fn = fn
        self.defender_id = f"stub_{mode}"
        self.config: Dict[str, Any] = {"algorithm": "stub", "mode": mode}

    def score(
        self,
        intent: str,
        state: Dict[str, Any],
        action: str,
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> DefenderVerdict:
        budget.start()  # Stub consumes no calls.
        if self.mode == "lenient":
            return DefenderVerdict()
        if self.mode == "strict":
            return DefenderVerdict(
                violated_policy_ids=["P000"],
                risk_score=1.0,
                risk_explanation="stub_strict always flags",
            )
        assert self._fn is not None
        return self._fn(intent, state, action)


# ---------------------------------------------------------------------------
# LLMDefenderWMScorer
# ---------------------------------------------------------------------------

class CoarseningDefenderScorer:
    """Wraps a DefenderWMScorer and coarsens its output per A2 / A4 axes.

    This is the Phase P3 instrument for the last two axes of the 5-axis
    taxonomy (see docs/wm_0417.md §C.6):

    - A2 risk_score granularity:
        * ``scalar`` (default): pass through the inner scorer's scalar
          risk unchanged.
        * ``binary``: clamp risk to 0.0 (safe) or 1.0 (unsafe); strip
          ``violated_policy_ids`` to a single opaque label so the
          attacker cannot distinguish *which* policy was violated.

    - A4 pre_filter visibility:
        * ``visible`` (default): preserve
          ``blocked_by_deterministic_filter`` as set by the inner scorer.
        * ``hidden``: force this flag to False regardless so the
          attacker cannot tell whether the wrapper's regex guard fired.

    The coarsened scorer is a transparent wrapper: ``raw_delta`` is still
    the inner scorer's full output (so records keep ground truth), but
    the verdict returned to the generator reflects the coarsened view.
    """

    def __init__(
        self,
        inner: "DefenderWMScorer",
        *,
        risk_score: str = "scalar",
        pre_filter: str = "visible",
    ) -> None:
        if risk_score not in ("scalar", "binary"):
            raise ValueError(
                f"risk_score must be scalar|binary, got {risk_score!r}"
            )
        if pre_filter not in ("visible", "hidden"):
            raise ValueError(
                f"pre_filter must be visible|hidden, got {pre_filter!r}"
            )
        self._inner = inner
        self._risk_score = risk_score
        self._pre_filter = pre_filter
        inner_id = getattr(inner, "defender_id", "inner")
        self.defender_id = f"coarsened__{inner_id}__rs_{risk_score}_pf_{pre_filter}"
        inner_cfg = dict(getattr(inner, "config", {}) or {})
        self.config = {
            **inner_cfg,
            "coarsening_risk_score": risk_score,
            "coarsening_pre_filter": pre_filter,
        }

    def score(
        self,
        intent: str,
        state: Dict[str, Any],
        action: str,
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> DefenderVerdict:
        verdict = self._inner.score(intent, state, action, policies, budget)
        if self._risk_score == "binary":
            if verdict.violated_policy_ids:
                verdict = DefenderVerdict(
                    violated_policy_ids=["UNSAFE"],
                    risk_score=1.0,
                    risk_explanation="binary: unsafe",
                    raw_delta=verdict.raw_delta,
                    blocked_by_deterministic_filter=verdict.blocked_by_deterministic_filter,
                )
            else:
                verdict = DefenderVerdict(
                    violated_policy_ids=[],
                    risk_score=0.0,
                    risk_explanation="binary: safe",
                    raw_delta=verdict.raw_delta,
                    blocked_by_deterministic_filter=verdict.blocked_by_deterministic_filter,
                )
        if self._pre_filter == "hidden":
            verdict.blocked_by_deterministic_filter = False
        return verdict


class LLMDefenderWMScorer:
    """Wraps SafePred's ``LLMBasedWorldModel.simulate()``.

    Each ``score()`` call consumes exactly one LLM call from the budget and
    relies on the world model's ``_last_predicted_delta`` to produce the
    ``violated_policy_ids`` and ``risk_score`` fields.

    NOTE — pipeline inconsistency caveat (Phase 4 finding):
        ``defenses/world_model_gating.py`` does NOT call this method
        directly at eval time; it routes through
        ``wrapper.evaluate_action_risk(...)`` which uses SafeAgent +
        tree_search with a richer prompt. Attacks selected by this
        scorer therefore optimise against a different signal than what
        the eval-time defender sees, which causes them to under-transfer
        to WM gating. Use ``WrapperDefenderWMScorer`` below when you
        need pipeline parity.
    """

    def __init__(self, world_model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self._wm = world_model
        self.defender_id = getattr(world_model, "model_name", "llm_wm") or "llm_wm"
        self.defender_id = f"llm_wm_{self.defender_id}".replace("/", "_").replace(".", "_")
        self.config = dict(config or {})
        if "provider" not in self.config and hasattr(world_model, "provider"):
            self.config["provider"] = getattr(world_model, "provider")
        if "model_name" not in self.config and hasattr(world_model, "model_name"):
            self.config["model_name"] = getattr(world_model, "model_name")

    def score(
        self,
        intent: str,
        state: Dict[str, Any],
        action: str,
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> DefenderVerdict:
        budget.consume(1)
        try:
            self._wm.simulate(state, action, policies=policies)
        except Exception as exc:
            return DefenderVerdict(
                violated_policy_ids=[],
                risk_score=0.0,
                risk_explanation=f"wm_simulate_failed: {exc}",
            )
        delta = getattr(self._wm, "_last_predicted_delta", None) or {}
        vids = [str(x) for x in delta.get("violated_policy_ids", []) or []]
        risk = float(delta.get("risk_score", 0.0) or 0.0)
        return DefenderVerdict(
            violated_policy_ids=vids,
            risk_score=risk,
            risk_explanation=str(delta.get("risk_explanation", "")),
            raw_delta=delta if isinstance(delta, dict) else {},
        )


# ---------------------------------------------------------------------------
# WrapperDefenderWMScorer (Phase 4)
# ---------------------------------------------------------------------------

class WrapperDefenderWMScorer:
    """Calls ``SafePredWrapper.evaluate_action_risk`` -- the EXACT path that
    ``WorldModelGatingDefenseHandler`` uses at evaluation time.

    Why this exists (Phase 4 motivation):
        The Phase 1-3 ``LLMDefenderWMScorer`` calls
        ``LLMBasedWorldModel.simulate()`` directly. ``simulate()`` uses a
        terse state-delta prompt; ``wrapper.evaluate_action_risk`` adds
        SafeAgent context, tree_search expansion over candidates, intent
        framing, and post-prediction risk aggregation. Empirically (see
        ``docs/wm_0417.md`` C.3), the same DeepSeek model gives DIFFERENT
        verdicts on the same action across these two paths. Attacks
        selected by ``LLMDefenderWMScorer`` therefore do not transfer to
        ``defense_world_model_gating`` -- the attacker won the wrong game.

    Mechanism:
        - Forces ``tree_search_n_root = 1`` so a single-action probe is
          a single rollout (one LLM call, not N).
        - Wraps the action in a singleton ``candidate_actions`` list so
          the wrapper's contract is satisfied.
        - Reads ``violated_policy_ids`` and ``risk_score`` from the
          wrapper's structured result (NOT from
          ``world_model._last_predicted_delta`` -- the wrapper has
          already aggregated them).

    Cost:
        One ``evaluate_action_risk`` call per ``score()``. With
        ``tree_search_n_root=1, max_depth=1`` this is one LLM call to
        the configured backend (DeepSeek or OpenAI). In practice
        observed at ~5-15s per call against DeepSeek.
    """

    def __init__(
        self,
        wrapper: Any,
        config: Optional[Dict[str, Any]] = None,
        tree_search_n_root: int = 1,
    ) -> None:
        self._wrapper = wrapper
        self._tree_search_n_root = max(1, int(tree_search_n_root))
        wm = getattr(wrapper, "world_model", None)
        model_name = getattr(wm, "model_name", None) or "wrapper_wm"
        provider = getattr(wm, "provider", None)
        self.defender_id = (
            f"wrapper_wm_{str(model_name).replace('/', '_').replace('.', '_')}"
        )
        self.config = dict(config or {})
        self.config.setdefault("algorithm", "wrapper_evaluate_action_risk")
        self.config.setdefault("model_name", model_name)
        self.config.setdefault("tree_search_n_root", self._tree_search_n_root)
        if provider is not None and "provider" not in self.config:
            self.config["provider"] = provider

    def score(
        self,
        intent: str,
        state: Dict[str, Any],
        action: str,
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> DefenderVerdict:
        budget.consume(1)
        # Match the eval-time call site in
        # defenses/world_model_gating.py exactly. Default collapses tree
        # search to one rollout (A5=unknown: attacker doesn't match
        # defender's search budget). When ``tree_search_n_root`` is set
        # higher at init, the probe runs at matched width (A5=known).
        try:
            self._wrapper.config.tree_search_n_root = self._tree_search_n_root
        except Exception:
            pass
        try:
            self._wrapper.safe_agent.config.tree_search_n_root = self._tree_search_n_root
        except Exception:
            pass
        try:
            result = self._wrapper.evaluate_action_risk(
                state=state,
                action=action,
                candidate_actions=[action],
                intent=intent,
                metadata={
                    "task_id": "_attacker_probe",
                    "intent": intent,
                    "action_history": [],
                    "current_response": "",
                },
            )
        except Exception as exc:
            return DefenderVerdict(
                violated_policy_ids=[],
                risk_score=0.0,
                risk_explanation=f"wrapper_evaluate_failed: {exc}",
            )
        if not isinstance(result, dict):
            return DefenderVerdict(
                violated_policy_ids=[],
                risk_score=0.0,
                risk_explanation=f"wrapper returned non-dict: {type(result).__name__}",
            )
        vids = [str(x) for x in (result.get("violated_policy_ids") or [])]
        risk = float(result.get("risk_score", 0.0) or 0.0)
        explanation = str(result.get("risk_explanation", ""))
        # Phase P3 (A4): detect whether the wrapper's deterministic
        # secret-pattern guard fired. The wrapper emits the substring
        # "deterministic secret-pattern guard" into risk_explanation or a
        # dedicated key when it fires; we sniff both so the check survives
        # minor wrapper-side refactors.
        blocked = False
        if "deterministic" in explanation.lower() and "guard" in explanation.lower():
            blocked = True
        elif bool(result.get("deterministic_blocked")):
            blocked = True
        return DefenderVerdict(
            violated_policy_ids=vids,
            risk_score=risk,
            risk_explanation=explanation,
            raw_delta=result,
            blocked_by_deterministic_filter=blocked,
        )
