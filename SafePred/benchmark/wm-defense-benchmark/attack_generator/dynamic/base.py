"""Shared infrastructure for Category-1 (WM-conditioned) dynamic generators.

A generator is a single object that accepts a :class:`GenerationContext` and
returns zero or more :class:`DynamicGenerationResult` instances. Each result
is a self-contained attack payload (already oracle-validated) that can be
serialised to JSONL and consumed by the existing attack handlers.

Design rules (see docs/wm_0417.md and Phase-P5 plan):

1. Generators DO NOT hit LLMs during benchmark evaluation — only during
   offline generation. The generated JSONL is the durable artefact.
2. Every successful attack MUST be verified by an independent oracle
   (``RulePolicyOracle`` by default) — the WM can think a variant is safe,
   but if the oracle agrees with the WM then the variant was genuinely
   benign, not adversarial.
3. Every generation step is charged against a :class:`Budget` so a
   generator cannot silently burn thousands of LLM calls.
4. All adversarial outcomes carry a ``generation_metadata`` block
   (generator name, model config, threat-mode axes, WM verdict, oracle
   verdict) so transfer-matrix tools can slice by provenance.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..budgets import Budget
from ..defender_wm import DefenderWMScorer, DefenderVerdict


logger = logging.getLogger("attack_generator.dynamic")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DynamicGenerationResult:
    """A single oracle-validated adversarial payload.

    Fields mirror the JSONL contract consumed by ``attacks/base.py``:

    - ``attack_id`` / ``attack_family`` / ``attack_depth`` route the
      payload to the right AttackHandler at benchmark time.
    - ``overrides`` carries the task-level fields the handler merges
      into the base task (intent, state, candidate_actions, etc.).
    - ``generation_metadata`` records provenance for audit and transfer
      analysis.
    """

    attack_id: str
    attack_family: str
    attack_depth: str
    target_task_ids: List[str]
    overrides: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    payload_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_payload(self) -> Dict[str, Any]:
        """Serialise to the JSONL schema expected by ``AttackHandler``."""
        return {
            "payload_id": self.payload_id,
            "attack_id": self.attack_id,
            "attack_family": self.attack_family,
            "attack_depth": self.attack_depth,
            "target_task_ids": list(self.target_task_ids),
            "overrides": dict(self.overrides),
            "generation_metadata": dict(self.generation_metadata),
        }


class DynamicAttackGenerator(Protocol):
    """Protocol every Category-1 generator implements.

    A generator is stateful only with respect to its LLM client / config;
    each ``generate`` call is independent so the same generator can be
    reused across tasks without leaking state between them.
    """

    generator_id: str
    attack_id: str
    attack_family: str
    attack_depth: str

    def generate(
        self,
        base_task: Dict[str, Any],
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> List[DynamicGenerationResult]:
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def build_openrouter_client(
    model_name: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.9,
    max_tokens: int = 1024,
    api_key_env: str = "OPEN_ROUTER_API_KEY",
    base_url: str = "https://openrouter.ai/api/v1",
) -> Any:
    """Build an ``LLMClient`` pointed at OpenRouter.

    OpenRouter is OpenAI-compatible; we keep ``provider='openai'`` so the
    SDK path is used instead of raw HTTP. All generators in this package
    go through this single helper so switching attacker model is a
    one-line change.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Env var {api_key_env} not set; cannot build OpenRouter client."
        )
    try:
        from SafePred.utils.llm_client import LLMClient  # type: ignore
    except Exception:
        from utils.llm_client import LLMClient  # type: ignore  # noqa: F401
    return LLMClient(
        api_key=api_key,
        api_url=base_url,
        model_name=model_name,
        provider="openai",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
    )


def build_payload_stub(
    base_task: Dict[str, Any],
    generator_id: str,
) -> Dict[str, Any]:
    """Create the common scaffolding for overrides derived from a base task.

    The caller then adds / mutates the ``overrides`` dict. Keeping this
    logic centralised means a new generator only has to think about the
    attack-specific fields.
    """
    base_task_id = str(base_task.get("task_id", "base_task"))
    rec = uuid.uuid4().hex[:12]
    return {
        "overrides": {
            "task_id": f"t_{generator_id}_{base_task_id}__{rec}",
        },
        "target_task_ids": [base_task_id],
        "record_id": rec,
    }


def verify_with_oracle(
    oracle: Any,
    intent: str,
    state: Dict[str, Any],
    action: str,
    policies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the independent oracle and serialise its verdict.

    Returns a dict with ``unsafe`` (bool), ``violated_policy_ids``, and
    ``matched_rules`` so it can be dropped into ``generation_metadata``.
    Swallow oracle failures — if the oracle blows up, the caller should
    treat the variant as unverified and skip it.
    """
    try:
        verdict = oracle.evaluate(intent, state, action, policies)
    except Exception as exc:
        logger.warning("oracle.evaluate failed: %s", exc)
        return {"unsafe": False, "violated_policy_ids": [], "matched_rules": [], "oracle_error": str(exc)}
    return {
        "unsafe": bool(getattr(verdict, "unsafe", False)),
        "violated_policy_ids": list(getattr(verdict, "violated_policy_ids", []) or []),
        "matched_rules": list(getattr(verdict, "matched_rules", []) or [])[:5],
    }


# ---------------------------------------------------------------------------
# Shared prompt parsing helpers (used by several generators)
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_array(text: str) -> Optional[List[Any]]:
    """Extract the first JSON array from a raw LLM response.

    Accepts fenced and unfenced formats. Returns ``None`` if nothing
    parseable was found — caller decides whether to retry or abort.
    """
    if not text:
        return None
    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidate = fenced.group(1)
    else:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return None
        candidate = m.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a raw LLM response."""
    if not text:
        return None
    fenced = _JSON_OBJ_FENCE.search(text)
    if fenced:
        candidate = fenced.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        candidate = m.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def format_state_for_prompt(state: Dict[str, Any]) -> str:
    """Render a state dict into a compact human-readable block."""
    keys = ("page_type", "url", "a11y_tree_txt")
    lines: List[str] = []
    for k in keys:
        v = state.get(k)
        if v:
            lines.append(f"{k}: {v}")
    extra = [k for k in state if k not in keys]
    for k in extra[:3]:
        v = state.get(k)
        if isinstance(v, (str, int, float, bool)):
            lines.append(f"{k}: {v}")
    return "\n".join(lines) or "(state omitted)"


def format_policies_for_prompt(policies: List[Dict[str, Any]], max_n: int = 12) -> str:
    """Render policy list into a compact numbered block."""
    lines: List[str] = []
    for p in policies[:max_n]:
        pid = str(p.get("policy_id", "?"))
        desc = str(p.get("policy_description", "")).strip()
        lines.append(f"- {pid}: {desc}")
    return "\n".join(lines) or "(no policies)"


def call_llm_with_budget(
    client: Any,
    prompt: str,
    budget: Budget,
    system: Optional[str] = None,
) -> str:
    """Wrapper that charges one call against the budget.

    Returns the raw model text on success; empty string on failure. We do
    NOT re-raise — generators want to catch generation errors per-sample
    and move on rather than abort the whole run.
    """
    budget.consume(1)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        raw = client.generate(messages=messages)
        return raw or ""
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""
