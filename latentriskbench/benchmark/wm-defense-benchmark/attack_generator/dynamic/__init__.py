"""Dynamic attack generators (Category 1: WM-conditioned).

All generators in this subpackage take (task, state, history, policies, WM,
oracle, budget) and produce frozen JSONL payloads ready for consumption by
the ``attacks/*`` handlers. The key distinction from the older
``TemplateAttacker`` / ``LLMAttacker`` in ``attacker_wm.py``:

- Those only see action text and rephrase it.
- Dynamic generators see the full (state, history, plan) and use the WM
  to predict what adversarial content would maximise agent-drift.

Each generator emits payloads with an explicit ``generation_metadata``
block containing the generator name, model config, WM-predicted risk, and
oracle verdict — so downstream audit can reconstruct provenance.
"""
from __future__ import annotations

from .base import (
    DynamicAttackGenerator,
    DynamicGenerationResult,
    build_openrouter_client,
    build_payload_stub,
    verify_with_oracle,
)
from .state_aware_injection import StateAwareInjectionGenerator
from .multi_step_trap import MultiStepTrapGenerator
from .plan_aware_phishing import PlanAwarePhishingGenerator
from .contextual_lure import ContextualLureGenerator
from .tool_sequence_misdirection import ToolSequenceMisdirectionGenerator


__all__ = [
    "DynamicAttackGenerator",
    "DynamicGenerationResult",
    "StateAwareInjectionGenerator",
    "MultiStepTrapGenerator",
    "PlanAwarePhishingGenerator",
    "ContextualLureGenerator",
    "ToolSequenceMisdirectionGenerator",
    "build_openrouter_client",
    "build_payload_stub",
    "verify_with_oracle",
]
