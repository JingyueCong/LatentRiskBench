"""World-model-guided attack generator (Phase 1).

Generates adversarial attack payloads by searching a world model for action
variants that (a) the defender world model predicts as safe, and (b) an
independent policy oracle still flags as unsafe.

Public API:
    Budget, BudgetExhausted
    GenerationRecord, GenerationRequest, GenerationResult, VariantScore
    AttackerWM, TemplateAttacker, LLMAttacker
    DefenderWMScorer, StubDefenderWMScorer, LLMDefenderWMScorer
    RephraseGenerator

Private submodules should not be imported directly by other packages; the
``attacks/wm_adaptive.py`` handler only reads the generated JSONL payloads,
never the generator objects themselves.
"""
from __future__ import annotations

from .base import (
    AttackerAxes,
    BUDGET_OPTIONS,
    GenerationRequest,
    GenerationResult,
    POLICY_OPTIONS,
    PRE_FILTER_OPTIONS,
    RISK_SCORE_OPTIONS,
    ROLLOUT_OPTIONS,
    VariantScore,
)
from .budgets import Budget, BudgetExhausted
from .generation_record import GenerationRecord
from .attacker_wm import (
    AttackerWM,
    LLMAttacker,
    TemplateAttacker,
    build_attacker,
    valid_threat_modes,
)
from .defender_wm import (
    CoarseningDefenderScorer,
    DefenderVerdict,
    DefenderWMScorer,
    LLMDefenderWMScorer,
    StubDefenderWMScorer,
    WrapperDefenderWMScorer,
)
from .rephrase_generator import RephraseGenerator
from .action_proposer import (
    ActionProposer,
    LLMActionProposer,
    TemplateActionProposer,
    infer_attack_intent,
)
from .beam_generator import BeamGenerator, BeamNode
from .state_synthesis import fold_delta_into_state
from .adversarial_planner import (
    AdversarialPlanner,
    AgentPolicyModel,
    FirstCandidateAgentPolicy,
    KeywordBlacklistAgentPolicy,
    PlannerResult,
    PlannerVariantScore,
)

__all__ = [
    "ActionProposer",
    "AdversarialPlanner",
    "AgentPolicyModel",
    "AttackerAxes",
    "AttackerWM",
    "BeamGenerator",
    "BeamNode",
    "BUDGET_OPTIONS",
    "CoarseningDefenderScorer",
    "Budget",
    "BudgetExhausted",
    "DefenderVerdict",
    "DefenderWMScorer",
    "FirstCandidateAgentPolicy",
    "GenerationRecord",
    "GenerationRequest",
    "GenerationResult",
    "KeywordBlacklistAgentPolicy",
    "LLMActionProposer",
    "LLMAttacker",
    "LLMDefenderWMScorer",
    "PlannerResult",
    "PlannerVariantScore",
    "POLICY_OPTIONS",
    "PRE_FILTER_OPTIONS",
    "RISK_SCORE_OPTIONS",
    "ROLLOUT_OPTIONS",
    "RephraseGenerator",
    "StubDefenderWMScorer",
    "TemplateActionProposer",
    "TemplateAttacker",
    "VariantScore",
    "WrapperDefenderWMScorer",
    "build_attacker",
    "fold_delta_into_state",
    "infer_attack_intent",
    "valid_threat_modes",
]
