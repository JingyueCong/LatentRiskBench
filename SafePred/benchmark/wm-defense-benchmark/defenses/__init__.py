from __future__ import annotations

from typing import Any, Dict, List

from .base import DefenseHandler
from .guard_agent import GuardAgentDefenseHandler
from .keyword_list import KeywordListDefenseHandler
from .llama_guard import LlamaGuardDefenseHandler
from .llm_judge import LLMJudgeDefenseHandler
from .llm_judge_strong import StrongLLMJudgeDefenseHandler
from .none import NoDefenseHandler
from .rule_filter import RuleFilterDefenseHandler, TrajectoryRuleFilterDefenseHandler
from .world_model_gating import WorldModelGatingDefenseHandler

_REGISTRY = {
    "defense_none": NoDefenseHandler(),
    "none": NoDefenseHandler(),
    "defense_keyword_list": KeywordListDefenseHandler(),
    "defense_rule_filter": RuleFilterDefenseHandler(),
    "defense_rule_filter_trajectory": TrajectoryRuleFilterDefenseHandler(),
    "defense_llm_judge": LLMJudgeDefenseHandler(),
    # Recognized strong baselines (issue LLM calls; OpenRouter/custom creds).
    "defense_llm_judge_strong": StrongLLMJudgeDefenseHandler(),
    "defense_llama_guard": LlamaGuardDefenseHandler(),
    "defense_guard_agent": GuardAgentDefenseHandler(),
    "defense_world_model_gating": WorldModelGatingDefenseHandler(),
    "defense_default": WorldModelGatingDefenseHandler(),
}


def list_defense_handlers() -> List[DefenseHandler]:
    ordered: List[DefenseHandler] = []
    seen = set()
    for handler in _REGISTRY.values():
        key = handler.defense_id
        if key in seen:
            continue
        seen.add(key)
        ordered.append(handler)
    return ordered


def list_defense_metadata() -> List[Dict[str, Any]]:
    return [handler.metadata() for handler in list_defense_handlers()]


def get_defense_handler(defense_id: str) -> DefenseHandler:
    return _REGISTRY.get(defense_id, WorldModelGatingDefenseHandler())
