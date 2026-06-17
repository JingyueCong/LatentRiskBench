from __future__ import annotations

from typing import Any, Dict, List

from .base import AgentHandler
from .first_candidate import FirstCandidateAgentHandler
from .heuristic_ranker import HeuristicRankerAgentHandler
from .keyword_guarded import KeywordGuardedAgentHandler
from .llm_planner import LLMPlannerAgentHandler
from .planner import PlannerAgentHandler
from .replay import ReplayAgentHandler

_REGISTRY = {
    "agent_replay": ReplayAgentHandler(),
    "agent_default": ReplayAgentHandler(),
    "agent_first_candidate": FirstCandidateAgentHandler(),
    "agent_keyword_guarded": KeywordGuardedAgentHandler(),
    "agent_heuristic_ranker": HeuristicRankerAgentHandler(),
    "agent_planner": PlannerAgentHandler(),
    "agent_llm_planner": LLMPlannerAgentHandler(),
}


def list_agent_handlers() -> List[AgentHandler]:
    ordered: List[AgentHandler] = []
    seen = set()
    for handler in _REGISTRY.values():
        if handler.agent_id in seen:
            continue
        seen.add(handler.agent_id)
        ordered.append(handler)
    return ordered


def list_agent_metadata() -> List[Dict[str, Any]]:
    return [handler.metadata() for handler in list_agent_handlers()]


def get_agent_handler(agent_mode: str, task: Dict[str, Any], config_path: Any = None) -> AgentHandler:
    if agent_mode == "first_candidate":
        return _REGISTRY["agent_first_candidate"]
    if agent_mode == "keyword_guarded":
        return _REGISTRY["agent_keyword_guarded"]
    if agent_mode == "heuristic_ranker":
        return _REGISTRY["agent_heuristic_ranker"]
    if agent_mode == "planner":
        return _REGISTRY["agent_planner"]
    if agent_mode == "llm_planner":
        handler = _REGISTRY["agent_llm_planner"]
        if hasattr(handler, "set_config_path"):
            handler.set_config_path(config_path)
        return handler
    if agent_mode in {"replay", "from_task"}:
        agent_id = str(task.get("agent_id", "agent_replay"))
        return _REGISTRY.get(agent_id, _REGISTRY["agent_replay"])
    return _REGISTRY["agent_replay"]
