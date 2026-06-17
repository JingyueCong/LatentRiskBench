from __future__ import annotations

from typing import Any, Dict, List

from .base import AttackHandler
from .multi_step_trigger import MultiStepTriggerAttackHandler
from .none import NoAttackHandler
from .observation_tampering import ObservationTamperingAttackHandler
from .latent_space import LatentSpaceAttackHandler
from .prompt_framing import PromptFramingAttackHandler
from .prompt_injection import PromptInjectionAttackHandler
from .wm_adaptive import WMAdaptiveAttackHandler

_FAMILY_REGISTRY = {
    "none": NoAttackHandler(),
    "prompt_injection": PromptInjectionAttackHandler(),
    "observation_tampering": ObservationTamperingAttackHandler(),
    "multi_step_trigger": MultiStepTriggerAttackHandler(),
    "prompt_framing": PromptFramingAttackHandler(),
    "latent_space": LatentSpaceAttackHandler(),
    "wm_adaptive": WMAdaptiveAttackHandler(),
}

_ID_REGISTRY = {
    "attack_none": _FAMILY_REGISTRY["none"],
    "none": _FAMILY_REGISTRY["none"],
    "attack_l1_injection": _FAMILY_REGISTRY["prompt_injection"],
    "attack_observation_tampering": _FAMILY_REGISTRY["observation_tampering"],
    "attack_l2_chain": _FAMILY_REGISTRY["multi_step_trigger"],
    "attack_prompt_framing": _FAMILY_REGISTRY["prompt_framing"],
    "attack_l3_latent_space": _FAMILY_REGISTRY["latent_space"],
    "attack_l4_wm_adaptive": _FAMILY_REGISTRY["wm_adaptive"],
}


def list_attack_handlers() -> List[AttackHandler]:
    ordered: List[AttackHandler] = []
    seen = set()
    for handler in _FAMILY_REGISTRY.values():
        key = (handler.attack_id, handler.attack_family)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(handler)
    return ordered


def list_attack_metadata() -> List[Dict[str, Any]]:
    return [handler.metadata() for handler in list_attack_handlers()]


def get_attack_handler(task: Dict[str, Any]) -> AttackHandler:
    attack_family = str(task.get("attack_family", ""))
    if attack_family and attack_family in _FAMILY_REGISTRY:
        return _FAMILY_REGISTRY[attack_family]

    attack_id = str(task.get("attack_id", "attack_none"))
    if attack_id in _ID_REGISTRY:
        return _ID_REGISTRY[attack_id]

    if attack_id in {"attack_none", "none"}:
        return _FAMILY_REGISTRY["none"]

    return AttackHandler(attack_id=attack_id, attack_family=attack_family or attack_id)
