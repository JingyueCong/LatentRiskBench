from __future__ import annotations

from typing import Any, Dict

from .base import AttackHandler


class LatentSpaceAttackHandler(AttackHandler):
    def __init__(self) -> None:
        super().__init__(
            attack_id="attack_l3_latent_space",
            attack_family="latent_space",
        )

    def infer_attack_depth(self, task: Dict[str, Any]) -> str:
        if "attack_depth" in task:
            return str(task["attack_depth"])
        return "L3"
