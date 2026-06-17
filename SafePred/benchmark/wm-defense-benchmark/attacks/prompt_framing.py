from __future__ import annotations

from typing import Any, Dict

from .base import AttackHandler


class PromptFramingAttackHandler(AttackHandler):
    """Hand-crafted semantic-framing attacks (social engineering via prompt).

    Previously mis-labeled as ``attack_l3_latent_space``; those payloads never
    touched a latent space and were human-authored scenarios (emergency-mode
    context, camouflage as routine sync, pre-approved change framing, etc.).
    They are kept as a distinct family so the ``attack_l3_latent_space`` ID
    can be reserved for WM-generated / gradient-based latent-space attacks.

    Depth stays L3 because these require compositional semantic reasoning
    to resist, not surface pattern matching.
    """

    def __init__(self) -> None:
        super().__init__(
            attack_id="attack_prompt_framing",
            attack_family="prompt_framing",
        )

    def infer_attack_depth(self, task: Dict[str, Any]) -> str:
        if "attack_depth" in task:
            return str(task["attack_depth"])
        return "L3"
