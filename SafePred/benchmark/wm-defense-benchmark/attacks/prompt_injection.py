from .base import AttackHandler


class PromptInjectionAttackHandler(AttackHandler):
    def __init__(self) -> None:
        super().__init__(attack_id="attack_l1_injection", attack_family="prompt_injection")
