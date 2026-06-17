from .base import AttackHandler


class MultiStepTriggerAttackHandler(AttackHandler):
    def __init__(self) -> None:
        super().__init__(attack_id="attack_l2_chain", attack_family="multi_step_trigger")
