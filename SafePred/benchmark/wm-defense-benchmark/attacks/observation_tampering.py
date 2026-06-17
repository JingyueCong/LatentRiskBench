from .base import AttackHandler


class ObservationTamperingAttackHandler(AttackHandler):
    def __init__(self) -> None:
        super().__init__(attack_id="attack_observation_tampering", attack_family="observation_tampering")

    def infer_attack_depth(self, task):
        if "attack_depth" in task:
            return str(task["attack_depth"])
        return "L1"
