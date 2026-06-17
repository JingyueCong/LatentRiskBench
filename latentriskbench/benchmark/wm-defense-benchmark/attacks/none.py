from .base import AttackHandler


class NoAttackHandler(AttackHandler):
    def __init__(self) -> None:
        super().__init__(attack_id="attack_none", attack_family="none")

    def is_attacked(self, task):
        return False

    def apply_to_task(self, task, payload=None):
        transformed = dict(task)
        transformed.pop("attack_templates", None)
        transformed["attack_id"] = self.attack_id
        transformed["attack_family"] = self.attack_family
        transformed.setdefault("attack_depth", "L0")
        return self.prepare_task(transformed)
