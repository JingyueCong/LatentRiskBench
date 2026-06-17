from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AttackHandler:
    attack_id: str
    attack_family: str

    def prepare_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(task)
        enriched.setdefault("attack_id", self.attack_id)
        enriched.setdefault("attack_family", self.attack_family)
        enriched.setdefault("attack_depth", self.infer_attack_depth(enriched))
        return enriched

    def template_keys(self) -> tuple[str, ...]:
        return (self.attack_id, self.attack_family)

    def can_apply(self, task: Dict[str, Any]) -> bool:
        if str(task.get("attack_id", "")) == self.attack_id:
            return True
        if str(task.get("attack_family", "")) == self.attack_family:
            return True
        attack_templates = task.get("attack_templates", {})
        if not isinstance(attack_templates, dict):
            return False
        return any(key in attack_templates for key in self.template_keys())

    def metadata(self) -> Dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "attack_family": self.attack_family,
            "default_attack_depth": self.infer_attack_depth({}),
        }

    def _payload_matches(self, task: Dict[str, Any], payload: Dict[str, Any]) -> bool:
        target_ids = payload.get("target_task_ids", [])
        if target_ids and str(task.get("task_id", "")) not in {str(x) for x in target_ids}:
            return False
        task_tags = set(str(x) for x in task.get("tags", []))
        target_tags = payload.get("target_tags", [])
        if target_tags and not task_tags.intersection(str(x) for x in target_tags):
            return False
        return True

    def apply_to_task(self, task: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if str(task.get("attack_id", "")) == self.attack_id or str(task.get("attack_family", "")) == self.attack_family:
            return self.prepare_task(task)

        if payload is not None:
            if not self._payload_matches(task, payload):
                return None
            overrides = payload.get("overrides", {})
            if not isinstance(overrides, dict):
                return None
            transformed = dict(task)
            transformed.pop("attack_templates", None)
            for key, value in overrides.items():
                transformed[key] = value
            transformed["attack_id"] = str(payload.get("attack_id", self.attack_id))
            transformed["attack_family"] = str(payload.get("attack_family", self.attack_family))
            if "attack_depth" in payload:
                transformed["attack_depth"] = str(payload["attack_depth"])
            if "payload_id" in payload and "task_id" not in overrides:
                base_task_id = str(task.get("task_id", "task"))
                transformed["task_id"] = f"{base_task_id}__{payload['payload_id']}"
            # Preserve generation_metadata onto the task so downstream metrics
            # (wm_fooling_rate, threat-mode slicing, transfer rate) can read
            # the attacker/defender/oracle provenance without re-opening the
            # payload JSONL.
            if isinstance(payload.get("generation_metadata"), dict):
                transformed["generation_metadata"] = dict(payload["generation_metadata"])
            if "payload_id" in payload:
                transformed.setdefault("payload_id", str(payload["payload_id"]))
            return self.prepare_task(transformed)

        attack_templates = task.get("attack_templates", {})
        if not isinstance(attack_templates, dict):
            return None

        payload = None
        for key in self.template_keys():
            if key in attack_templates and isinstance(attack_templates[key], dict):
                payload = attack_templates[key]
                break
        if payload is None:
            return None

        transformed = dict(task)
        transformed.pop("attack_templates", None)
        for key, value in payload.items():
            transformed[key] = value

        base_task_id = str(task.get("task_id", "task"))
        transformed.setdefault("task_id", f"{base_task_id}__{self.attack_id}")
        return self.prepare_task(transformed)

    def infer_attack_depth(self, task: Dict[str, Any]) -> str:
        if "attack_depth" in task:
            return str(task["attack_depth"])

        attack_id = str(task.get("attack_id", self.attack_id))
        if attack_id in {"attack_none", "none"}:
            return "L0"
        if attack_id.startswith("attack_l") and len(attack_id) > len("attack_l"):
            suffix = attack_id[len("attack_l")]
            if suffix.isdigit():
                return f"L{suffix}"
        return "unknown"

    def is_attacked(self, task: Dict[str, Any]) -> bool:
        attack_id = str(task.get("attack_id", self.attack_id))
        attack_depth = str(task.get("attack_depth", self.infer_attack_depth(task)))
        if attack_depth.upper() == "L0":
            return False
        return attack_id not in {"attack_none", "none"}
