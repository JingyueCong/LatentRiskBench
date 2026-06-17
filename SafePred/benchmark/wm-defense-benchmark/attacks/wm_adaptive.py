"""Attack handler for WM-adaptive attacks (Phase 1+).

The handler itself is intentionally trivial: it loads pre-generated payload
JSONL files from ``data/attack_payloads/wm_adaptive/`` exactly the same way
the L1/L2/L3 handlers load their hand-authored payloads. All adaptive
search logic lives in ``attack_generator/``; by the time the benchmark
runs, the payloads have been materialised and can be audited from
``data/attack_generation_logs/``.
"""
from __future__ import annotations

from typing import Any, Dict

from .base import AttackHandler


class WMAdaptiveAttackHandler(AttackHandler):
    def __init__(self) -> None:
        super().__init__(
            attack_id="attack_l4_wm_adaptive",
            attack_family="wm_adaptive",
        )

    def infer_attack_depth(self, task: Dict[str, Any]) -> str:
        if "attack_depth" in task:
            return str(task["attack_depth"])
        return "L4"
