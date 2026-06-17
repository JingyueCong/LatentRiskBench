"""Audit record for a single generation attempt.

Every generation attempt (one seed action -> N variants) produces exactly
one ``GenerationRecord`` regardless of success. Records are:
  * written under ``data/attack_generation_logs/<record_id>.json``,
  * referenced from the produced payload via ``generation_metadata.log_path``,
  * sufficient for bit-exact replay without re-calling the LLM.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


@dataclass
class GenerationRecord:
    """Full audit trail for one generation attempt.

    All LLM-side fields (prompts, raw responses) are captured so that a
    downstream reviewer can replay the exact decision flow. All oracle-side
    fields are captured so that a reviewer can verify the oracle was
    independent of the WM output.
    """

    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    base_task_id: str = ""
    seed_action: str = ""
    threat_mode: str = "white_box"
    generator_algorithm: str = "rephrase"

    attacker_config: Dict[str, Any] = field(default_factory=dict)
    defender_config: Dict[str, Any] = field(default_factory=dict)
    oracle_config: Dict[str, Any] = field(default_factory=dict)
    # Phase P2: explicit 5-axis attacker-knowledge snapshot. The field is
    # intentionally a plain dict (not a reference to ``AttackerAxes``) so
    # records round-trip through JSON without a dataclass registry.
    # Populated by the generator from the run-time AttackerAxes; payloads
    # materialised from records inherit this provenance.
    attacker_axes: Dict[str, Any] = field(default_factory=dict)

    attacker_prompt: str = ""
    attacker_raw_response: str = ""
    attacker_prompt_hash: str = ""

    variants: List[Dict[str, Any]] = field(default_factory=list)  # VariantScore.to_dict()

    selected_variant_indices: List[int] = field(default_factory=list)
    budget_snapshot: Dict[str, Any] = field(default_factory=dict)

    status: str = "pending"
    reason: Optional[str] = None

    def set_attacker_io(self, prompt: str, raw_response: str) -> None:
        self.attacker_prompt = prompt
        self.attacker_raw_response = raw_response
        self.attacker_prompt_hash = _hash(prompt)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, log_dir: Path) -> Path:
        """Persist the record as ``log_dir/<record_id>.json``."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"{self.record_id}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return out
