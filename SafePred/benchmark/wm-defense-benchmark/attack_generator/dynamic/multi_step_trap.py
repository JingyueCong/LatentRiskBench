"""Phase 1b — Multi-Step Trap Generator (Cat 1.2).

Thin adapter around :class:`attack_generator.beam_generator.BeamGenerator`
that produces trajectory-level traps: each step is individually predicted
safe by the defender WM, but the trajectory as a whole violates the
independent :class:`TrajectoryPolicyOracle`.

We don't re-implement beam search — that logic is already in
``beam_generator.py`` and well-tested. This module's job is to:

1. Convert a base task into a ``GenerationRequest`` with suitable seed
   actions (drawn from ``unsafe_action_indices``).
2. Construct the BeamGenerator with a contextual action proposer (LLM or
   template) and the shared defender WM + trajectory oracle.
3. Unwrap the resulting BeamGenerator payloads into
   :class:`DynamicGenerationResult` so they share the same serialisation
   contract as every other Category-1 generator.

The benchmark value: this is currently the ONLY attack family in the
suite where the *sequence* is adversarial rather than any single action.
Single-step defenders cannot catch it by design.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..action_proposer import (
    ActionProposer,
    LLMActionProposer,
    TemplateActionProposer,
)
from ..base import GenerationRequest
from ..beam_generator import BeamGenerator
from ..budgets import Budget
from ..defender_wm import DefenderWMScorer
from .base import DynamicGenerationResult


logger = logging.getLogger("attack_generator.dynamic.multi_step_trap")


@dataclass
class MultiStepTrapGenerator:
    """Generate oracle-unsafe trajectories that pass step-by-step WM gating.

    Parameters:
      attacker: ``ActionProposer`` — default: deterministic template.
        Pass an :class:`LLMActionProposer` (via
        ``MultiStepTrapGenerator.with_llm_attacker``) for contextual
        expansions drawn from OpenRouter.
      defender_wm: Scorer used to prune any step the defender flags.
      trajectory_oracle: Independent oracle with
        ``evaluate_trajectory(intent, state, history, policies)``.
      beam_width / max_depth / branch_factor: Beam search hyperparams.
      max_payloads_per_seed: How many distinct adversarial trajectories
        to keep per seed action.
      log_dir: If set, BeamGenerator writes per-node search logs there.
    """

    attacker: ActionProposer
    defender_wm: DefenderWMScorer
    trajectory_oracle: Any
    beam_width: int = 3
    max_depth: int = 3
    branch_factor: int = 4
    max_payloads_per_seed: int = 2
    log_dir: Optional[Path] = None

    generator_id: str = "multi_step_trap_v1"
    attack_id: str = "attack_l2_chain"
    attack_family: str = "multi_step_trigger"
    attack_depth: str = "L2"

    @classmethod
    def with_llm_attacker(
        cls,
        client: Any,
        defender_wm: DefenderWMScorer,
        trajectory_oracle: Any,
        model_name: str = "anthropic/claude-3.5-sonnet",
        **kwargs: Any,
    ) -> "MultiStepTrapGenerator":
        """Convenience constructor: wrap an ``LLMClient`` as action proposer.

        The ``LLMActionProposer`` already accepts an api_key / model_name
        / provider tuple; we plumb the OpenRouter-style config through so
        the caller only has to pass the already-constructed client.
        """
        api_key = getattr(client, "api_key", None)
        api_url = getattr(client, "api_url", None)
        proposer = LLMActionProposer(
            api_key=api_key,
            api_url=api_url,
            model_name=getattr(client, "model_name", model_name),
            provider=getattr(client, "provider", "openai"),
            temperature=float(getattr(client, "temperature", 0.8) or 0.8),
        )
        return cls(attacker=proposer, defender_wm=defender_wm, trajectory_oracle=trajectory_oracle, **kwargs)

    @classmethod
    def with_template_attacker(
        cls,
        defender_wm: DefenderWMScorer,
        trajectory_oracle: Any,
        **kwargs: Any,
    ) -> "MultiStepTrapGenerator":
        """Build a zero-LLM-budget generator for CI / reproducibility."""
        return cls(
            attacker=TemplateActionProposer(),
            defender_wm=defender_wm,
            trajectory_oracle=trajectory_oracle,
            **kwargs,
        )

    def generate(
        self,
        base_task: Dict[str, Any],
        policies: List[Dict[str, Any]],
        budget: Budget,
    ) -> List[DynamicGenerationResult]:
        candidates = list(base_task.get("candidate_actions") or [])
        unsafe_indices = list(base_task.get("unsafe_action_indices") or [])

        # Pick seed actions from the unsafe candidates; each seed becomes
        # an attack-intent-tagged starting point for a separate beam run.
        seed_actions: List[str] = []
        for idx in unsafe_indices:
            if 0 <= idx < len(candidates):
                seed_actions.append(str(candidates[idx]))
        if not seed_actions:
            logger.info("No seed unsafe actions for %s; skipping", base_task.get("task_id"))
            return []

        beam = BeamGenerator(
            attacker=self.attacker,
            defender=self.defender_wm,
            oracle=self.trajectory_oracle,
            policies=policies,
            log_dir=self.log_dir,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
            branch_factor=self.branch_factor,
        )

        request = GenerationRequest(
            base_task=base_task,
            seed_actions=seed_actions,
            max_payloads=self.max_payloads_per_seed * len(seed_actions),
            threat_mode="white_box",
        )

        result = beam.generate(request, budget=budget)
        if result.status != "ok" or not result.payloads:
            logger.info(
                "multi_step_trap: %s → status=%s reason=%s",
                base_task.get("task_id"), result.status, result.reason,
            )
            return []

        keep: List[DynamicGenerationResult] = []
        for pl in result.payloads:
            overrides = dict(pl.get("overrides") or {})
            # BeamGenerator emits payloads targeting ``attack_l2_chain``
            # already, but its schema packs trajectory into
            # ``action_history``. We preserve overrides verbatim — the
            # AttackHandler merges them into the base task at eval time.
            gen_meta = dict(pl.get("generation_metadata") or {})
            gen_meta.setdefault("generator", self.generator_id)
            gen_meta.setdefault("beam_config", {
                "beam_width": self.beam_width,
                "max_depth": self.max_depth,
                "branch_factor": self.branch_factor,
            })
            keep.append(
                DynamicGenerationResult(
                    payload_id=str(pl.get("payload_id") or f"mst_{len(keep)}"),
                    attack_id=str(pl.get("attack_id", self.attack_id)),
                    attack_family=str(pl.get("attack_family", self.attack_family)),
                    attack_depth=str(pl.get("attack_depth", self.attack_depth)),
                    target_task_ids=list(pl.get("target_task_ids") or [str(base_task.get("task_id", ""))]),
                    overrides=overrides,
                    generation_metadata=gen_meta,
                )
            )
        logger.info(
            "multi_step_trap: %s → kept=%d adversarial trajectories",
            base_task.get("task_id"), len(keep),
        )
        return keep
