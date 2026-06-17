"""Compute budget enforcement.

Every LLM call inside the attack generator must be routed through
``Budget.consume()`` so the generator cannot silently balloon into a compute
runaway. ``BudgetExhausted`` is raised when either the call count or wall
time cap is crossed; generators should catch it and return a null result
with ``reason="budget"`` rather than bubbling the exception up to the CLI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


class BudgetExhausted(RuntimeError):
    """Raised when either the LLM-call or wall-time budget is crossed."""


@dataclass
class Budget:
    """Bounded compute budget for a single generation attempt.

    Attributes:
        max_llm_calls: Hard cap on LLM calls for this attempt. Non-positive
            values disable the cap.
        wall_time_sec: Hard cap on wall-clock seconds since ``start()``.
            Non-positive values disable the cap.
        calls_used: Running counter of consumed calls (updated by consume()).
        started_at: Unix timestamp set by ``start()``. ``None`` until start.
    """

    max_llm_calls: int = 0
    wall_time_sec: float = 0.0
    calls_used: int = 0
    started_at: Optional[float] = field(default=None)

    def start(self) -> None:
        """Begin the wall-clock timer. Idempotent."""
        if self.started_at is None:
            self.started_at = time.monotonic()

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def remaining_calls(self) -> Optional[int]:
        if self.max_llm_calls <= 0:
            return None
        return max(0, self.max_llm_calls - self.calls_used)

    def consume(self, calls: int = 1) -> None:
        """Consume ``calls`` LLM calls; raise BudgetExhausted if over cap.

        The counter is incremented BEFORE the check so that a caller that
        catches the exception still sees an accurate ``calls_used``.
        """
        if calls < 0:
            raise ValueError(f"calls must be non-negative, got {calls}")
        self.start()
        self.calls_used += calls
        if self.max_llm_calls > 0 and self.calls_used > self.max_llm_calls:
            raise BudgetExhausted(
                f"LLM call budget exhausted: used={self.calls_used}, cap={self.max_llm_calls}"
            )
        if self.wall_time_sec > 0 and self.elapsed() > self.wall_time_sec:
            raise BudgetExhausted(
                f"Wall-time budget exhausted: elapsed={self.elapsed():.2f}s, cap={self.wall_time_sec}s"
            )

    def snapshot(self) -> dict:
        """Return a serializable summary for audit logs."""
        return {
            "max_llm_calls": self.max_llm_calls,
            "wall_time_sec": self.wall_time_sec,
            "calls_used": self.calls_used,
            "elapsed_sec": round(self.elapsed(), 3),
        }
