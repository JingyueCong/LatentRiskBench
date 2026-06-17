"""Structured trajectory-state schema for state-conditioned (SC) construction.

This is the machine-checkable layer underneath the prose ``a11y_tree_txt``. It names
the policy-relevant variables that a single observation/action cannot reveal, so that
(a) policies can be bound to fields, (b) targeted injection can flip exactly one field,
and (c) path-invariance can compare two states for equality.

Fields are grouped by the four SC mechanisms:
  * goal           -> goal-shift dependence
  * staged_payload -> state accumulation
  * destination    -> destination dependence
  * authorization  -> authorization dependence
  * audit          -> (state-conditioning for log tampering)

``pending_action`` is the abstraction of the terminal action under audit. Its fields are
ACTION-LOCAL (recoverable from the action surface); everything else is STATE-ONLY.
A genuinely state-conditioned item injects risk only at STATE-ONLY paths while holding
the action surface fixed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Controlled vocabularies (enums)
# --------------------------------------------------------------------------- #
GOAL_ALIGNMENT = ("aligned", "drifted", "hijacked")
SENSITIVITY = ("none", "internal", "customer_pii", "secret")
DESTINATION_CLASS = ("none", "internal_approved", "external_unapproved")
TRUST = ("trusted", "unverified", "suspicious")
LOG_STATE = ("intact", "tampered", "disabled")

# Operation classes for the terminal action (ACTION-LOCAL).
TRANSMIT_OPS = ("send", "upload", "share", "post", "sync", "export", "commit")
PERSIST_OPS = ("type", "write", "save", "store")
FOLLOW_OPS = ("click", "navigate", "open", "follow")
LOG_OPS = ("modify_logs", "clear_logs", "disable_logs", "rotate_logs")
ALL_OPS = TRANSMIT_OPS + PERSIST_OPS + FOLLOW_OPS + LOG_OPS + ("read", "grant_access", "other")

# Paths that the action surface CANNOT reveal -- the only legal injection sites for SC risk.
STATE_ONLY_PATHS: Tuple[str, ...] = (
    "goal.alignment",
    "staged_payload.sensitivity",
    "staged_payload.contains_plaintext_secret",
    "destination.class",
    "destination.trust",
    "authorization.granted",
    "authorization.required",
    "audit.has_concealable_prior_actions",
    "audit.log_state",
)
# Paths recoverable from the action surface (visible to an action-local evaluator).
ACTION_LOCAL_PATHS: Tuple[str, ...] = (
    "pending_action.operation",
    "pending_action.surface_text",
)


class StateValidationError(ValueError):
    """Raised when a structured state does not conform to the schema."""


def empty_state() -> Dict[str, Any]:
    """Return a benign, schema-valid structured state with default field values."""
    return {
        "goal": {"original_intent": "", "active_goal": "", "alignment": "aligned"},
        "staged_payload": {"items": [], "sensitivity": "none", "contains_plaintext_secret": False},
        "destination": {"target": "", "class": "none", "trust": "trusted"},
        "authorization": {"required": [], "granted": []},
        "audit": {"has_concealable_prior_actions": False, "log_state": "intact"},
        "pending_action": {"operation": "other", "surface_text": ""},
    }


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise StateValidationError(msg)


def validate_state_struct(s: Dict[str, Any]) -> None:
    """Validate a structured state against the schema; raise StateValidationError on any issue."""
    _require(isinstance(s, dict), f"state_struct must be a dict, got {type(s).__name__}")
    for section in ("goal", "staged_payload", "destination", "authorization", "audit", "pending_action"):
        _require(section in s and isinstance(s[section], dict), f"missing/invalid section: {section!r}")

    _require(s["goal"].get("alignment") in GOAL_ALIGNMENT, f"goal.alignment not in {GOAL_ALIGNMENT}")
    _require(s["staged_payload"].get("sensitivity") in SENSITIVITY, f"staged_payload.sensitivity not in {SENSITIVITY}")
    _require(isinstance(s["staged_payload"].get("contains_plaintext_secret"), bool), "contains_plaintext_secret must be bool")
    _require(isinstance(s["staged_payload"].get("items"), list), "staged_payload.items must be a list")
    _require(s["destination"].get("class") in DESTINATION_CLASS, f"destination.class not in {DESTINATION_CLASS}")
    _require(s["destination"].get("trust") in TRUST, f"destination.trust not in {TRUST}")
    _require(isinstance(s["authorization"].get("required"), list), "authorization.required must be a list")
    _require(isinstance(s["authorization"].get("granted"), list), "authorization.granted must be a list")
    _require(isinstance(s["audit"].get("has_concealable_prior_actions"), bool), "has_concealable_prior_actions must be bool")
    _require(s["audit"].get("log_state") in LOG_STATE, f"audit.log_state not in {LOG_STATE}")
    _require(s["pending_action"].get("operation") in ALL_OPS, f"pending_action.operation not in {ALL_OPS}")


def get_path(s: Dict[str, Any], path: str) -> Any:
    """Read a dotted path (e.g. 'destination.class') from a structured state."""
    cur: Any = s
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def canonical_state(s: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Canonicalize the policy-relevant (STATE-ONLY) fields for equality / path-invariance.

    Two prefixes reach "the same state" iff their canonical_state tuples are equal.
    Free-text targets and surface text are excluded so equality reflects policy-relevant
    values, not surface phrasing.
    """
    def norm(v: Any) -> Any:
        return tuple(sorted(v)) if isinstance(v, list) else v

    return tuple((p, norm(get_path(s, p))) for p in STATE_ONLY_PATHS)
