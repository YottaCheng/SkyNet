"""Typed data structures for the attack laboratory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from attack_lab.budget import BudgetEvent

PublicLabel = Literal["PASS", "BLOCK", "INVALID"]
DefenceDecision = Literal["PASS", "BLOCK"]


@dataclass(frozen=True)
class AttackProposal:
    """Attacker-proposed field changes for one attempt."""

    changes: Mapping[str, Any]
    raw_command: str = ""


@dataclass(frozen=True)
class InternalDefenceResult:
    """Researcher-only defence outcome; not public by default."""

    risk_score: float
    threshold: float
    decision: DefenceDecision
    runtime_ms: float
    defender_name: str
    artefact_id: str


@dataclass(frozen=True)
class PublicFeedback:
    """Attacker-visible feedback only."""

    label: PublicLabel
    message: str
    attempt: int
    remaining_attempts: int
    q_remaining: int | None = None
    e_remaining: int | None = None


@dataclass(frozen=True)
class Observation:
    """Attacker-visible state for the current attempt."""

    case_id: str
    attempt: int
    max_attempts: int
    visible_fields: dict[str, Any]
    mutable_fields: tuple[str, ...]
    proxy_actions: dict[str, tuple[str, ...]]
    feedback_mode: str
    instructions: str
    remaining_attempts: int
    q_remaining: int
    e_remaining: int
    last_feedback: PublicFeedback | None = None


@dataclass(frozen=True)
class ValidityResult:
    """Outcome of constraint validation before defence scoring."""

    is_valid: bool
    errors: tuple[str, ...] = ()
    candidate_features: dict[str, Any] | None = None


@dataclass(frozen=True)
class StepRecord:
    """Full internal evidence for one attempt."""

    attempt: int
    proposed_changes: dict[str, Any]
    validity: ValidityResult
    candidate_case_id: str
    internal_defence: InternalDefenceResult | None
    public_feedback: PublicFeedback
    success: bool
    elapsed_ms: float
    budget_event: BudgetEvent | None = None
    submitted_edit_cost: int = 0
    transition_edit_count: int = 0


@dataclass(frozen=True)
class EpisodeResult:
    """Terminal summary of one attack episode."""

    case_id: str
    success: bool
    attempts_used: int
    max_attempts: int
    stop_reason: str
    steps: tuple[StepRecord, ...] = field(default_factory=tuple)
    q_used: int = 0
    e_used: int = 0
    scored_defender_queries: int = 0
    invalid_submissions: int = 0
    unique_fields_ever_manipulated: tuple[str, ...] = ()
    attempts_to_success: int | None = None
    budget_spec: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "success": self.success,
            "attempts_used": self.attempts_used,
            "max_attempts": self.max_attempts,
            "stop_reason": self.stop_reason,
            "q_used": self.q_used,
            "e_used": self.e_used,
            "scored_defender_queries": self.scored_defender_queries,
            "attempts_to_success": self.attempts_to_success,
        }


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses / mappings to JSON-serialisable structures."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, float):
        # Preserve NaN as null for JSON logs.
        if obj != obj:  # noqa: PLR0124
            return None
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    # numpy scalars
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)
