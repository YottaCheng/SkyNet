"""Per-attacker Q/E budget accounting for the attack laboratory.

Budget meaning (development mechanism only; not a final Q/E freeze):

- ``Q_max``: maximum number of candidates an attacker may submit against one
  anchor.
- ``E_max``: cumulative submitted field-edit budget.  Each submission costs
  the number of attacker-mutable fields that differ from the original anchor
  in that submission's projected candidate.

``E`` is only a cumulative submitted field-edit budget.  It is not a real
crime cost, financial cost, or real workload measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

StopReason = Literal[
    "success",
    "q_exhausted",
    "e_exhausted",
    "attacker_stopped",
    "invalid_environment",
    "policy_error",
    "bypass_pass",  # legacy alias retained in environment mapping
    "budget_exhausted",  # legacy alias
    "attacker_quit",
]


@dataclass(frozen=True)
class BudgetSpec:
    """Immutable per-attacker budget configuration.

    Formal scientific ``Q_max`` / ``E_max`` values are deliberately not frozen
    here.  Tests and development runs must supply explicitly labelled dummy
    budgets.
    """

    q_max: int
    e_max: int
    invalid_charges_q: bool = True
    invalid_charges_proposed_e: bool = True
    stop_on_success: bool = True
    label: str = "development_dummy_budget_not_final_scientific_freeze"

    def __post_init__(self) -> None:
        if self.q_max < 1:
            raise ValueError("q_max must be >= 1.")
        if self.e_max < 0:
            raise ValueError("e_max must be >= 0.")

    @classmethod
    def development_dummy(
        cls,
        *,
        q_max: int,
        e_max: int,
        label: str = "development_dummy_budget_not_final_scientific_freeze",
    ) -> "BudgetSpec":
        """Construct an explicitly labelled non-final dummy budget."""
        return cls(q_max=q_max, e_max=e_max, label=label)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetCheckResult:
    """Outcome of a pre-submission budget gate."""

    allowed: bool
    submitted_edit_cost: int
    edited_fields: tuple[str, ...]
    transition_edit_count: int
    transition_fields: tuple[str, ...]
    reject_reason: str | None = None


@dataclass(frozen=True)
class BudgetEvent:
    """One ledger event for a charged or rejected submission."""

    attempt: int
    submitted_edit_cost: int
    transition_edit_count: int
    edited_fields: tuple[str, ...]
    transition_fields: tuple[str, ...]
    q_charged: int
    e_charged: int
    q_used: int
    e_used: int
    q_remaining: int
    e_remaining: int
    scored_defender_query: bool
    invalid_submission: bool
    budget_rejected: bool
    reject_reason: str | None = None


@dataclass
class BudgetLedger:
    """Mutable per-episode budget ledger owned only by AttackEnvironment."""

    spec: BudgetSpec
    q_used: int = 0
    e_used: int = 0
    scored_defender_queries: int = 0
    invalid_submissions: int = 0
    unique_fields_ever_manipulated: set[str] = field(default_factory=set)
    events: list[BudgetEvent] = field(default_factory=list)
    _previous_candidate: dict[str, Any] | None = field(default=None, init=False)

    @property
    def q_remaining(self) -> int:
        return max(0, self.spec.q_max - self.q_used)

    @property
    def e_remaining(self) -> int:
        return max(0, self.spec.e_max - self.e_used)

    def snapshot(self) -> dict[str, Any]:
        return {
            "budget_spec": self.spec.to_dict(),
            "q_used": self.q_used,
            "e_used": self.e_used,
            "q_remaining": self.q_remaining,
            "e_remaining": self.e_remaining,
            "scored_defender_queries": self.scored_defender_queries,
            "invalid_submissions": self.invalid_submissions,
            "unique_fields_ever_manipulated": sorted(
                self.unique_fields_ever_manipulated
            ),
            "events": [asdict(event) for event in self.events],
        }

    def precheck(
        self,
        *,
        submitted_edit_cost: int,
        edited_fields: tuple[str, ...],
        transition_edit_count: int,
        transition_fields: tuple[str, ...],
    ) -> BudgetCheckResult:
        """Fail-closed gate before validator/D1.  Does not mutate the ledger."""
        if self.q_remaining < 1:
            return BudgetCheckResult(
                allowed=False,
                submitted_edit_cost=submitted_edit_cost,
                edited_fields=edited_fields,
                transition_edit_count=transition_edit_count,
                transition_fields=transition_fields,
                reject_reason="q_exhausted",
            )
        if submitted_edit_cost > self.e_remaining:
            return BudgetCheckResult(
                allowed=False,
                submitted_edit_cost=submitted_edit_cost,
                edited_fields=edited_fields,
                transition_edit_count=transition_edit_count,
                transition_fields=transition_fields,
                reject_reason="e_exhausted",
            )
        return BudgetCheckResult(
            allowed=True,
            submitted_edit_cost=submitted_edit_cost,
            edited_fields=edited_fields,
            transition_edit_count=transition_edit_count,
            transition_fields=transition_fields,
            reject_reason=None,
        )

    def charge_submission(
        self,
        *,
        attempt: int,
        check: BudgetCheckResult,
        is_valid: bool,
        scored: bool,
    ) -> BudgetEvent:
        """Apply default Q/E charging rules after a successful precheck."""
        if not check.allowed:
            raise RuntimeError("Cannot charge a budget-rejected submission.")

        if is_valid:
            q_charge = 1
            e_charge = check.submitted_edit_cost
        else:
            q_charge = 1 if self.spec.invalid_charges_q else 0
            e_charge = (
                check.submitted_edit_cost
                if self.spec.invalid_charges_proposed_e
                else 0
            )
            self.invalid_submissions += 1

        if q_charge > self.q_remaining or e_charge > self.e_remaining:
            # Defensive fail-closed: should be unreachable after precheck.
            raise RuntimeError("Budget charge would exceed remaining allowance.")

        self.q_used += q_charge
        self.e_used += e_charge
        if scored:
            if not is_valid:
                raise RuntimeError(
                    "Invalid submissions must not increment scored_defender_queries."
                )
            self.scored_defender_queries += 1
        self.unique_fields_ever_manipulated.update(check.edited_fields)

        event = BudgetEvent(
            attempt=attempt,
            submitted_edit_cost=check.submitted_edit_cost,
            transition_edit_count=check.transition_edit_count,
            edited_fields=check.edited_fields,
            transition_fields=check.transition_fields,
            q_charged=q_charge,
            e_charged=e_charge,
            q_used=self.q_used,
            e_used=self.e_used,
            q_remaining=self.q_remaining,
            e_remaining=self.e_remaining,
            scored_defender_query=scored,
            invalid_submission=not is_valid,
            budget_rejected=False,
            reject_reason=None,
        )
        self.events.append(event)
        return event

    def record_budget_rejection(
        self,
        *,
        attempt: int,
        check: BudgetCheckResult,
    ) -> BudgetEvent:
        """Record a refused submission that did not charge Q/E or call D1."""
        event = BudgetEvent(
            attempt=attempt,
            submitted_edit_cost=check.submitted_edit_cost,
            transition_edit_count=check.transition_edit_count,
            edited_fields=check.edited_fields,
            transition_fields=check.transition_fields,
            q_charged=0,
            e_charged=0,
            q_used=self.q_used,
            e_used=self.e_used,
            q_remaining=self.q_remaining,
            e_remaining=self.e_remaining,
            scored_defender_query=False,
            invalid_submission=False,
            budget_rejected=True,
            reject_reason=check.reject_reason,
        )
        self.events.append(event)
        return event

    def note_candidate(self, candidate: Mapping[str, Any]) -> None:
        self._previous_candidate = dict(candidate)

    @property
    def previous_candidate(self) -> Mapping[str, Any] | None:
        return self._previous_candidate


def values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        import pandas as pd

        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


def compute_edit_metrics(
    *,
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mutable_feature_names: tuple[str, ...],
    previous_candidate: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], int, tuple[str, ...], int]:
    """Compare a projected candidate with the anchor and previous candidate.

    ``submitted_edit_cost`` counts mutable fields differing from the original
    anchor.  The same field differing across successive submissions is charged
    on every submission.
    """
    edited = tuple(
        sorted(
            name
            for name in mutable_feature_names
            if name in candidate
            and name in anchor
            and not values_equal(candidate[name], anchor[name])
        )
    )
    prior = previous_candidate if previous_candidate is not None else anchor
    transition = tuple(
        sorted(
            name
            for name in mutable_feature_names
            if name in candidate
            and name in prior
            and not values_equal(candidate[name], prior[name])
        )
    )
    return edited, len(edited), transition, len(transition)
