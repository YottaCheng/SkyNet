"""Public attacker feedback policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from attack_lab.types import InternalDefenceResult, PublicFeedback, ValidityResult

FeedbackMode = Literal["label_only"]


@dataclass(frozen=True)
class FeedbackPolicy:
    """Map internal defence evidence to public attacker-visible feedback.

    Additional modes may be added later; only label_only is implemented now.
    """

    mode: FeedbackMode = "label_only"

    def __post_init__(self) -> None:
        if self.mode != "label_only":
            raise ValueError(
                f"Unsupported feedback mode {self.mode!r}; "
                "only 'label_only' is implemented."
            )

    def for_invalid(
        self,
        validity: ValidityResult,
        *,
        attempt: int,
        remaining_attempts: int,
        q_remaining: int | None = None,
        e_remaining: int | None = None,
    ) -> PublicFeedback:
        # Validation detail is researcher-internal.  Returning field-specific
        # errors would create an oracle for hidden-field and constraint probing.
        _ = validity
        message = "INVALID proposal: governance or constraint failure."
        return PublicFeedback(
            label="INVALID",
            message=message,
            attempt=attempt,
            remaining_attempts=remaining_attempts,
            q_remaining=q_remaining,
            e_remaining=e_remaining,
        )

    def for_scored(
        self,
        internal: InternalDefenceResult,
        *,
        attempt: int,
        remaining_attempts: int,
        q_remaining: int | None = None,
        e_remaining: int | None = None,
    ) -> PublicFeedback:
        # Deliberately omit risk_score and threshold from the public object.
        if internal.decision == "PASS":
            return PublicFeedback(
                label="PASS",
                message="Application PASS.",
                attempt=attempt,
                remaining_attempts=remaining_attempts,
                q_remaining=q_remaining,
                e_remaining=e_remaining,
            )
        return PublicFeedback(
            label="BLOCK",
            message="Application BLOCK.",
            attempt=attempt,
            remaining_attempts=remaining_attempts,
            q_remaining=q_remaining,
            e_remaining=e_remaining,
        )

    def for_budget_rejected(
        self,
        *,
        reason: str,
        attempt: int,
        remaining_attempts: int,
        q_remaining: int | None = None,
        e_remaining: int | None = None,
    ) -> PublicFeedback:
        """Public notice when Q/E is insufficient.  No score/threshold leaked."""
        message = f"INVALID proposal: submission refused ({reason})."
        return PublicFeedback(
            label="INVALID",
            message=message,
            attempt=attempt,
            remaining_attempts=remaining_attempts,
            q_remaining=q_remaining,
            e_remaining=e_remaining,
        )
