"""Transport retry versus strategic Q contract (mocks only, no live API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.budget import BudgetSpec
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.query_semantics import (
    QuerySemanticsError,
    RetryPolicy,
    assert_submission_charges_q,
    assert_transport_retry_does_not_charge_q,
    charges_q,
    classify_event,
    is_api_failure,
    is_transport_retry,
)
from attack_lab.types import AttackProposal, DefenceDecision, InternalDefenceResult
from attack_lab.validator import ConstraintValidator


@dataclass
class CountingDefender:
    name: str = "counting"
    artefact_id: str = "test"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)

    def score_application(self, features: Mapping[str, Any]) -> InternalDefenceResult:
        self.calls += 1
        decision: DefenceDecision = "BLOCK"
        return InternalDefenceResult(
            risk_score=0.9,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


def test_transport_retry_does_not_charge_q() -> None:
    assert charges_q("transport_retry") is False
    assert charges_q("schema_transport_recovery") is False
    assert charges_q("parse_local_generation") is False
    assert charges_q("attack_submission") is True
    assert charges_q("defence_feedback_regeneration") is True
    assert is_transport_retry("timeout") is True
    assert is_transport_retry("transport_error") is True
    assert is_api_failure("timeout") is True
    assert is_api_failure("q_exhausted") is False
    assert (
        classify_event(
            env_step_called=False,
            defence_feedback_received=False,
            transport_error="TimeoutError",
            parse_status="timeout",
        )
        == "transport_retry"
    )


def test_retry_policy_fails_closed_if_transport_would_charge_q() -> None:
    with pytest.raises(QuerySemanticsError):
        RetryPolicy(
            max_transport_retries_per_call=2,
            timeout_seconds=90.0,
            transport_retry_does_not_charge_q=False,
        )


def test_env_step_charges_q_but_unused_ledger_does_not(
    starting_case, governance_policy, tmp_path: Path
) -> None:
    defender = CountingDefender()
    logger = TrajectoryLogger(run_dir=tmp_path / "q", run_id="q")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=defender,
        validator=ConstraintValidator.from_policy(
            governance_policy, enabled_action_keys=("income",)
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=BudgetSpec.development_dummy(q_max=5, m_max=2, label="q_contract"),
    )
    before = env.ledger.q_used
    assert_transport_retry_does_not_charge_q(env.ledger, env.ledger)
    env.step(AttackProposal(changes={"income": 0.2}, raw_command="submit"))
    assert env.ledger.q_used == before + 1
    assert defender.calls == 1
    after_ledger = env.ledger
    # Reconstruct a before snapshot via values.
    class _Snap:
        q_used = before
        scored_defender_queries = 0

    assert_submission_charges_q(_Snap(), after_ledger, expected_charge=1)
    with pytest.raises(QuerySemanticsError):
        assert_transport_retry_does_not_charge_q(_Snap(), after_ledger)
