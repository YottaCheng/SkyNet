"""Budget accounting and match-orchestrator skeleton tests.

Budget protocol: B = (Q, m)
- Q: max candidate submissions per episode
- m: max edit_distance(candidate, original_anchor) per candidate
- Cumulative edit totals never reject or stop an episode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.budget import BudgetSpec
from attack_lab.environment import AttackEnvironment, GuardedDefender
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator, ScriptedAttacker
from attack_lab.types import (
    AttackProposal,
    DefenceDecision,
    InternalDefenceResult,
)
from attack_lab.validator import ConstraintValidator


@dataclass
class CountingDefender:
    name: str = "counting"
    artefact_id: str = "test"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)
    pass_when_income_below: float | None = None

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult:
        self.calls += 1
        decision: DefenceDecision = "BLOCK"
        score = 0.9
        if (
            self.pass_when_income_below is not None
            and float(features["income"]) < self.pass_when_income_below
        ):
            decision = "PASS"
            score = 0.1
        return InternalDefenceResult(
            risk_score=score,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


def _env(
    *,
    starting_case,
    governance_policy,
    tmp_path: Path,
    budget: BudgetSpec,
    enabled: tuple[str, ...] = ("income", "customer_age"),
    defender: CountingDefender | None = None,
) -> tuple[AttackEnvironment, CountingDefender]:
    defender = defender or CountingDefender()
    validator = ConstraintValidator.from_policy(
        governance_policy, enabled_action_keys=enabled
    )
    logger = TrajectoryLogger(run_dir=tmp_path / "budget_run", run_id="budget_run")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=defender,
        validator=validator,
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=budget,
    )
    return env, defender


def test_q_exhausted_blocks_further_submissions(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=2, m_max=100, label="dummy_q")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    env.step(AttackProposal(changes={"income": 0.2}))
    env.step(AttackProposal(changes={"income": 0.3}))
    assert env.done
    assert env.result().stop_reason == "q_exhausted"
    assert defender.calls == 2

    with pytest.raises(RuntimeError, match="already finished"):
        env.step(AttackProposal(changes={"income": 0.4}))
    assert defender.calls == 2


def test_candidate_with_three_edits_rejected_when_m_is_two(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=5, m_max=2, label="dummy_m")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "customer_age", "home_phone_configuration"),
    )
    proxy = governance_policy.fields["phone_home_valid"].resolved_proxy_actions
    anchor_phone = starting_case.features["phone_home_valid"]
    action_name = next(
        name for name, value in proxy.items() if int(value) != int(anchor_phone)
    )
    refused = env.step(
        AttackProposal(
            changes={
                "income": 0.2,
                "customer_age": 40,
                "home_phone_configuration": action_name,
            }
        )
    )
    assert refused.submitted_edit_cost == 3
    assert refused.budget_event is not None
    assert refused.budget_event.budget_rejected is True
    assert refused.budget_event.reject_reason == "m_exceeded"
    assert refused.public_feedback.label == "INVALID"
    assert env.result().stop_reason == "m_exceeded"
    assert defender.calls == 0
    assert env.ledger.q_used == 0
    assert env.ledger.total_edits_used == 0


def test_five_candidates_with_two_edits_each_when_q5_m2(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=5, m_max=2, label="dummy_qm")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "keep_alive_session"),
    )
    flipped_kas = 0 if int(starting_case.features["keep_alive_session"]) == 1 else 1
    for index in range(5):
        record = env.step(
            AttackProposal(
                changes={
                    "income": 0.11 + 0.01 * index,
                    "keep_alive_session": flipped_kas,
                }
            )
        )
        assert record.budget_event is not None
        assert record.budget_event.budget_rejected is False
        assert record.submitted_edit_cost == 2
        assert record.public_feedback.label == "BLOCK"
    assert env.done
    assert env.result().stop_reason == "q_exhausted"
    assert defender.calls == 5
    assert env.ledger.q_used == 5
    assert env.ledger.edits_per_candidate == [2, 2, 2, 2, 2]
    assert env.ledger.total_edits_used == 10


def test_cumulative_edits_do_not_stop_episode(
    starting_case, governance_policy, tmp_path
) -> None:
    """Under old E=3, three 1-edit candidates would exhaust; under m they proceed."""
    budget = BudgetSpec.development_dummy(q_max=5, m_max=1, label="dummy_no_cumul")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income",),
    )
    for value in (0.11, 0.12, 0.13, 0.14):
        record = env.step(AttackProposal(changes={"income": value}))
        assert record.budget_event is not None
        assert record.budget_event.budget_rejected is False
        assert record.submitted_edit_cost == 1
    assert env.done is False
    assert env.ledger.total_edits_used == 4
    assert env.ledger.q_used == 4
    assert defender.calls == 4
    # Fifth submission still allowed under Q; cumulative edits are reporting-only.
    env.step(AttackProposal(changes={"income": 0.15}))
    assert env.result().stop_reason == "q_exhausted"
    assert env.ledger.total_edits_used == 5


def test_edit_distance_always_measured_from_original_anchor(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=3, m_max=20, label="dummy_anchor")
    env, _defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "customer_age"),
    )
    anchor_age = int(starting_case.features["customer_age"])
    alt_age = 40 if anchor_age != 40 else 50
    first = env.step(
        AttackProposal(changes={"income": 0.2, "customer_age": alt_age})
    )
    assert first.validity.is_valid
    assert first.submitted_edit_cost == 2
    # Episode-static age remains locked != anchor, so later candidates still bill
    # age against the original anchor (not against the previous candidate).
    second = env.step(AttackProposal(changes={"income": 0.25}))
    assert second.budget_event is not None
    assert "customer_age" in second.budget_event.edited_fields
    assert second.submitted_edit_cost == 2
    third = env.step(AttackProposal(changes={"income": 0.3}))
    assert third.submitted_edit_cost == 2
    assert env.ledger.total_edits_used == 6
    # Transition from C2->C3 is only income, but charged distance remains vs anchor.
    assert third.transition_edit_count == 1


def test_same_field_repeated_across_candidates_is_recharged(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=4, m_max=10, label="dummy_repeat")
    env, _defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    for value in (0.11, 0.12, 0.13, 0.14):
        record = env.step(AttackProposal(changes={"income": value}))
        assert record.submitted_edit_cost == 1
    assert env.ledger.total_edits_used == 4
    assert env.ledger.unique_fields_ever_manipulated == {"income"}


def test_invalid_submission_charges_q_and_m_but_not_scored_queries(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=3, m_max=10, label="dummy_invalid")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    record = env.step(AttackProposal(changes={"income": 999.0}))  # out of bounds
    assert record.public_feedback.label == "INVALID"
    assert record.internal_defence is None
    assert defender.calls == 0
    assert env.ledger.q_used == 1
    assert env.ledger.total_edits_used == 1
    assert env.ledger.scored_defender_queries == 0
    assert env.ledger.invalid_submissions == 1


def test_m_zero_rejects_any_positive_edit_without_calling_d1(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=1, m_max=0, label="dummy_zero_m")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    record = env.step(AttackProposal(changes={"income": 0.2}))
    assert record.budget_event.budget_rejected is True
    assert record.budget_event.reject_reason == "m_exceeded"
    assert defender.calls == 0
    assert env.ledger.q_used == 0
    assert env.ledger.total_edits_used == 0


def test_pass_stops_immediately(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=5, m_max=10, label="dummy_pass")
    defender = CountingDefender(pass_when_income_below=0.5)
    env, _d = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
        defender=defender,
    )
    record = env.step(AttackProposal(changes={"income": 0.2}))
    assert record.validity.is_valid
    assert record.success is True
    assert env.done is True
    result = env.result()
    assert result.stop_reason == "success"
    assert result.attempts_to_success == 1
    assert result.scored_defender_queries == 1


def test_four_attackers_receive_independent_identical_budgets(
    starting_case, governance_policy, tmp_path
) -> None:
    shared_budget = BudgetSpec.development_dummy(
        q_max=2, m_max=3, label="dummy_shared_template"
    )
    orchestrator = MatchOrchestrator()
    results = []
    for index in range(4):
        attacker = ScriptedAttacker(
            attacker_id=f"sim_a{index}",
            proposals=(
                AttackProposal(changes={"income": 0.2 + 0.01 * index}),
                AttackProposal(changes={"income": 0.3 + 0.01 * index}),
            ),
        )
        logger = TrajectoryLogger(
            run_dir=tmp_path / f"match_{index}",
            run_id=f"match_{index}",
        )
        logger.run_dir.mkdir(parents=True)
        result = orchestrator.run_episode(
            attacker,
            MatchConfig(
                attacker_id=attacker.attacker_id,
                anchor=starting_case,
                policy=governance_policy,
                budget=shared_budget,
                feedback_policy=FeedbackPolicy(mode="label_only"),
                defender=CountingDefender(),
                seed=100 + index,
                enabled_action_keys=("income",),
                logger=logger,
            ),
        )
        results.append(result)

    assert len(results) == 4
    for result in results:
        assert result.budget_spec["q_max"] == 2
        assert result.budget_spec["m_max"] == 3
        assert result.budget_spec["e_max"] == 3  # deprecated mirror
        assert result.q_used == 2
        assert result.total_edits_used == 2
        assert result.e_used == 2
        assert result.stop_reason == "q_exhausted"
    assert {result.q_used for result in results} == {2}


def test_environment_defender_cannot_be_called_directly(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=2, m_max=5, label="dummy_guard")
    env, inner = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    assert isinstance(env.defender, GuardedDefender)
    with pytest.raises(RuntimeError, match="only permitted through AttackEnvironment"):
        env.defender.score_application(starting_case.features)
    assert inner.calls == 0
    env.step(AttackProposal(changes={"income": 0.2}))
    assert inner.calls == 1


@dataclass
class CapabilityProbeAttacker:
    attacker_id: str = "probe"
    received: Any = field(default=None, init=False)
    step_received: Any = field(default=None, init=False)

    def run(self, episode) -> None:
        self.received = episode
        self.step_received = episode.step(AttackProposal(changes={"income": 0.2}))


def test_orchestrator_passes_narrow_capability_without_d1_internals(
    starting_case, governance_policy, tmp_path
) -> None:
    attacker = CapabilityProbeAttacker()
    logger = TrajectoryLogger(run_dir=tmp_path / "capability", run_id="capability")
    logger.run_dir.mkdir()
    result = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id=attacker.attacker_id,
            anchor=starting_case,
            policy=governance_policy,
            budget=BudgetSpec.development_dummy(
                q_max=1, m_max=2, label="capability_test"
            ),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=CountingDefender(),
            seed=7,
            enabled_action_keys=("income",),
            logger=logger,
        ),
    )

    episode = attacker.received
    assert episode is not None
    for forbidden in (
        "defender",
        "threshold",
        "model",
        "feature_importance",
        "researcher_diagnostics",
        "logger",
        "result",
    ):
        assert not hasattr(episode, forbidden)
    for forbidden in ("label", "initial_score", "initial_decision"):
        assert not hasattr(episode.starting_case, forbidden)
    assert "fraud_bool" not in episode.starting_case.features
    assert not hasattr(attacker.step_received, "internal_defence")
    assert attacker.step_received.public_feedback.label == "BLOCK"
    assert result.scored_defender_queries == 1


def test_month7_not_selected_by_orchestrator_config(
    starting_case, governance_policy, tmp_path
) -> None:
    assert starting_case.data_split == "dev_month6"
    budget = BudgetSpec.development_dummy(q_max=1, m_max=5, label="dummy_split")
    logger = TrajectoryLogger(run_dir=tmp_path / "split", run_id="split")
    logger.run_dir.mkdir()
    result = MatchOrchestrator().run_episode(
        ScriptedAttacker(
            attacker_id="sim",
            proposals=(AttackProposal(changes={"income": 0.2}),),
        ),
        MatchConfig(
            attacker_id="sim",
            anchor=starting_case,
            policy=governance_policy,
            budget=budget,
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=CountingDefender(),
            seed=0,
            enabled_action_keys=("income",),
            logger=logger,
        ),
    )
    assert result.anchor_id == starting_case.case_id
    assert "month7" not in result.to_dict()


def test_legacy_e_max_alias_maps_to_m_max() -> None:
    spec = BudgetSpec.development_dummy(q_max=3, e_max=2, label="legacy_alias")
    assert spec.m_max == 2
    assert spec.e_max == 2
