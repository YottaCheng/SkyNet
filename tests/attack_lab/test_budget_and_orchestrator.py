"""Budget accounting and match-orchestrator skeleton tests."""

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
    budget = BudgetSpec.development_dummy(q_max=2, e_max=100, label="dummy_q")
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


def test_e_exhausted_blocks_without_calling_d1(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=5, e_max=1, label="dummy_e")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    # First submission: one edited field, consumes E=1.
    first = env.step(AttackProposal(changes={"income": 0.2}))
    assert first.public_feedback.label == "BLOCK"
    assert first.submitted_edit_cost == 1
    assert defender.calls == 1
    # Second positive-cost submission must be refused before D1.
    refused = env.step(AttackProposal(changes={"income": 0.3}))
    assert refused.budget_event is not None
    assert refused.budget_event.budget_rejected is True
    assert refused.public_feedback.label == "INVALID"
    assert env.result().stop_reason == "e_exhausted"
    assert defender.calls == 1


def test_same_field_repeated_across_candidates_is_recharged(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=4, e_max=10, label="dummy_repeat")
    env, _defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    for value in (0.11, 0.12, 0.13, 0.14):
        record = env.step(AttackProposal(changes={"income": value}))
        assert record.submitted_edit_cost == 1
    assert env.ledger.e_used == 4
    assert env.ledger.unique_fields_ever_manipulated == {"income"}


def test_episode_static_fields_charge_relative_to_anchor_each_submission(
    starting_case, governance_policy, tmp_path
) -> None:
    # phone_home_valid is episode-locked; attackers use the abstract proxy action.
    budget = BudgetSpec.development_dummy(q_max=3, e_max=20, label="dummy_lock")
    env, _defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "home_phone_configuration"),
    )
    proxy = governance_policy.fields["phone_home_valid"].resolved_proxy_actions
    anchor_phone = starting_case.features["phone_home_valid"]
    # Choose a proxy that resolves to the opposite of the anchor when possible.
    action_name = next(
        name for name, value in proxy.items() if int(value) != int(anchor_phone)
    )
    first = env.step(
        AttackProposal(
            changes={"income": 0.2, "home_phone_configuration": action_name}
        )
    )
    assert first.validity.is_valid
    assert first.submitted_edit_cost == 2
    # Later submissions keep the locked phone value != anchor, so phone still bills.
    second = env.step(AttackProposal(changes={"income": 0.25}))
    assert second.budget_event is not None
    assert "phone_home_valid" in second.budget_event.edited_fields
    assert second.submitted_edit_cost == 2
    third = env.step(AttackProposal(changes={"income": 0.3}))
    assert third.submitted_edit_cost == 2
    assert env.ledger.e_used == 6


def test_invalid_submission_charges_q_and_e_but_not_scored_queries(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=3, e_max=10, label="dummy_invalid")
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
    assert env.ledger.e_used == 1
    assert env.ledger.scored_defender_queries == 0
    assert env.ledger.invalid_submissions == 1


def test_budget_shortfall_never_calls_d1(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=1, e_max=0, label="dummy_zero_e")
    env, defender = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=budget,
    )
    # edit_cost=1 > e_remaining=0 -> reject before D1.
    record = env.step(AttackProposal(changes={"income": 0.2}))
    assert record.budget_event.budget_rejected is True
    assert defender.calls == 0
    assert env.ledger.q_used == 0
    assert env.ledger.e_used == 0


def test_pass_stops_immediately(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=5, e_max=10, label="dummy_pass")
    # Use an in-domain income value under the compiled train support.
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
        q_max=2, e_max=3, label="dummy_shared_template"
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
        assert result.budget_spec["e_max"] == 3
        assert result.q_used == 2
        assert result.e_used == 2
        assert result.stop_reason == "q_exhausted"
    # Budgets are independent: one attacker's spend does not reduce another's.
    assert {result.q_used for result in results} == {2}


def test_environment_defender_cannot_be_called_directly(
    starting_case, governance_policy, tmp_path
) -> None:
    budget = BudgetSpec.development_dummy(q_max=2, e_max=5, label="dummy_guard")
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


def test_month7_not_selected_by_orchestrator_config(
    starting_case, governance_policy, tmp_path
) -> None:
    # Guard: orchestrator uses the provided month-6-shaped anchor only.
    assert starting_case.data_split == "dev_month6"
    budget = BudgetSpec.development_dummy(q_max=1, e_max=5, label="dummy_split")
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
