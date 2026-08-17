"""Tests for identity-composition-proxy constraint profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
from attack_lab.budget import AttackBudget
from attack_lab.constraint_profile import IdentityCompositionProfile
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import (
    AttackProposal,
    DefenceDecision,
    InternalDefenceResult,
)
from attack_lab.validator import ConstraintValidator


@dataclass
class BlockDefender:
    name: str = "block"
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


@pytest.fixture()
def profile() -> IdentityCompositionProfile:
    return IdentityCompositionProfile.load()


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    ).get_pool(starting_case.case_id)


def test_profile_loads_and_fingerprints(profile: IdentityCompositionProfile) -> None:
    assert profile.profile_version == "identity-composition-proxy-v1.0.0"
    assert profile.inherits_governance_version == "attack-governance-v2.0.0"
    assert len(profile.profile_fingerprint) == 64
    composite = profile.composite_experiment_fingerprint(
        governance_fingerprint="abc"
    )
    assert len(composite) == 64
    public = profile.public_view()
    assert "d1_risk_score" in public["explicitly_hidden"]
    assert "threshold" not in public["episode_state"]


def test_one_persona_one_contact_legal(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "email_is_free"],
        candidate_features={
            "employment_status": "Self-employed",
            "email_is_free": 1,
        },
    )
    assert check.is_allowed
    assert check.persona_edited == ("employment_status",)
    assert check.contact_edited == ("email_is_free",)


def test_two_persona_illegal(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "customer_age"]
    )
    assert not check.is_allowed


def test_two_contact_illegal(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(["email_is_free", "phone_home_valid"])
    assert not check.is_allowed


def test_persona_plus_income_illegal(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(["employment_status", "income"])
    assert not check.is_allowed
    assert check.other_edited == ("income",)


def test_one_edit_illegal(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(["employment_status"])
    assert not check.is_allowed


def test_three_edits_illegal(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "email_is_free", "income"]
    )
    assert not check.is_allowed


def test_persona_field_locked_after_first(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["housing_status", "email_is_free"],
        candidate_features={"housing_status": "R", "email_is_free": 1},
        persona_locked=True,
        locked_persona_field="employment_status",
        locked_persona_value="Self-employed",
    )
    assert not check.is_allowed


def test_persona_value_locked_after_first(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "email_is_free"],
        candidate_features={
            "employment_status": "Other",
            "email_is_free": 1,
        },
        persona_locked=True,
        locked_persona_field="employment_status",
        locked_persona_value="Self-employed",
    )
    assert not check.is_allowed


def test_contact_field_may_change_later(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "phone_mobile_valid"],
        candidate_features={
            "employment_status": "Self-employed",
            "phone_mobile_valid": 0,
        },
        persona_locked=True,
        locked_persona_field="employment_status",
        locked_persona_value="Self-employed",
    )
    assert check.is_allowed


def test_contact_value_may_change_later(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "email_is_free"],
        candidate_features={
            "employment_status": "Self-employed",
            "email_is_free": 0,
        },
        persona_locked=True,
        locked_persona_field="employment_status",
        locked_persona_value="Self-employed",
    )
    assert check.is_allowed


def test_forbidden_still_rejected(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "email_is_free"],
        forbidden_fields=("employment_status",),
    )
    assert not check.is_allowed


def test_readonly_still_rejected(profile: IdentityCompositionProfile) -> None:
    check = profile.check_edited_features(
        ["employment_status", "bank_months_count"],
        read_only_fields=("bank_months_count",),
    )
    assert not check.is_allowed


def test_budget_still_via_interface(
    starting_case, governance_policy, reference_pool, profile, tmp_path
) -> None:
    budget = AttackBudget(q_max=5, m_max=2)
    logger = TrajectoryLogger(run_dir=tmp_path / "e", run_id="e")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=BlockDefender(),
        validator=ConstraintValidator.from_policy(
            governance_policy,
            reference_pool=reference_pool,
            require_reference_provenance=True,
            enabled_action_keys=tuple(
                key
                for key in governance_policy.available_action_keys
                if (rule := governance_policy.field_for_action(key)) is not None
                and rule.feature
                in {
                    "employment_status",
                    "customer_age",
                    "email_is_free",
                    "phone_home_valid",
                    "phone_mobile_valid",
                    "name_email_similarity",
                }
            ),
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=budget.to_budget_spec(),
        constraint_profile=profile,
        read_only_context_fields=reference_pool.read_only_context_fields,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=7,
        constraint_profile=profile,
    )
    attacker.run(env)
    assert env.budget.m_max == 2
    assert env.budget.q_max == 5
    for step in env.result().steps:
        assert step.submitted_edit_cost == 2


def test_a0_enum_only_profile_legal(
    starting_case, governance_policy, reference_pool, profile, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)
    logger = TrajectoryLogger(run_dir=tmp_path / "a0", run_id="a0")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    enabled = tuple(
        key
        for key in governance_policy.available_action_keys
        if (rule := governance_policy.field_for_action(key)) is not None
        and rule.feature
        in {
            "employment_status",
            "customer_age",
            "housing_status",
            "email_is_free",
            "phone_home_valid",
            "phone_mobile_valid",
            "name_email_similarity",
        }
    )
    attacker = ConstrainedRandomAttacker(
        seed=11,
        reference_pool=reference_pool,
        m_max=2,
        constraint_profile=profile,
        max_local_resamples=0,
    )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a0",
            anchor=starting_case,
            policy=governance_policy,
            budget=budget.to_budget_spec(),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=BlockDefender(),
            seed=11,
            enabled_action_keys=enabled,
            logger=logger,
            reference_pool=reference_pool,
            constraint_profile=profile,
        ),
    )
    # With the composition profile enabled on a rich action set, A0 must be
    # able to freeze at least one legal candidate (random or enum-fallback).
    assert match.q_used >= 1, match.stop_reason
    for step in match.trajectory:
        assert step.submitted_edit_cost == 2
        edited = set(step.research_meta.get("edited_fields") or [])
        # Fallback meta uses edited_fields feature names.
        if edited:
            persona = edited & set(profile.persona_profile_fields)
            contact = edited & set(profile.contact_identity_fields)
            assert len(persona) == 1
            assert len(contact) == 1


def test_a2_enum_only_profile_legal(
    starting_case, governance_policy, reference_pool, profile, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)
    logger = TrajectoryLogger(run_dir=tmp_path / "a2", run_id="a2")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    enabled = tuple(
        key
        for key in governance_policy.available_action_keys
        if (rule := governance_policy.field_for_action(key)) is not None
        and rule.feature
        in {
            "employment_status",
            "customer_age",
            "email_is_free",
            "phone_home_valid",
            "phone_mobile_valid",
            "name_email_similarity",
        }
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=13,
        constraint_profile=profile,
    )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a2",
            anchor=starting_case,
            policy=governance_policy,
            budget=budget.to_budget_spec(),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=BlockDefender(),
            seed=13,
            enabled_action_keys=enabled,
            logger=logger,
            reference_pool=reference_pool,
            constraint_profile=profile,
        ),
    )
    hashes = []
    for step in match.trajectory:
        assert step.submitted_edit_cost == 2
        hashes.append(step.research_meta.get("candidate_hash"))
        meta = dict(step.research_meta or {})
        assert "d1_risk_score" not in meta
        assert "threshold" not in meta
        assert meta.get("hidden_from_attacker")
    assert len(hashes) == len(set(hashes))


def test_env_rejects_profile_violation_without_d1(
    starting_case, governance_policy, reference_pool, profile, tmp_path
) -> None:
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=BlockDefender(),
        validator=ConstraintValidator.from_policy(
            governance_policy,
            reference_pool=reference_pool,
            # Literal path: this test asserts identity-composition rejection,
            # not K-pool provenance (A1/A3-compatible default).
            require_reference_provenance=False,
            enabled_action_keys=(
                "employment_status",
                "email_is_free",
                "income",
            ),
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=AttackBudget(q_max=3, m_max=2).to_budget_spec(),
        constraint_profile=profile,
        read_only_context_fields=reference_pool.read_only_context_fields,
    )
    # Illegal: persona + income
    step = env.step(
        AttackProposal(changes={"employment_status": "CA", "income": 0.5})
    )
    assert step.public_feedback.label == "INVALID"
    assert any("Profile rejected" in err for err in step.validity.errors)
    inner = getattr(env.defender, "_inner", env.defender)
    assert getattr(inner, "calls", 0) == 0


def test_no_d1_leak_in_profile_public_view(profile: IdentityCompositionProfile) -> None:
    public = profile.public_view(q_max=5, m_max=2, queries_remaining=5)
    hidden = set(public["explicitly_hidden"])
    assert "d1_risk_score" in hidden
    assert "d1_threshold" in hidden
    assert "shap_or_feature_importance" in hidden
    assert "risk_score" not in public["episode_state"]
    assert "threshold" not in public["episode_state"]
