"""Executable governance, isolation, locking, and proxy-action tests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.budget import BudgetSpec
from attack_lab.cli import main
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.governance import (
    CompiledGovernancePolicy,
    GovernanceError,
    GovernanceLoader,
    PolicyCompiler,
)
from attack_lab.logger import TrajectoryLogger
from attack_lab.types import (
    AttackProposal,
    DefenceDecision,
    InternalDefenceResult,
)
from attack_lab.validator import AttackLabValidationError, ConstraintValidator


@dataclass
class CountingBlockDefender:
    name: str = "counting_block"
    artefact_id: str = "test_only"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult:
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


def _environment(
    *,
    starting_case,
    policy,
    enabled: tuple[str, ...],
    tmp_path: Path,
    max_attempts: int = 4,
) -> tuple[AttackEnvironment, CountingBlockDefender, TrajectoryLogger]:
    defender = CountingBlockDefender()
    validator = ConstraintValidator.from_policy(
        policy, enabled_action_keys=enabled
    )
    logger = TrajectoryLogger(
        run_dir=tmp_path / "governed_episode",
        run_id="governed_episode",
    )
    logger.run_dir.mkdir(parents=True)
    environment = AttackEnvironment(
        starting_case=starting_case,
        defender=defender,
        validator=validator,
        feedback_policy=FeedbackPolicy(mode="label_only"),
        max_attempts=max_attempts,
        logger=logger,
        budget=BudgetSpec.development_dummy(
            q_max=max_attempts,
            e_max=1_000_000,
            label="test_dummy_budget_not_final",
        ),
    )
    return environment, defender, logger


def test_governance_loader_ignores_legacy_status(
    governance_csv: Path, tmp_path: Path
) -> None:
    with governance_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    for row in rows:
        row["status"] = "deliberately_irrelevant_preprocessing_value"
    altered = tmp_path / "altered_status.csv"
    with altered.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    original = GovernanceLoader.load_csv(governance_csv)
    loaded = GovernanceLoader.load_csv(altered)
    assert [
        (row.feature, row.governance_status, row.agent_mutability)
        for row in loaded
    ] == [
        (row.feature, row.governance_status, row.agent_mutability)
        for row in original
    ]
    assert all(not hasattr(row, "status") for row in loaded)


def test_unfrozen_governance_status_blocks_loading(
    governance_csv: Path, tmp_path: Path
) -> None:
    with governance_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    rows[2]["governance_status"] = "working"
    altered = tmp_path / "unfrozen.csv"
    with altered.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(GovernanceError, match="not frozen"):
        GovernanceLoader.load_csv(altered)


def test_compiler_rejects_any_non_training_month(
    governance_csv: Path, synthetic_frame
) -> None:
    rules = GovernanceLoader.load_csv(governance_csv)
    with pytest.raises(GovernanceError, match="months 0-5 only"):
        PolicyCompiler.compile(
            rules,
            synthetic_frame,
            source_path=governance_csv,
        )


def test_attack_cli_fails_closed_without_compiled_policy(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--case-id",
            "900001",
            "--mutable-fields",
            "income",
            "--max-attempts",
            "1",
            "--governance-policy",
            str(tmp_path / "missing.json"),
        ]
    )
    assert exit_code == 2


def test_compiled_domains_are_typed_and_train_supported(
    governance_policy, synthetic_frame
) -> None:
    income = governance_policy.fields["income"]
    payment = governance_policy.fields["payment_type"]
    phone = governance_policy.fields["phone_home_valid"]
    assert income.data_type == "float"
    assert income.lower_bound is not None
    assert income.upper_bound is not None
    assert payment.data_type == "categorical"
    train_values = set(
        synthetic_frame.loc[
            synthetic_frame["month"].between(0, 5), "payment_type"
        ]
    )
    assert set(payment.allowed_values) == train_values
    assert phone.data_type == "binary"
    assert set(phone.allowed_values) == {0, 1}


def test_proxy_mapping_is_reproducible_and_serialisable(
    governance_csv: Path, synthetic_frame, governance_policy, tmp_path: Path
) -> None:
    training = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    second = PolicyCompiler.compile(
        GovernanceLoader.load_csv(governance_csv),
        training,
        source_path=governance_csv,
    )
    assert second.policy_fingerprint == governance_policy.policy_fingerprint
    proxy = governance_policy.fields["name_email_similarity"]
    support = set(training["name_email_similarity"])
    assert set(proxy.resolved_proxy_actions.values()) <= support
    assert proxy.proxy_seed == 20260801

    saved = tmp_path / "compiled_governance.json"
    governance_policy.save(saved)
    restored = CompiledGovernancePolicy.load(saved)
    assert restored.policy_fingerprint == governance_policy.policy_fingerprint
    assert (
        restored.fields["name_email_similarity"].resolved_proxy_actions
        == proxy.resolved_proxy_actions
    )


def test_proxy_rejects_direct_numeric_write_and_accepts_abstract_action(
    baseline_features, governance_policy
) -> None:
    validator = ConstraintValidator.from_policy(
        governance_policy,
        enabled_action_keys=("name_email_alignment",),
    )
    direct = AttackProposal(changes={"name_email_similarity": 0.9})
    direct_locks = validator.prepare_episode_locks(baseline_features, direct)
    direct_result = validator.validate(
        baseline_features,
        direct,
        locked_values=direct_locks.locked_values,
        pre_feedback_errors=direct_locks.errors,
    )
    assert not direct_result.is_valid
    assert all(
        "name_email_similarity" not in error
        for error in direct_result.errors
    )

    abstract = AttackProposal(
        changes={"name_email_alignment": "high_alignment"}
    )
    abstract_locks = validator.prepare_episode_locks(
        baseline_features, abstract
    )
    abstract_result = validator.validate(
        baseline_features,
        abstract,
        locked_values=abstract_locks.locked_values,
        pre_feedback_errors=abstract_locks.errors,
    )
    assert abstract_result.is_valid
    assert abstract_result.candidate_features is not None
    resolved = abstract_result.candidate_features["name_email_similarity"]
    assert resolved == governance_policy.fields[
        "name_email_similarity"
    ].resolved_proxy_actions["high_alignment"]


def test_hidden_scalars_never_enter_observation(
    starting_case, governance_policy, tmp_path: Path
) -> None:
    env, _defender, _logger = _environment(
        starting_case=starting_case,
        policy=governance_policy,
        enabled=("income", "name_email_alignment"),
        tmp_path=tmp_path,
    )
    observation = env.observation()
    assert "zip_count_4w" not in observation.visible_fields
    assert "velocity_6h" not in observation.visible_fields
    assert "name_email_similarity" not in observation.visible_fields
    assert "name_email_alignment" in observation.proxy_actions
    assert observation.proxy_actions["name_email_alignment"] == (
        "low_alignment",
        "typical_alignment",
        "high_alignment",
    )


def test_all_episode_fields_lock_on_first_submission_even_when_omitted(
    starting_case, governance_policy, tmp_path: Path
) -> None:
    env, defender, _logger = _environment(
        starting_case=starting_case,
        policy=governance_policy,
        enabled=("income", "employment_status"),
        tmp_path=tmp_path,
    )
    first = env.step(AttackProposal(changes={"income": 0.2}))
    assert first.public_feedback.label == "BLOCK"
    assert defender.calls == 1
    assert "employment_status" not in env.observation().mutable_fields

    anchor = starting_case.features["employment_status"]
    replacement = "CB" if anchor != "CB" else "CC"
    second = env.step(
        AttackProposal(changes={"employment_status": replacement})
    )
    assert second.public_feedback.label == "INVALID"
    assert defender.calls == 1
    assert any(
        "cannot change" in error.lower() for error in second.validity.errors
    )


def test_invalid_first_submission_still_locks_before_feedback(
    starting_case, governance_policy, tmp_path: Path
) -> None:
    env, defender, _logger = _environment(
        starting_case=starting_case,
        policy=governance_policy,
        enabled=("employment_status",),
        tmp_path=tmp_path,
    )
    first = env.step(
        AttackProposal(changes={"employment_status": "NOT_SUPPORTED"})
    )
    assert first.public_feedback.label == "INVALID"
    assert defender.calls == 0

    anchor = starting_case.features["employment_status"]
    replacement = "CB" if anchor != "CB" else "CC"
    second = env.step(
        AttackProposal(changes={"employment_status": replacement})
    )
    assert second.public_feedback.label == "INVALID"
    assert defender.calls == 0
    assert any(
        "cannot change" in error.lower() for error in second.validity.errors
    )


EXPECTED_PER_ATTEMPT = (
    "income",
    "intended_balcon_amount",
    "payment_type",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "name_email_similarity",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
)
EXPECTED_EPISODE_STATIC = (
    "customer_age",
    "prev_address_months_count",
    "current_address_months_count",
    "employment_status",
    "housing_status",
)
EXPECTED_FORBIDDEN = (
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "bank_months_count",
    "has_other_cards",
    "device_distinct_emails_8w",
)
EXPECTED_NOT_APPLICABLE = (
    "fraud_bool",
    "month",
    "days_since_request",
    "device_fraud_count",
)
RELEASED_FROM_STATIC = (
    "name_email_similarity",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
)


def test_governance_v2_mutability_partition(governance_policy) -> None:
    assert governance_policy.policy_version == "attack-governance-v2.0.0"
    assert set(governance_policy.per_attempt_fields) == set(EXPECTED_PER_ATTEMPT)
    assert len(governance_policy.per_attempt_fields) == 13
    assert set(governance_policy.episode_static_fields) == set(
        EXPECTED_EPISODE_STATIC
    )
    assert len(governance_policy.episode_static_fields) == 5
    assert set(governance_policy.forbidden_fields) == set(EXPECTED_FORBIDDEN)
    assert len(governance_policy.forbidden_fields) == 10
    assert set(governance_policy.not_applicable_fields) == set(
        EXPECTED_NOT_APPLICABLE
    )
    assert len(governance_policy.not_applicable_fields) == 4
    assert set(governance_policy.action_fields) == set(EXPECTED_PER_ATTEMPT) | set(
        EXPECTED_EPISODE_STATIC
    )
    assert len(governance_policy.action_fields) == 18


def test_released_email_phone_fields_have_no_episode_lock(
    governance_policy,
) -> None:
    for name in RELEASED_FROM_STATIC:
        rule = governance_policy.fields[name]
        assert rule.agent_mutability == "allowed"
        assert rule.is_episode_locked is False
        assert all(
            item.get("type") != "episode_lock_on_first_submission"
            for item in rule.hard_constraints
        )
    assert "name_email_alignment" in governance_policy.proxy_action_keys
    assert "home_phone_configuration" in governance_policy.proxy_action_keys
    assert "mobile_phone_configuration" in governance_policy.proxy_action_keys


def test_five_episode_static_fields_still_lock(
    starting_case, governance_policy, tmp_path: Path
) -> None:
    for field in EXPECTED_EPISODE_STATIC:
        rule = governance_policy.fields[field]
        assert rule.agent_mutability == "allowed_if_episode_locked"
        assert rule.is_episode_locked is True
        assert any(
            item.get("type") == "episode_lock_on_first_submission"
            for item in rule.hard_constraints
        )

    env, defender, _logger = _environment(
        starting_case=starting_case,
        policy=governance_policy,
        enabled=("income", "customer_age", "housing_status"),
        tmp_path=tmp_path,
    )
    first = env.step(AttackProposal(changes={"income": 0.2}))
    assert first.public_feedback.label == "BLOCK"
    assert defender.calls == 1
    assert "customer_age" not in env.observation().mutable_fields
    assert "housing_status" not in env.observation().mutable_fields

    age = starting_case.features["customer_age"]
    alt_age = 40 if int(age) != 40 else 50
    refused = env.step(AttackProposal(changes={"customer_age": alt_age}))
    assert refused.public_feedback.label == "INVALID"
    assert defender.calls == 1


def test_released_fields_can_change_across_submissions(
    starting_case, governance_policy, tmp_path: Path
) -> None:
    env, defender, _logger = _environment(
        starting_case=starting_case,
        policy=governance_policy,
        enabled=("email_is_free", "home_phone_configuration"),
        tmp_path=tmp_path,
        max_attempts=3,
    )
    anchor_email = int(starting_case.features["email_is_free"])
    first_email = 0 if anchor_email == 1 else 1
    second_email = anchor_email  # may return toward anchor on next candidate
    proxy = governance_policy.fields["phone_home_valid"].resolved_proxy_actions
    anchor_phone = int(starting_case.features["phone_home_valid"])
    first_phone = next(
        name for name, value in proxy.items() if int(value) != anchor_phone
    )
    second_phone = next(
        name for name, value in proxy.items() if int(value) == anchor_phone
    )

    first = env.step(
        AttackProposal(
            changes={
                "email_is_free": first_email,
                "home_phone_configuration": first_phone,
            }
        )
    )
    assert first.validity.is_valid
    assert first.public_feedback.label == "BLOCK"
    assert defender.calls == 1
    # Still mutable after first submission under v2 per-attempt semantics.
    assert "email_is_free" in env.observation().mutable_fields
    assert "home_phone_configuration" in env.observation().proxy_actions

    second = env.step(
        AttackProposal(
            changes={
                "email_is_free": second_email,
                "home_phone_configuration": second_phone,
            }
        )
    )
    assert second.validity.is_valid
    assert second.public_feedback.label == "BLOCK"
    assert defender.calls == 2


def test_read_only_context_fields_are_not_actions(governance_policy) -> None:
    for name in ("bank_months_count", "has_other_cards"):
        assert name not in governance_policy.action_fields
        assert governance_policy.fields[name].agent_mutability == "forbidden"
        assert name in governance_policy.forbidden_fields
        assert name not in governance_policy.available_action_keys
    with pytest.raises(AttackLabValidationError, match="not permitted"):
        ConstraintValidator.from_policy(
            governance_policy,
            enabled_action_keys=("income", "bank_months_count"),
        )


def test_forbidden_field_mutation_rejected(
    baseline_features, governance_policy
) -> None:
    validator = ConstraintValidator.from_policy(
        governance_policy,
        enabled_action_keys=("income",),
    )
    proposal = AttackProposal(
        changes={"income": 0.2, "credit_risk_score": 0}
    )
    locks = validator.prepare_episode_locks(baseline_features, proposal)
    result = validator.validate(
        baseline_features,
        proposal,
        locked_values=locks.locked_values,
        pre_feedback_errors=locks.errors,
    )
    assert result.is_valid is False


def test_policy_manifest_logs_proxy_seed_and_public_log_hides_validation_detail(
    starting_case, governance_policy, tmp_path: Path
) -> None:
    env, defender, logger = _environment(
        starting_case=starting_case,
        policy=governance_policy,
        enabled=("name_email_alignment",),
        tmp_path=tmp_path,
    )
    assert logger.governance_manifest_path.is_file()
    manifest = json.loads(
        logger.governance_manifest_path.read_text(encoding="utf-8")
    )
    mapping = manifest["governance"]["proxy_mappings"][
        "name_email_alignment"
    ]
    assert mapping["seed"] == 20260801
    assert mapping["support_split"] == "train_months_0_5"

    record = env.step(
        AttackProposal(changes={"name_email_similarity": 0.9})
    )
    assert record.public_feedback.label == "INVALID"
    assert defender.calls == 0
    public = logger.public_transcript_path.read_text(encoding="utf-8")
    internal = logger.trajectory_path.read_text(encoding="utf-8")
    assert "not permitted by governance" not in public
    assert "not permitted by governance" in internal
