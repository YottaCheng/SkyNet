"""Constraint validation and output-path protection tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from attack_lab.cases import AttackLabCaseError, assert_month6_only
from attack_lab.paths import (
    ATTACK_LAB_ROOT,
    AttackLabPathError,
    assert_not_protected,
    new_run_directory,
)
from attack_lab.types import AttackProposal
from attack_lab.validator import ConstraintValidator


def test_non_governed_action_rejected(
    baseline_features, validator: ConstraintValidator
) -> None:
    proposal = AttackProposal(changes={"zip_count_4w": 99})
    result = validator.validate(baseline_features, proposal)
    assert not result.is_valid
    assert any("not permitted" in err.lower() for err in result.errors)


def test_malformed_numeric_rejected(
    baseline_features, validator: ConstraintValidator
) -> None:
    proposal = AttackProposal(changes={"income": "not-a-number"})
    result = validator.validate(baseline_features, proposal)
    assert not result.is_valid
    assert any("could not parse" in err.lower() for err in result.errors)


def test_out_of_schema_categorical_rejected(
    baseline_features, governance_policy
) -> None:
    validator = ConstraintValidator.from_policy(
        governance_policy,
        enabled_action_keys=("payment_type",),
    )
    proposal = AttackProposal(changes={"payment_type": "ZZ_NOT_IN_VOCAB"})
    result = validator.validate(baseline_features, proposal)
    assert not result.is_valid
    assert any("train-supported domain" in err.lower() for err in result.errors)


def test_valid_mutable_change_accepted(
    baseline_features, validator: ConstraintValidator
) -> None:
    proposal = AttackProposal(changes={"income": 0.25})
    result = validator.validate(baseline_features, proposal)
    assert result.is_valid
    assert result.candidate_features is not None
    assert result.candidate_features["income"] == 0.25
    assert result.candidate_features["customer_age"] == baseline_features["customer_age"]


def test_month7_cannot_be_selected() -> None:
    for split in ("test", "month7", "month_7", "holdout_test", "dev_month7"):
        with pytest.raises(AttackLabCaseError, match="Month 7"):
            assert_month6_only(split)


def test_existing_output_directories_cannot_be_overwritten(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "attack_lab.paths.ATTACK_LAB_ROOT",
        tmp_path / "attack_lab",
    )
    first = new_run_directory("run_alpha")
    assert first.is_dir()
    with pytest.raises(AttackLabPathError, match="overwrite"):
        new_run_directory("run_alpha")


def test_protected_artefact_trees_refuse_writes() -> None:
    protected = Path(
        "/Users/ziyaoch/ucl/dissertation/05_outputs/xgboost_challenge/probe.txt"
    )
    with pytest.raises(AttackLabPathError, match="protected"):
        assert_not_protected(protected)


def test_attack_lab_root_is_isolated() -> None:
    assert ATTACK_LAB_ROOT.name == "attack_lab"
    assert "xgboost_challenge" not in str(ATTACK_LAB_ROOT)
