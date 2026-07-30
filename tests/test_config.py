"""Unit tests for the frozen configuration source of truth."""

from __future__ import annotations

from dataclasses import replace

import pytest

from baf_data.config import FROZEN_CONFIG, SentinelRule

EXPECTED_SHA256 = "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"


def test_expected_hash_matches_audit_record() -> None:
    assert FROZEN_CONFIG.expected_sha256 == EXPECTED_SHA256


def test_schema_has_32_unique_columns() -> None:
    names = FROZEN_CONFIG.raw_column_names
    assert len(names) == 32
    assert len(set(names)) == 32


def test_feature_columns_exclude_target_split_and_frozen_exclusions() -> None:
    features = FROZEN_CONFIG.feature_columns
    assert "fraud_bool" not in features
    assert "month" not in features
    assert "device_fraud_count" not in features
    assert "days_since_request" not in features
    assert "credit_risk_score" not in features
    # 32 columns minus target, split column and three exclusions.
    assert len(features) == 27


def test_feature_columns_preserve_raw_order() -> None:
    raw_order = {name: i for i, name in enumerate(FROZEN_CONFIG.raw_column_names)}
    positions = [raw_order[name] for name in FROZEN_CONFIG.feature_columns]
    assert positions == sorted(positions)


def test_sentinel_rules_cover_exactly_the_six_verified_columns() -> None:
    rules = {rule.column: rule for rule in FROZEN_CONFIG.sentinel_rules}
    assert set(rules) == {
        "prev_address_months_count",
        "current_address_months_count",
        "intended_balcon_amount",
        "bank_months_count",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
    }
    assert rules["intended_balcon_amount"].strategy == "below"
    assert rules["intended_balcon_amount"].value == 0
    for column in set(rules) - {"intended_balcon_amount"}:
        assert rules[column].strategy == "equals"
        assert rules[column].value == -1


def test_no_sentinel_rule_targets_velocity_or_credit_risk() -> None:
    ruled = {rule.column for rule in FROZEN_CONFIG.sentinel_rules}
    assert "velocity_6h" not in ruled
    assert "credit_risk_score" not in ruled


def test_split_months_are_the_frozen_temporal_split() -> None:
    assert FROZEN_CONFIG.split_months == {
        "train": (0, 1, 2, 3, 4, 5),
        "dev": (6,),
        "test": (7,),
    }


def test_binary_features_are_declared_in_the_frozen_config() -> None:
    assert FROZEN_CONFIG.binary_features == (
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "has_other_cards",
        "foreign_request",
        "keep_alive_session",
    )
    features = set(FROZEN_CONFIG.feature_columns)
    kinds = {spec.name: spec.kind for spec in FROZEN_CONFIG.raw_columns}
    for binary in FROZEN_CONFIG.binary_features:
        assert binary in features
        assert kinds[binary] == "integer"


def test_validate_rejects_unknown_binary_feature() -> None:
    bad = replace(FROZEN_CONFIG, binary_features=("no_such_column",))
    with pytest.raises(ValueError, match="not a raw column"):
        bad.validate()


def test_validate_rejects_non_integer_binary_feature() -> None:
    bad = replace(FROZEN_CONFIG, binary_features=("income",))
    with pytest.raises(ValueError, match="integer-kind"):
        bad.validate()


def test_validate_rejects_excluded_column_as_binary_feature() -> None:
    bad = replace(FROZEN_CONFIG, binary_features=("device_fraud_count",))
    with pytest.raises(ValueError, match="not a feature column"):
        bad.validate()


def test_validate_rejects_sentinel_rule_on_unknown_column() -> None:
    bad = replace(
        FROZEN_CONFIG,
        sentinel_rules=(SentinelRule("no_such_column", "equals", -1),),
    )
    with pytest.raises(ValueError, match="unknown column"):
        bad.validate()


def test_validate_rejects_overlapping_split_months() -> None:
    bad = replace(
        FROZEN_CONFIG,
        split_months={"train": (0, 1, 2, 3, 4, 5), "dev": (5, 6), "test": (7,)},
    )
    with pytest.raises(ValueError, match="multiple splits"):
        bad.validate()
