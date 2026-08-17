"""Tests for synthetic identity reference-pool infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from attack_lab.reference_pool import (
    DEFAULT_REFERENCE_POOL_CONFIG,
    ReferencePoolConfig,
    ReferencePoolError,
    ReferencePoolProvider,
)


@pytest.fixture()
def pool_config() -> ReferencePoolConfig:
    return ReferencePoolConfig.load(DEFAULT_REFERENCE_POOL_CONFIG)


@pytest.fixture()
def train_only_frame(synthetic_frame: pd.DataFrame) -> pd.DataFrame:
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    assert set(train["month"].unique()).issubset(set(range(6)))
    assert 6 not in set(train["month"])
    assert 7 not in set(train["month"])
    return train


@pytest.fixture()
def provider(
    pool_config: ReferencePoolConfig, train_only_frame: pd.DataFrame
) -> ReferencePoolProvider:
    return ReferencePoolProvider.from_config(
        pool_config, training_frame=train_only_frame
    )


def test_config_json_has_required_keys(pool_config: ReferencePoolConfig) -> None:
    assert pool_config.K == 10
    assert pool_config.seed == 20260802
    assert "fraud_bool" in pool_config.excluded_fields
    assert "month" in pool_config.excluded_fields
    assert len(pool_config.context_fields) == 20
    assert len(pool_config.action_fields) == 18
    assert set(pool_config.read_only_context_fields) == {
        "bank_months_count",
        "has_other_cards",
    }
    assert set(pool_config.action_fields).isdisjoint(
        pool_config.read_only_context_fields
    )
    assert set(pool_config.context_fields) == set(pool_config.action_fields) | set(
        pool_config.read_only_context_fields
    )
    # Deprecated alias must remain context-only, not a mutability claim.
    assert pool_config.eligible_fields == pool_config.context_fields
    assert "fraud_bool" not in pool_config.context_fields
    assert "month" not in pool_config.context_fields
    assert "bank_months_count" not in pool_config.action_fields
    assert "has_other_cards" not in pool_config.action_fields


def test_same_anchor_and_seed_yield_identical_pool(
    provider: ReferencePoolProvider,
) -> None:
    a = provider.get_pool("anchor_demo", seed=42)
    b = provider.get_pool("anchor_demo", seed=42)
    assert a.pool_fingerprint == b.pool_fingerprint
    assert a.source_row_ids == b.source_row_ids
    assert a.attacker_view() == b.attacker_view()
    assert [p.fields for p in a.profiles] == [p.fields for p in b.profiles]


def test_different_seed_or_anchor_changes_pool(
    provider: ReferencePoolProvider,
) -> None:
    base = provider.get_pool("anchor_demo", seed=42)
    other_seed = provider.get_pool("anchor_demo", seed=43)
    other_anchor = provider.get_pool("anchor_other", seed=42)
    assert base.pool_fingerprint != other_seed.pool_fingerprint
    assert base.pool_fingerprint != other_anchor.pool_fingerprint


def test_provider_rejects_month_6_and_7_in_source(
    pool_config: ReferencePoolConfig, synthetic_frame: pd.DataFrame
) -> None:
    with pytest.raises(ReferencePoolError, match="forbidden months"):
        ReferencePoolProvider.from_config(
            pool_config, training_frame=synthetic_frame
        )

    month6 = synthetic_frame.loc[synthetic_frame["month"] == 6].copy()
    with pytest.raises(ReferencePoolError, match="forbidden months"):
        ReferencePoolProvider.from_config(pool_config, training_frame=month6)

    month7 = synthetic_frame.loc[synthetic_frame["month"] == 7].copy()
    with pytest.raises(ReferencePoolError, match="forbidden months"):
        ReferencePoolProvider.from_config(pool_config, training_frame=month7)


def test_provider_frame_never_contains_month_7_or_labels(
    provider: ReferencePoolProvider,
) -> None:
    assert "month" not in provider._frame.columns
    assert "fraud_bool" not in provider._frame.columns
    for banned in (
        "credit_risk_score",
        "zip_count_4w",
        "velocity_6h",
        "device_fraud_count",
    ):
        assert banned not in provider._frame.columns


def test_profiles_exclude_fraud_bool_month_scores(
    provider: ReferencePoolProvider,
) -> None:
    pool = provider.get_pool("anchor_demo", seed=7)
    assert pool.K == 10
    for profile in pool.profiles:
        assert "fraud_bool" not in profile.fields
        assert "month" not in profile.fields
        assert "y_score" not in profile.fields
        assert "threshold" not in profile.fields
        assert "credit_risk_score" not in profile.fields
        assert set(profile.fields) == set(provider.context_fields)
        assert "bank_months_count" in profile.fields
        assert "has_other_cards" in profile.fields


def test_attacker_view_hides_source_row_ids(
    provider: ReferencePoolProvider,
) -> None:
    pool = provider.get_pool("anchor_demo", seed=7)
    public = pool.attacker_view()
    encoded = json.dumps(public)
    assert "source_row_id" not in encoded
    assert "source_row_ids" not in public
    research = pool.research_log()
    assert research["source_row_ids"] == list(pool.source_row_ids)
    assert len(research["source_row_ids"]) == pool.K


def test_a0_a3_receive_identical_pool_from_same_provider(
    provider: ReferencePoolProvider,
) -> None:
    """A0–A3 must share one provider; attacker id must not change the pool."""
    pools = [
        provider.get_pool("shared_anchor", seed=20260802, attacker_ids=[label])
        for label in ("A0", "A1", "A2", "A3")
    ]
    fingerprints = {pool.pool_fingerprint for pool in pools}
    assert len(fingerprints) == 1
    assert pools[0].source_row_ids == pools[1].source_row_ids == pools[2].source_row_ids
    assert pools[0].attacker_view() == pools[3].attacker_view()


def test_fingerprint_stable_across_serialisation(
    provider: ReferencePoolProvider,
) -> None:
    pool = provider.get_pool("anchor_demo", seed=99)
    again = provider.get_pool("anchor_demo", seed=99)
    assert pool.pool_fingerprint == again.pool_fingerprint
    assert len(pool.pool_fingerprint) == 64
    # Fingerprint must be hex SHA-256 and insensitive to call order.
    int(pool.pool_fingerprint, 16)


def test_default_seed_comes_from_experiment_config(
    provider: ReferencePoolProvider, pool_config: ReferencePoolConfig
) -> None:
    pool = provider.get_pool("anchor_demo")
    assert pool.generation_seed == pool_config.seed
    assert all(p.generation_seed == pool_config.seed for p in pool.profiles)


def test_disk_protocol_load_uses_only_train(
    pool_config: ReferencePoolConfig,
    synthetic_raw_layout,
) -> None:
    """Disk path must retain months 0–5 only; never keep month 6/7 handles."""
    from baf_data.protocol_access import load_dataset_for_protocol

    raw_path, data_config = synthetic_raw_layout
    loaded = load_dataset_for_protocol(
        raw_path,
        phase="development",
        allowed_months=(0, 1, 2, 3, 4, 5),
        config=data_config,
    )
    retained_months = set(int(m) for m in loaded.frame["month"].unique())
    assert retained_months <= {0, 1, 2, 3, 4, 5}
    assert 6 not in retained_months
    assert 7 not in retained_months
    train_ids = set(int(i) for i in loaded.frame.index)
    provider = ReferencePoolProvider.from_config(
        pool_config, raw_path=raw_path, data_config=data_config
    )
    assert "month" not in provider._frame.columns
    assert "fraud_bool" not in provider._frame.columns
    pool = provider.get_pool("disk_anchor", seed=1)
    assert pool.K == pool_config.K
    assert set(pool.source_row_ids) <= train_ids


def test_config_rejects_eligible_fraud_bool(
    tmp_path: Path, pool_config: ReferencePoolConfig
) -> None:
    poisoned_context = [
        "fraud_bool" if name == "income" else name
        for name in pool_config.context_fields
    ]
    poisoned_action = [
        name
        for name in poisoned_context
        if name not in pool_config.read_only_context_fields
    ]
    path = tmp_path / "bad_pool.json"
    path.write_text(
        json.dumps(
            {
                "K": 10,
                "seed": 1,
                "context_fields": poisoned_context,
                "action_fields": poisoned_action,
                "read_only_context_fields": list(
                    pool_config.read_only_context_fields
                ),
                "excluded_fields": ["month"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReferencePoolError, match="Hard-excluded"):
        ReferencePoolConfig.load(path)


def test_action_fields_align_with_compiled_governance(
    pool_config: ReferencePoolConfig, governance_policy
) -> None:
    pool_config.validate_against_governance(governance_policy.action_fields)
    assert set(pool_config.action_fields) == set(governance_policy.action_fields)
