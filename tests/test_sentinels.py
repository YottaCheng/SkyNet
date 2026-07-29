"""Unit tests for sentinel normalisation (pure function behaviour)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from baf_data.config import FROZEN_CONFIG
from baf_data.sentinels import normalise_sentinels

RULES = FROZEN_CONFIG.sentinel_rules
RULED_COLUMNS = {rule.column for rule in RULES}


def test_only_specified_fields_and_values_are_converted(synthetic_frame) -> None:
    result, counts = normalise_sentinels(synthetic_frame, RULES)

    for rule in RULES:
        original = synthetic_frame[rule.column]
        mask = original == rule.value if rule.strategy == "equals" else original < rule.value
        assert counts[rule.column] == int(mask.sum())
        assert result[rule.column][mask].isna().all()
        # Non-sentinel values are numerically unchanged.
        np.testing.assert_allclose(
            result[rule.column][~mask].to_numpy(dtype="float64"),
            original[~mask].to_numpy(dtype="float64"),
        )

    # Every column without a rule is byte-for-byte identical.
    for column in synthetic_frame.columns:
        if column not in RULED_COLUMNS:
            pdt.assert_series_equal(result[column], synthetic_frame[column])


def test_valid_negative_velocity_6h_values_are_preserved(synthetic_frame) -> None:
    result, _ = normalise_sentinels(synthetic_frame, RULES)
    negatives_before = synthetic_frame["velocity_6h"] < 0
    assert negatives_before.sum() == 2
    pdt.assert_series_equal(result["velocity_6h"], synthetic_frame["velocity_6h"])


def test_credit_risk_score_values_are_untouched(synthetic_frame) -> None:
    result, _ = normalise_sentinels(synthetic_frame, RULES)
    assert (synthetic_frame["credit_risk_score"] == -1).sum() > 0
    assert (synthetic_frame["credit_risk_score"] < -1).sum() > 0
    pdt.assert_series_equal(
        result["credit_risk_score"], synthetic_frame["credit_risk_score"]
    )


def test_no_rows_dropped_and_input_not_mutated(synthetic_frame) -> None:
    before = synthetic_frame.copy(deep=True)
    result, _ = normalise_sentinels(synthetic_frame, RULES)
    assert len(result) == len(synthetic_frame)
    assert result.isna().any().any()  # sentinels became NaN in the copy...
    pdt.assert_frame_equal(synthetic_frame, before)  # ...but the input is intact.


def test_normalisation_is_idempotent(synthetic_frame) -> None:
    once, counts_once = normalise_sentinels(synthetic_frame, RULES)
    twice, counts_twice = normalise_sentinels(once, RULES)
    pdt.assert_frame_equal(once, twice)
    assert all(count == 0 for count in counts_twice.values())
    assert set(counts_once) == RULED_COLUMNS


def test_intended_balcon_amount_all_negatives_converted_not_only_minus_one(
    synthetic_frame,
) -> None:
    result, counts = normalise_sentinels(synthetic_frame, RULES)
    original = synthetic_frame["intended_balcon_amount"]
    assert counts["intended_balcon_amount"] == int((original < 0).sum())
    assert not (result["intended_balcon_amount"] < 0).any()
    assert (result["intended_balcon_amount"] >= 0).sum() == int((original >= 0).sum())
