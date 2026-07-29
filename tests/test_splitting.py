"""Unit tests for the deterministic temporal split."""

from __future__ import annotations

import pandas as pd
import pytest

from baf_data.errors import SplitValidationError
from baf_data.splitting import build_temporal_indices


def test_every_row_lands_in_exactly_one_split(synthetic_frame, synthetic_config) -> None:
    indices = build_temporal_indices(synthetic_frame, synthetic_config)
    assert set(indices) == {"train", "dev", "test"}

    combined = indices["train"].append(indices["dev"]).append(indices["test"])
    assert len(combined) == len(synthetic_frame)
    assert combined.duplicated().sum() == 0
    assert set(combined) == set(synthetic_frame.index)


def test_splits_contain_only_their_frozen_months(synthetic_frame, synthetic_config) -> None:
    indices = build_temporal_indices(synthetic_frame, synthetic_config)
    months = synthetic_frame["month"]
    assert set(months.loc[indices["train"]].unique()) == {0, 1, 2, 3, 4, 5}
    assert set(months.loc[indices["dev"]].unique()) == {6}
    assert set(months.loc[indices["test"]].unique()) == {7}


def test_split_is_deterministic(synthetic_frame, synthetic_config) -> None:
    first = build_temporal_indices(synthetic_frame, synthetic_config)
    second = build_temporal_indices(synthetic_frame, synthetic_config)
    for name in first:
        pd.testing.assert_index_equal(first[name], second[name])


def test_unassigned_month_value_is_rejected(synthetic_frame, synthetic_config) -> None:
    corrupted = synthetic_frame.copy()
    corrupted.loc[corrupted.index[-1], "month"] = 8
    with pytest.raises(SplitValidationError, match="not assigned to any split"):
        build_temporal_indices(corrupted, synthetic_config)
