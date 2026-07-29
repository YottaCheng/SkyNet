"""Unit tests for feature/target views and frozen-schema enforcement."""

from __future__ import annotations

import pandas.testing as pdt
import pytest

from baf_data.errors import SchemaValidationError
from baf_data.sentinels import normalise_sentinels
from baf_data.splitting import build_temporal_indices
from baf_data.views import create_feature_target_views, validate_feature_schema

FORBIDDEN_IN_X = (
    "fraud_bool",
    "month",
    "device_fraud_count",
    "days_since_request",
    "credit_risk_score",
)


@pytest.fixture()
def views(synthetic_frame, synthetic_config):
    normalised, _ = normalise_sentinels(synthetic_frame, synthetic_config.sentinel_rules)
    indices = build_temporal_indices(normalised, synthetic_config)
    return create_feature_target_views(normalised, indices, synthetic_config), indices


def test_forbidden_columns_absent_from_every_x_view(views, synthetic_config) -> None:
    split_views, _ = views
    for view in split_views.values():
        for column in FORBIDDEN_IN_X:
            assert column not in view.X.columns
        assert tuple(view.X.columns) == synthetic_config.feature_columns


def test_targets_are_unchanged_and_no_rows_dropped(views, synthetic_frame) -> None:
    split_views, indices = views
    total_rows = 0
    for name, view in split_views.items():
        total_rows += len(view.X)
        assert len(view.X) == len(indices[name])
        pdt.assert_series_equal(view.y, synthetic_frame["fraud_bool"].loc[indices[name]])
    assert total_rows == len(synthetic_frame)


def test_validate_feature_schema_rejects_forbidden_column(views, synthetic_config) -> None:
    split_views, _ = views
    X_bad = split_views["train"].X.copy()
    X_bad["fraud_bool"] = 0
    with pytest.raises(SchemaValidationError, match="Forbidden column"):
        validate_feature_schema(X_bad, synthetic_config)


def test_validate_feature_schema_rejects_missing_column(views, synthetic_config) -> None:
    split_views, _ = views
    X_bad = split_views["train"].X.drop(columns=["income"])
    with pytest.raises(SchemaValidationError, match="do not match"):
        validate_feature_schema(X_bad, synthetic_config)
