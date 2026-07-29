"""Structural tests for the unfitted Logistic Regression scaffold.

These tests build and inspect the pipeline only. They never call
``fit``/``predict``, never read Base.csv and never touch months 6 or 7.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from baf_data.config import FROZEN_CONFIG
from baf_models.logistic import (
    CLASSIFIER_STEP,
    PREPROCESSING_STEP,
    LogisticBaselineConfig,
    build_logistic_pipeline,
)
from baf_models.preprocessing import build_preprocessor, split_feature_kinds

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "logistic_baseline.yaml"

CUSTOM_CONFIG = LogisticBaselineConfig(
    C=0.5,
    max_iter=250,
    solver="liblinear",
    random_state=7,
    class_weight={0: 1.0, 1: 12.5},
)


@pytest.fixture(autouse=True)
def forbid_dataset_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test in this module that tries to load a dataset."""

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("The model scaffold must not read any dataset.")

    monkeypatch.setattr(pd, "read_csv", _forbidden)
    monkeypatch.setattr(pd, "read_parquet", _forbidden)


def test_builder_returns_sklearn_pipeline() -> None:
    pipeline = build_logistic_pipeline(CUSTOM_CONFIG)
    assert isinstance(pipeline, Pipeline)


def test_pipeline_has_preprocessing_and_classifier_stages() -> None:
    pipeline = build_logistic_pipeline(CUSTOM_CONFIG)
    assert [name for name, _ in pipeline.steps] == [PREPROCESSING_STEP, CLASSIFIER_STEP]
    assert isinstance(pipeline.named_steps[PREPROCESSING_STEP], ColumnTransformer)


def test_classifier_is_logistic_regression() -> None:
    pipeline = build_logistic_pipeline(CUSTOM_CONFIG)
    assert isinstance(pipeline.named_steps[CLASSIFIER_STEP], LogisticRegression)


def test_configuration_values_are_passed_to_the_classifier() -> None:
    classifier = build_logistic_pipeline(CUSTOM_CONFIG).named_steps[CLASSIFIER_STEP]
    assert classifier.C == 0.5
    assert classifier.max_iter == 250
    assert classifier.solver == "liblinear"
    assert classifier.random_state == 7
    assert classifier.class_weight == {0: 1.0, 1: 12.5}


def test_yaml_config_loads_and_parameterises_the_pipeline() -> None:
    config = LogisticBaselineConfig.from_yaml(YAML_PATH)
    assert config.C == 1.0
    assert config.max_iter == 1000
    assert config.solver == "lbfgs"
    assert config.random_state == 42
    assert config.class_weight == "balanced"

    classifier = build_logistic_pipeline(config).named_steps[CLASSIFIER_STEP]
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 42


def test_pipeline_is_returned_unfitted() -> None:
    pipeline = build_logistic_pipeline(CUSTOM_CONFIG)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[CLASSIFIER_STEP])


def test_preprocessor_covers_exactly_the_frozen_feature_columns() -> None:
    numeric, categorical = split_feature_kinds(FROZEN_CONFIG)
    assert set(numeric).isdisjoint(categorical)
    assert set(numeric) | set(categorical) == set(FROZEN_CONFIG.feature_columns)
    assert len(numeric) + len(categorical) == 27
    assert set(categorical) == {
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    }
    for forbidden in ("fraud_bool", "month", "device_fraud_count",
                      "days_since_request", "credit_risk_score"):
        assert forbidden not in numeric
        assert forbidden not in categorical


def test_preprocessor_pipelines_have_required_components() -> None:
    preprocessor = build_preprocessor(FROZEN_CONFIG)
    transformers = dict(
        (name, (transformer, columns))
        for name, transformer, columns in preprocessor.transformers
    )
    assert set(transformers) == {"numeric", "categorical"}

    numeric_steps = dict(transformers["numeric"][0].steps)
    assert isinstance(numeric_steps["imputer"], SimpleImputer)
    assert numeric_steps["imputer"].strategy == "median"
    assert numeric_steps["imputer"].add_indicator is True
    assert isinstance(numeric_steps["scaler"], StandardScaler)

    categorical_steps = dict(transformers["categorical"][0].steps)
    assert isinstance(categorical_steps["onehot"], OneHotEncoder)
    assert categorical_steps["onehot"].handle_unknown == "ignore"


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="C must be positive"):
        LogisticBaselineConfig(
            C=0.0, max_iter=100, solver="lbfgs", random_state=0, class_weight=None
        )
    with pytest.raises(ValueError, match="Unsupported solver"):
        LogisticBaselineConfig(
            C=1.0, max_iter=100, solver="not-a-solver", random_state=0, class_weight=None
        )
    with pytest.raises(ValueError, match="class_weight"):
        LogisticBaselineConfig(
            C=1.0, max_iter=100, solver="lbfgs", random_state=0,
            class_weight="unbalanced",  # type: ignore[arg-type]
        )
