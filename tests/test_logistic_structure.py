"""Structural and behavioural tests for the unfitted LR scaffold.

Structure tests inspect the four-branch preprocessing design and the
pipeline composition. Behaviour tests fit **only the preprocessor** on
small synthetic in-memory frames to verify transformer semantics.
No test reads Base.csv, fits the classifier, or touches months 6/7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from baf_data.config import FROZEN_CONFIG
from baf_data.sentinels import normalise_sentinels
from baf_models.logistic import (
    CLASSIFIER_STEP,
    PREPROCESSING_STEP,
    LogisticBaselineConfig,
    build_logistic_pipeline,
)
from baf_models.preprocessing import (
    BinaryPassthrough,
    build_preprocessor,
    feature_groups,
)
from conftest import make_synthetic_frame

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "logistic_baseline.yaml"

CUSTOM_CONFIG = LogisticBaselineConfig(
    C=0.5,
    max_iter=250,
    solver="liblinear",
    random_state=7,
    class_weight={0: 1.0, 1: 12.5},
)

EXCLUDED_COLUMNS = (
    "fraud_bool",
    "month",
    "device_fraud_count",
    "days_since_request",
    "credit_risk_score",
)


@pytest.fixture(autouse=True)
def forbid_dataset_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test in this module that tries to load a dataset."""

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("The model scaffold must not read any dataset.")

    monkeypatch.setattr(pd, "read_csv", _forbidden)
    monkeypatch.setattr(pd, "read_parquet", _forbidden)


def synthetic_feature_matrix() -> pd.DataFrame:
    """Synthetic X with the frozen feature columns and NaN sentinels."""
    frame = make_synthetic_frame()
    normalised, _ = normalise_sentinels(frame, FROZEN_CONFIG.sentinel_rules)
    return normalised[list(FROZEN_CONFIG.feature_columns)]


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature grouping (schema-derived, no duplicated lists)
# ---------------------------------------------------------------------------

def test_groups_partition_the_frozen_features_exactly() -> None:
    groups = feature_groups(FROZEN_CONFIG)
    value_branches = (
        set(groups.continuous) | set(groups.binary) | set(groups.categorical)
    )
    assert value_branches == set(FROZEN_CONFIG.feature_columns)
    assert len(groups.continuous) + len(groups.binary) + len(groups.categorical) == 27
    assert set(groups.continuous).isdisjoint(groups.binary)
    assert set(groups.continuous).isdisjoint(groups.categorical)
    assert set(groups.binary).isdisjoint(groups.categorical)


def test_group_membership_matches_the_frozen_register() -> None:
    groups = feature_groups(FROZEN_CONFIG)
    assert set(groups.binary) == {
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "has_other_cards",
        "foreign_request",
        "keep_alive_session",
    }
    assert groups.binary == FROZEN_CONFIG.binary_features
    assert set(groups.categorical) == {
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    }
    assert len(groups.continuous) == 16
    # Indicators exist for exactly the frozen sentinel-rule columns and
    # every indicator source is a continuous feature.
    assert set(groups.missing_indicator_sources) == {
        rule.column for rule in FROZEN_CONFIG.sentinel_rules
    }
    assert set(groups.missing_indicator_sources) <= set(groups.continuous)


def test_binary_membership_has_a_single_executable_source_of_truth() -> None:
    """The model layer declares no binary field list of its own.

    Binary membership must come exclusively from the injected
    DataLayerConfig: no module-level list survives in baf_models, and
    changing the config must change the derived grouping accordingly.
    """
    import baf_models.preprocessing as preprocessing_module

    assert not hasattr(preprocessing_module, "BINARY_FEATURES")

    from dataclasses import replace

    reduced = ("email_is_free", "foreign_request")
    modified_config = replace(FROZEN_CONFIG, binary_features=reduced)
    modified_config.validate()
    groups = feature_groups(modified_config)
    assert groups.binary == reduced
    # Fields dropped from the config's binary list fall back to the
    # continuous branch, proving derivation is config-driven.
    for reassigned in set(FROZEN_CONFIG.binary_features) - set(reduced):
        assert reassigned in groups.continuous


def test_excluded_columns_absent_from_every_branch() -> None:
    groups = feature_groups(FROZEN_CONFIG)
    for excluded in EXCLUDED_COLUMNS:
        assert excluded not in groups.continuous
        assert excluded not in groups.binary
        assert excluded not in groups.categorical
        assert excluded not in groups.missing_indicator_sources


def test_preprocessor_has_four_expected_branches() -> None:
    preprocessor = build_preprocessor(FROZEN_CONFIG)
    branches = {
        name: (transformer, columns)
        for name, transformer, columns in preprocessor.transformers
    }
    assert set(branches) == {"continuous", "missing_indicator", "binary", "categorical"}

    continuous_steps = dict(branches["continuous"][0].steps)
    assert isinstance(continuous_steps["imputer"], SimpleImputer)
    assert continuous_steps["imputer"].strategy == "median"
    assert continuous_steps["imputer"].add_indicator is False
    assert isinstance(continuous_steps["scaler"], StandardScaler)

    indicator = branches["missing_indicator"][0]
    assert isinstance(indicator, MissingIndicator)
    assert indicator.features == "all"

    assert isinstance(branches["binary"][0], BinaryPassthrough)

    categorical_steps = dict(branches["categorical"][0].steps)
    assert isinstance(categorical_steps["onehot"], OneHotEncoder)
    assert categorical_steps["onehot"].handle_unknown == "ignore"


# ---------------------------------------------------------------------------
# Transformer behaviour on synthetic in-memory data only
# ---------------------------------------------------------------------------

def _fit_transform_dense(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Fit ONLY the preprocessor on synthetic data; return dense output."""
    preprocessor = build_preprocessor(FROZEN_CONFIG)
    transformed = preprocessor.fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype="float64"), list(
        preprocessor.get_feature_names_out()
    )


def test_continuous_branch_imputes_and_scales() -> None:
    X = synthetic_feature_matrix()
    output, names = _fit_transform_dense(X)
    continuous_idx = [i for i, n in enumerate(names) if n.startswith("continuous__")]
    block = output[:, continuous_idx]
    assert not np.isnan(block).any()  # median imputation filled the NaNs
    np.testing.assert_allclose(block.mean(axis=0), 0.0, atol=1e-9)  # standardised


def test_missing_indicators_are_binary_and_unscaled() -> None:
    X = synthetic_feature_matrix()
    output, names = _fit_transform_dense(X)
    indicator_idx = [
        i for i, n in enumerate(names) if n.startswith("missing_indicator__")
    ]
    assert len(indicator_idx) == len(FROZEN_CONFIG.sentinel_rules)
    block = output[:, indicator_idx]
    assert np.isin(block, (0.0, 1.0)).all()
    assert block.sum() > 0  # the synthetic frame does contain missing values
    # Indicators agree with the actual NaN pattern of the source columns.
    groups_sources = [
        n.removeprefix("missing_indicator__missingindicator_") for n in
        (names[i] for i in indicator_idx)
    ]
    for column, block_col in zip(groups_sources, block.T):
        np.testing.assert_array_equal(block_col, X[column].isna().to_numpy(float))


def test_binary_branch_passes_values_through_unscaled() -> None:
    X = synthetic_feature_matrix()
    output, names = _fit_transform_dense(X)
    binary_idx = [i for i, n in enumerate(names) if n.startswith("binary__")]
    assert len(binary_idx) == len(FROZEN_CONFIG.binary_features)
    block = output[:, binary_idx]
    binary_names = [names[i].removeprefix("binary__") for i in binary_idx]
    np.testing.assert_array_equal(block, X[binary_names].to_numpy(dtype="float64"))


def test_binary_branch_rejects_non_binary_values() -> None:
    X = synthetic_feature_matrix().copy()
    X.loc[X.index[0], "email_is_free"] = 2
    preprocessor = build_preprocessor(FROZEN_CONFIG)
    with pytest.raises(ValueError, match="email_is_free"):
        preprocessor.fit_transform(X)


def test_categorical_branch_is_one_hot_and_unknown_safe() -> None:
    X = synthetic_feature_matrix()
    preprocessor = build_preprocessor(FROZEN_CONFIG)
    fitted_output = preprocessor.fit_transform(X)
    if hasattr(fitted_output, "toarray"):
        fitted_output = fitted_output.toarray()
    names = list(preprocessor.get_feature_names_out())
    categorical_idx = [i for i, n in enumerate(names) if n.startswith("categorical__")]
    block = np.asarray(fitted_output, dtype="float64")[:, categorical_idx]
    assert np.isin(block, (0.0, 1.0)).all()

    # An unseen category at transform time must not raise and must encode
    # to all-zeros within that feature's one-hot block.
    X_unseen = X.copy()
    X_unseen.loc[X_unseen.index[0], "payment_type"] = "ZZ_never_seen"
    transformed = preprocessor.transform(X_unseen)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    payment_idx = [
        i for i, n in enumerate(names) if n.startswith("categorical__payment_type_")
    ]
    assert len(payment_idx) > 0
    assert np.asarray(transformed)[0, payment_idx].sum() == 0.0


def test_classifier_remains_unfitted_after_preprocessor_experiments() -> None:
    X = synthetic_feature_matrix()
    pipeline = build_logistic_pipeline(CUSTOM_CONFIG)
    # Fitting a standalone preprocessor clone must not fit the pipeline.
    build_preprocessor(FROZEN_CONFIG).fit(X)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[CLASSIFIER_STEP])
