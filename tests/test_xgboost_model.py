"""Structural and minimal behavioural tests for the unfitted XGBoost scaffold.

Uses in-memory synthetic frames only. Never reads Base.csv, never scores
month 6 or month 7 as experimental splits, and never executes Git.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from baf_data.config import FROZEN_CONFIG
from baf_data.sentinels import normalise_sentinels
from baf_models.preprocessing import feature_groups
from baf_models.xgboost_model import (
    CLASSIFIER_STEP,
    PREPROCESSING_STEP,
    XGBoostBaselineConfig,
    build_xgboost_classifier,
    build_xgboost_pipeline,
)
from conftest import make_synthetic_frame
import baf_models.xgboost_model as xgboost_module

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "xgboost_baseline.yaml"

EXCLUDED_COLUMNS = (
    "fraud_bool",
    "month",
    "device_fraud_count",
    "days_since_request",
    "credit_risk_score",
)

GPU_PARAM_SUBSTRINGS = ("gpu", "cuda")


@pytest.fixture(autouse=True)
def forbid_real_dataset_and_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse Base.csv loads and any Git subprocess during these tests."""

    real_read_csv = pd.read_csv

    def _guarded_read_csv(filepath_or_buffer, *args, **kwargs):  # type: ignore[no-untyped-def]
        path = str(filepath_or_buffer)
        if "Base.csv" in path or "raw/baf" in path:
            raise AssertionError("XGBoost scaffold tests must not read Base.csv.")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _guarded_read_csv)
    monkeypatch.setattr(pd, "read_parquet", _guarded_read_csv)

    real_run = subprocess.run
    real_popen = subprocess.Popen

    def _forbid_git(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokens = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if tokens and Path(str(tokens[0])).name == "git":
            raise AssertionError("XGBoost scaffold tests must not execute Git.")
        return real_run(cmd, *args, **kwargs)

    def _forbid_git_popen(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokens = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if tokens and Path(str(tokens[0])).name == "git":
            raise AssertionError("XGBoost scaffold tests must not execute Git.")
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _forbid_git)
    monkeypatch.setattr(subprocess, "Popen", _forbid_git_popen)


def _synthetic_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Small synthetic feature matrix and labels (no Base.csv)."""
    frame = make_synthetic_frame()
    normalised, _ = normalise_sentinels(frame, FROZEN_CONFIG.sentinel_rules)
    X = normalised[list(FROZEN_CONFIG.feature_columns)]
    y = normalised[FROZEN_CONFIG.target_column]
    return X, y


# ---------------------------------------------------------------------------
# YAML and construction
# ---------------------------------------------------------------------------


def test_yaml_loads_fixed_parameters() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    assert config.objective == "binary:logistic"
    assert config.eval_metric == "aucpr"
    assert config.n_estimators == 500
    assert config.max_depth == 6
    assert config.learning_rate == pytest.approx(0.05)
    assert config.min_child_weight == pytest.approx(1)
    assert config.subsample == pytest.approx(0.8)
    assert config.colsample_bytree == pytest.approx(0.8)
    assert config.gamma == pytest.approx(0)
    assert config.reg_alpha == pytest.approx(0)
    assert config.reg_lambda == pytest.approx(1)
    assert config.scale_pos_weight == pytest.approx(1)
    assert config.tree_method == "hist"
    assert config.random_state == 42
    assert config.n_jobs == -1
    assert config.verbosity == 1


def test_builder_returns_unfitted_sklearn_pipeline() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    pipeline = build_xgboost_pipeline(config)
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == [PREPROCESSING_STEP, CLASSIFIER_STEP]
    assert isinstance(pipeline.named_steps[PREPROCESSING_STEP], ColumnTransformer)
    assert isinstance(pipeline.named_steps[CLASSIFIER_STEP], XGBClassifier)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[CLASSIFIER_STEP])


def test_classifier_fixed_parameters() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    classifier = build_xgboost_classifier(config)
    params = classifier.get_params()
    assert params["objective"] == "binary:logistic"
    assert params["eval_metric"] == "aucpr"
    assert params["tree_method"] == "hist"
    assert params["scale_pos_weight"] == 1
    assert params["random_state"] == 42
    assert params["n_estimators"] == 500
    assert params["max_depth"] == 6
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["n_jobs"] == -1
    assert params["verbosity"] == 1


def test_pipeline_classifier_matches_yaml() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    classifier = build_xgboost_pipeline(config).named_steps[CLASSIFIER_STEP]
    assert classifier.objective == "binary:logistic"
    assert classifier.eval_metric == "aucpr"
    assert classifier.tree_method == "hist"
    assert classifier.scale_pos_weight == 1
    assert classifier.random_state == 42


def test_no_gpu_or_cuda_parameters() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    classifier = build_xgboost_classifier(config)
    params = classifier.get_params()
    for name, value in params.items():
        lowered = name.lower()
        if any(token in lowered for token in GPU_PARAM_SUBSTRINGS):
            pytest.fail(f"Unexpected GPU/CUDA-related parameter set: {name}={value!r}")
        if isinstance(value, str) and any(
            token in value.lower() for token in GPU_PARAM_SUBSTRINGS
        ):
            pytest.fail(f"Unexpected GPU/CUDA-related value: {name}={value!r}")
    # Explicit CPU hist; device must not request CUDA/GPU.
    assert params["tree_method"] == "hist"
    device = params.get("device")
    assert device in (None, "cpu")


def test_build_does_not_read_base_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("build_xgboost_pipeline must not read any dataset.")

    monkeypatch.setattr(pd, "read_csv", _forbidden)
    monkeypatch.setattr(pd, "read_parquet", _forbidden)
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    pipeline = build_xgboost_pipeline(config)
    assert isinstance(pipeline, Pipeline)


def test_module_does_not_access_month_6_or_7_evaluation() -> None:
    source = inspect.getsource(xgboost_module)
    assert "month 6" not in source.lower()
    assert "month 7" not in source.lower()
    assert "month_6" not in source
    assert "month_7" not in source
    assert 'views["test"]' not in source
    assert "Base.csv" not in source


# ---------------------------------------------------------------------------
# Feature schema from FROZEN_CONFIG (no duplicated field lists)
# ---------------------------------------------------------------------------


def test_input_fields_come_from_frozen_data_layer_config() -> None:
    groups = feature_groups(FROZEN_CONFIG)
    value_branches = set(groups.continuous) | set(groups.binary) | set(groups.categorical)
    assert value_branches == set(FROZEN_CONFIG.feature_columns)
    assert len(FROZEN_CONFIG.feature_columns) == 27
    for excluded in EXCLUDED_COLUMNS:
        assert excluded not in FROZEN_CONFIG.feature_columns

    # xgboost_model must not maintain its own feature schema constants.
    assert not hasattr(xgboost_module, "FEATURE_COLUMNS")
    assert not hasattr(xgboost_module, "BINARY_FEATURES")
    source = inspect.getsource(xgboost_module)
    for name in FROZEN_CONFIG.feature_columns:
        # Module must not hard-code individual feature names.
        assert f'"{name}"' not in source
        assert f"'{name}'" not in source


def test_preprocessor_reuses_shared_builder() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    pipeline = build_xgboost_pipeline(config)
    preprocessor = pipeline.named_steps[PREPROCESSING_STEP]
    names = [name for name, _, _ in preprocessor.transformers]
    assert names == ["continuous", "missing_indicator", "binary", "categorical"]


# ---------------------------------------------------------------------------
# Minimal synthetic fit / predict (not real BAF training)
# ---------------------------------------------------------------------------


def test_minimal_synthetic_fit_predict() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    # Tiny booster for a smoke fit only; YAML defaults stay untouched.
    smoke = XGBoostBaselineConfig(
        objective=config.objective,
        eval_metric=config.eval_metric,
        n_estimators=5,
        max_depth=2,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        gamma=config.gamma,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=config.scale_pos_weight,
        tree_method=config.tree_method,
        random_state=config.random_state,
        n_jobs=1,
        verbosity=0,
    )
    X, y = _synthetic_xy()
    pipeline = build_xgboost_pipeline(smoke)
    pipeline.fit(X, y)
    scores = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)
    assert scores.shape == (len(X),)
    assert preds.shape == (len(X),)
    assert np.isfinite(scores).all()
    assert set(np.unique(preds)).issubset({0, 1})


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="objective"):
        XGBoostBaselineConfig(
            objective="reg:squarederror",
            eval_metric="aucpr",
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            tree_method="hist",
            random_state=0,
            n_jobs=1,
            verbosity=0,
        )
    with pytest.raises(ValueError, match="tree_method"):
        XGBoostBaselineConfig(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            tree_method="gpu_hist",
            random_state=0,
            n_jobs=1,
            verbosity=0,
        )
