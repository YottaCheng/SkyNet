"""Synthetic and artifact-isolation tests for the XGBoost development run.

Uses in-memory synthetic frames and the saved unweighted LR development
artefacts. Never reads Base.csv for training, never scores month 7, and
never executes Git.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from baf_data.config import FROZEN_CONFIG
from baf_data.sentinels import normalise_sentinels
from baf_data.splitting import build_temporal_indices
from baf_data.views import create_feature_target_views
from baf_models.artifacts import (
    DEFAULT_LR_UNWEIGHTED_DIR,
    LOGISTIC_OUTPUT_ROOT,
    XGBOOST_OUTPUT_ROOT,
    XGBOOST_RUN_ID,
    ArtifactError,
    assert_path_outside_logistic_outputs,
    build_xgboost_vs_lr_rows,
    load_saved_development_metrics,
    run_directory,
    save_variant_artifacts,
    save_xgboost_vs_lr_comparison,
)
from baf_models.evaluation import select_fpr_constrained_threshold
from baf_models.training import (
    extract_train_dev,
    fit_and_score_xgboost,
    score_fitted_pipeline,
)
from baf_models.xgboost_model import XGBoostBaselineConfig, build_xgboost_pipeline
import baf_models.evaluation as evaluation_module
import baf_models.training as training_module
import run_xgboost_baseline as xgb_cli

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "xgboost_baseline.yaml"
LR_UNWEIGHTED = DEFAULT_LR_UNWEIGHTED_DIR


@pytest.fixture(autouse=True)
def forbid_real_dataset_and_git(monkeypatch: pytest.MonkeyPatch) -> None:
    real_read_csv = pd.read_csv

    def _guarded_read_csv(filepath_or_buffer, *args, **kwargs):  # type: ignore[no-untyped-def]
        path = str(filepath_or_buffer)
        if "Base.csv" in path or "raw/baf" in path:
            raise AssertionError("XGBoost training tests must not read Base.csv.")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _guarded_read_csv)

    real_run = subprocess.run

    def _forbid_git(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokens = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if tokens and Path(str(tokens[0])).name == "git":
            raise AssertionError("XGBoost training tests must not execute Git.")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _forbid_git)


@pytest.fixture()
def synthetic_train_dev(synthetic_frame, synthetic_config):
    normalised, _ = normalise_sentinels(synthetic_frame, synthetic_config.sentinel_rules)
    indices = build_temporal_indices(normalised, synthetic_config)
    views = create_feature_target_views(normalised, indices, synthetic_config)
    bundle = extract_train_dev(
        {"train": views["train"], "dev": views["dev"]},
        feature_columns=synthetic_config.feature_columns,
        raw_sha256="synthetic",
        data_config=synthetic_config,
    )
    return bundle, synthetic_config, views


def _smoke_config() -> XGBoostBaselineConfig:
    base = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    return XGBoostBaselineConfig(
        objective=base.objective,
        eval_metric=base.eval_metric,
        n_estimators=5,
        max_depth=2,
        learning_rate=base.learning_rate,
        min_child_weight=base.min_child_weight,
        subsample=base.subsample,
        colsample_bytree=base.colsample_bytree,
        gamma=base.gamma,
        reg_alpha=base.reg_alpha,
        reg_lambda=base.reg_lambda,
        scale_pos_weight=base.scale_pos_weight,
        tree_method=base.tree_method,
        random_state=base.random_state,
        n_jobs=1,
        verbosity=0,
    )


def test_frozen_yaml_not_overridden_by_cli_defaults() -> None:
    config = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    assert config.n_estimators == 500
    assert config.scale_pos_weight == 1
    assert config.tree_method == "hist"
    assert config.random_state == 42
    # CLI argparse defaults must not redefine model hyperparameters.
    source = inspect.getsource(xgb_cli)
    assert "n_estimators" not in source or "from_yaml" in source
    assert "scale_pos_weight" not in source or "model_config.scale_pos_weight" in source


def test_fit_receives_only_training_rows(synthetic_train_dev, monkeypatch) -> None:
    bundle, config, _ = synthetic_train_dev
    seen: dict[str, object] = {}
    real_fit = Pipeline.fit

    def _spy_fit(self, X, y=None, **kwargs):  # type: ignore[no-untyped-def]
        seen["X"] = X
        seen["y"] = y
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", _spy_fit)
    fit_and_score_xgboost(bundle, _smoke_config(), config)
    assert seen["X"] is bundle.train.X
    assert seen["y"] is bundle.train.y


def test_predict_uses_development_only(synthetic_train_dev, monkeypatch) -> None:
    bundle, config, _ = synthetic_train_dev
    seen: dict[str, object] = {}
    real_predict_proba = Pipeline.predict_proba

    def _spy(self, X, **kwargs):  # type: ignore[no-untyped-def]
        seen["X"] = X
        return real_predict_proba(self, X, **kwargs)

    monkeypatch.setattr(Pipeline, "predict_proba", _spy)
    fit_and_score_xgboost(bundle, _smoke_config(), config)
    assert seen["X"] is bundle.development.X


def test_month7_not_in_training_path(synthetic_train_dev) -> None:
    bundle, config, all_views = synthetic_train_dev
    assert "test" not in {"train": bundle.train, "dev": bundle.development}
    # Training API never reads a test key.
    source = inspect.getsource(training_module)
    assert 'views["test"]' not in source
    fit_and_score_xgboost(bundle, _smoke_config(), config)
    # Presence of a test view in the outer mapping must not be required.
    assert all_views["test"].name == "test"


def test_scale_pos_weight_and_no_early_stopping(synthetic_train_dev) -> None:
    bundle, config, _ = synthetic_train_dev
    model = _smoke_config()
    assert model.scale_pos_weight == 1
    result = fit_and_score_xgboost(bundle, model, config)
    classifier = result.pipeline.named_steps["classifier"]
    params = classifier.get_params()
    assert params["scale_pos_weight"] == 1
    assert params["early_stopping_rounds"] is None
    assert params["tree_method"] == "hist"


def test_uses_shared_threshold_selection(synthetic_train_dev, monkeypatch) -> None:
    bundle, config, _ = synthetic_train_dev
    called = {"count": 0}
    real = select_fpr_constrained_threshold

    def _spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        called["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "select_fpr_constrained_threshold", _spy)
    # evaluate_development_scores imports the symbol at call time via module attr
    import baf_models.evaluation as ev

    monkeypatch.setattr(ev, "select_fpr_constrained_threshold", _spy)
    fit_and_score_xgboost(bundle, _smoke_config(), config)
    assert called["count"] >= 1


def test_output_path_does_not_overwrite_logistic() -> None:
    xgb_dir = run_directory(XGBOOST_OUTPUT_ROOT, XGBOOST_RUN_ID)
    assert_path_outside_logistic_outputs(xgb_dir)
    with pytest.raises(ArtifactError, match="logistic baseline"):
        assert_path_outside_logistic_outputs(LOGISTIC_OUTPUT_ROOT / "anything")
    assert xgb_dir != LOGISTIC_OUTPUT_ROOT / "logistic_dev_baseline_2026-07-29"


def test_forbidden_month7_artifact_names(tmp_path: Path) -> None:
    from baf_models.artifacts import assert_development_artifact_names

    with pytest.raises(ArtifactError):
        assert_development_artifact_names(
            {"bad": tmp_path / "scores_month7.csv"}
        )


def test_reads_real_lr_artifacts_for_comparison(synthetic_train_dev) -> None:
    if not LR_UNWEIGHTED.is_dir():
        pytest.skip("Saved unweighted LR artefacts are not available.")
    lr_metrics = load_saved_development_metrics(LR_UNWEIGHTED)
    assert lr_metrics["split"] == "development"
    assert lr_metrics["month"] == 6
    assert lr_metrics["variant"] == "unweighted"
    # Exact saved AUPRC from the frozen LR development run.
    assert lr_metrics["threshold_independent"]["auprc"] == pytest.approx(
        0.15475964663922073
    )

    bundle, config, _ = synthetic_train_dev
    result = fit_and_score_xgboost(bundle, _smoke_config(), config)
    rows = build_xgboost_vs_lr_rows(result, lr_variant_dir=LR_UNWEIGHTED)
    assert {row["model"] for row in rows} == {"xgboost", "logistic_unweighted"}
    lr_row = next(row for row in rows if row["model"] == "logistic_unweighted")
    assert lr_row["auprc"] == pytest.approx(0.15475964663922073)


def test_end_to_end_synthetic_artifacts(synthetic_train_dev, tmp_path: Path) -> None:
    bundle, config, _ = synthetic_train_dev
    result = fit_and_score_xgboost(bundle, _smoke_config(), config)
    out = tmp_path / "xgboost_baseline" / "run"
    paths = save_variant_artifacts(
        result, out / "xgboost", data_config=config, raw_sha256="synthetic"
    )
    assert paths["scores"].name == "development_month6_scores.csv"
    assert "month7" not in str(paths["scores"]).lower()
    assert "test" not in paths["scores"].name

    if LR_UNWEIGHTED.is_dir():
        csv_path, json_path = save_xgboost_vs_lr_comparison(
            result, out, lr_variant_dir=LR_UNWEIGHTED
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["month"] == 6
        assert csv_path.is_file()

    ranking = result.evaluation["threshold_independent"]
    assert 0.0 <= ranking["auprc"] <= 1.0
    assert np.isfinite(result.development_y_score).all()


def test_score_fitted_pipeline_refuses_non_dev_name(synthetic_train_dev) -> None:
    from baf_data.views import SplitView
    from baf_models.training import TrainDevBundle, TrainingError

    bundle, config, _ = synthetic_train_dev
    mutated = TrainDevBundle(
        train=bundle.train,
        development=SplitView(
            name="test", X=bundle.development.X, y=bundle.development.y
        ),
        feature_columns=bundle.feature_columns,
        raw_sha256=bundle.raw_sha256,
    )
    pipeline = build_xgboost_pipeline(_smoke_config(), config)
    pipeline.fit(bundle.train.X, bundle.train.y)
    with pytest.raises(TrainingError, match="non-development"):
        score_fitted_pipeline(
            mutated,
            "xgboost",
            pipeline,
            _smoke_config(),
            fit_seconds=0.1,
        )
