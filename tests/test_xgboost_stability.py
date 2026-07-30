"""Minimal tests for the fixed-seed XGBoost stability protocol."""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baf_data.sentinels import normalise_sentinels
from baf_data.splitting import build_temporal_indices
from baf_data.views import create_feature_target_views
from baf_models.artifacts import (
    FROZEN_XGBOOST_BASELINE_DIR,
    LOGISTIC_OUTPUT_ROOT,
    XGBOOST_STABILITY_OUTPUT_ROOT,
    XGBOOST_STABILITY_RUN_ID,
    ArtifactError,
    assert_stability_output_isolated,
    load_saved_development_metrics,
    run_directory,
    save_stability_artifacts,
)
from baf_models.stability import (
    STABILITY_SEEDS,
    StabilityError,
    assert_only_random_state_differs,
    build_stability_summary,
    config_with_seed,
    metrics_row_from_result,
    summarise_numeric,
    validate_stability_seeds,
)
from baf_models.training import extract_train_dev, fit_and_score_xgboost
from baf_models.xgboost_model import XGBoostBaselineConfig
import baf_models.stability as stability_module
import run_xgboost_stability as stability_cli

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "xgboost_baseline.yaml"
LR_UNWEIGHTED = (
    Path("/Users/ziyaoch/ucl/dissertation/05_outputs/logistic_baseline")
    / "logistic_dev_baseline_2026-07-29"
    / "unweighted"
)


@pytest.fixture(autouse=True)
def forbid_real_dataset_and_git(monkeypatch: pytest.MonkeyPatch) -> None:
    real_read_csv = pd.read_csv

    def _guarded_read_csv(filepath_or_buffer, *args, **kwargs):  # type: ignore[no-untyped-def]
        path = str(filepath_or_buffer)
        if "Base.csv" in path or "raw/baf" in path:
            raise AssertionError("Stability tests must not read Base.csv.")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _guarded_read_csv)

    real_run = subprocess.run

    def _forbid_git(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokens = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if tokens and Path(str(tokens[0])).name == "git":
            raise AssertionError("Stability tests must not execute Git.")
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
    return bundle, synthetic_config


def _smoke_base() -> XGBoostBaselineConfig:
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
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def test_seed_list_is_fixed_and_unique() -> None:
    assert STABILITY_SEEDS == (42, 43, 44, 45, 46)
    assert validate_stability_seeds(STABILITY_SEEDS) == STABILITY_SEEDS
    with pytest.raises(StabilityError):
        validate_stability_seeds([42, 43, 44, 45])
    with pytest.raises(StabilityError):
        validate_stability_seeds([42, 42, 43, 44, 45])


def test_only_random_state_changes() -> None:
    base = XGBoostBaselineConfig.from_yaml(YAML_PATH)
    assert base.random_state == 42
    for seed in STABILITY_SEEDS:
        modified = config_with_seed(base, seed)
        assert_only_random_state_differs(base, modified, seed)
        assert modified.n_estimators == base.n_estimators
        assert modified.scale_pos_weight == 1
        assert modified.tree_method == "hist"


def test_month7_not_in_stability_modules() -> None:
    for module in (stability_module, stability_cli):
        source = inspect.getsource(module)
        assert 'views["test"]' not in source
        assert "month_7" not in source
        lowered = source.lower()
        assert "month 7" not in lowered or "never" in lowered or "not" in lowered


def test_output_paths_do_not_overwrite_baselines() -> None:
    stability_dir = run_directory(
        XGBOOST_STABILITY_OUTPUT_ROOT, XGBOOST_STABILITY_RUN_ID
    )
    assert_stability_output_isolated(stability_dir)
    with pytest.raises(ArtifactError):
        assert_stability_output_isolated(LOGISTIC_OUTPUT_ROOT / "x")
    with pytest.raises(ArtifactError):
        assert_stability_output_isolated(FROZEN_XGBOOST_BASELINE_DIR)
    with pytest.raises(ArtifactError):
        assert_stability_output_isolated(FROZEN_XGBOOST_BASELINE_DIR / "xgboost")


def test_aggregate_statistics_correct() -> None:
    stats = summarise_numeric([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["sample_std"] == pytest.approx(1.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)


def test_summary_uses_exact_lr_and_does_not_rank_seeds(synthetic_train_dev) -> None:
    if not LR_UNWEIGHTED.is_dir():
        pytest.skip("Saved unweighted LR artefacts unavailable.")
    bundle, config = synthetic_train_dev
    base = _smoke_base()
    rows = []
    for seed in STABILITY_SEEDS:
        result = fit_and_score_xgboost(
            bundle, config_with_seed(base, seed), config, variant_name=f"s{seed}"
        )
        rows.append(metrics_row_from_result(result, seed))
    lr_metrics = load_saved_development_metrics(LR_UNWEIGHTED)
    summary = build_stability_summary(
        rows, lr_metrics=lr_metrics, base_random_state=42
    )
    assert summary["formal_candidate_random_state"] == 42
    assert "best_seed" not in summary
    assert "recommended_seed" not in json.dumps(summary)
    assert set(summary["aggregates"]) == {"auprc", "auroc", "tpr_at_fpr_le_5pct"}
    auprcs = [row["auprc"] for row in rows]
    assert summary["aggregates"]["auprc"]["mean"] == pytest.approx(float(np.mean(auprcs)))
    assert summary["aggregates"]["auprc"]["sample_std"] == pytest.approx(
        float(np.std(auprcs, ddof=1))
    )
    lr_auprc = lr_metrics["threshold_independent"]["auprc"]
    assert lr_auprc == pytest.approx(0.15475964663922073)
    for delta in summary["deltas_versus_unweighted_lr"]:
        seed_row = next(r for r in rows if r["seed"] == delta["seed"])
        assert delta["delta_auprc"] == pytest.approx(seed_row["auprc"] - lr_auprc)


def test_save_stability_artifacts_isolated(tmp_path: Path) -> None:
    rows = [
        {
            "seed": seed,
            "auprc": 0.1 + 0.01 * i,
            "auroc": 0.8,
            "brier_score": 0.01,
            "tpr_at_fpr_le_5pct": 0.5,
            "fpr_at_fpr_le_5pct": 0.05,
            "precision_at_fpr_le_5pct": 0.12,
            "review_rate_at_fpr_le_5pct": 0.055,
            "threshold_at_fpr_le_5pct": 0.05,
            "tp": 1,
            "fp": 1,
            "tn": 1,
            "fn": 1,
            "fit_seconds": 1.0,
            "predict_seconds": 0.1,
            "split": "development",
            "month": 6,
            "random_state": seed,
        }
        for i, seed in enumerate(STABILITY_SEEDS)
    ]
    summary = {
        "aggregates": {
            "auprc": summarise_numeric([r["auprc"] for r in rows]),
            "auroc": summarise_numeric([r["auroc"] for r in rows]),
            "tpr_at_fpr_le_5pct": summarise_numeric(
                [r["tpr_at_fpr_le_5pct"] for r in rows]
            ),
        },
        "deltas_versus_unweighted_lr": [
            {"seed": r["seed"], "delta_auprc": 0.01} for r in rows
        ],
    }
    out = tmp_path / "xgboost_stability" / "run"
    paths = save_stability_artifacts(
        output_dir=out,
        seed_rows=rows,
        summary=summary,
        base_config={"model": {"random_state": 42}},
    )
    assert paths["per_seed_csv"].is_file()
    assert "month7" not in str(paths["per_seed_csv"]).lower()
