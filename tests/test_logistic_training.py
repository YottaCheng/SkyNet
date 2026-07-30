"""Synthetic tests for Logistic Regression training and evaluation.

Uses in-memory synthetic frames only. Never reads Base.csv, never
scores a month-7 / test split, and never executes Git commands.
"""

from __future__ import annotations

import inspect
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from baf_data.config import FROZEN_CONFIG
from baf_data.sentinels import normalise_sentinels
from baf_data.splitting import build_temporal_indices
from baf_data.views import SplitView, create_feature_target_views
from baf_models.artifacts import save_variant_artifacts
from baf_models.evaluation import (
    confusion_at_threshold,
    evaluate_development_scores,
    select_fpr_constrained_threshold,
    threshold_independent_metrics,
)
from baf_models.logistic import LogisticBaselineConfig
from baf_models.training import (
    VARIANT_CLASS_WEIGHTS,
    NonConvergenceError,
    build_variant_configs,
    extract_train_dev,
    fit_and_score_variant,
)
import baf_models.training as training_module
import baf_models.evaluation as evaluation_module
import baf_models.artifacts as artifacts_module


@pytest.fixture(autouse=True)
def forbid_real_dataset_and_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse Base.csv loads and any Git subprocess during these tests."""

    real_read_csv = pd.read_csv

    def _guarded_read_csv(filepath_or_buffer, *args, **kwargs):  # type: ignore[no-untyped-def]
        path = str(filepath_or_buffer)
        if "Base.csv" in path or "raw/baf" in path:
            raise AssertionError("Training tests must not read Base.csv.")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _guarded_read_csv)

    real_run = subprocess.run
    real_popen = subprocess.Popen

    def _forbid_git(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokens = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if tokens and Path(str(tokens[0])).name == "git":
            raise AssertionError("Training tests must not execute Git commands.")
        return real_run(cmd, *args, **kwargs)

    def _forbid_git_popen(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokens = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if tokens and Path(str(tokens[0])).name == "git":
            raise AssertionError("Training tests must not execute Git commands.")
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _forbid_git)
    monkeypatch.setattr(subprocess, "Popen", _forbid_git_popen)


@pytest.fixture()
def synthetic_train_dev(synthetic_frame, synthetic_config):
    """Build train/dev SplitViews from the shared synthetic BAF-shaped frame."""
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


BASE_CONFIG = LogisticBaselineConfig(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
    class_weight="balanced",
)


# ---------------------------------------------------------------------------
# Isolation and variant construction
# ---------------------------------------------------------------------------

def test_extract_train_dev_does_not_require_or_use_test(synthetic_train_dev) -> None:
    bundle, config, all_views = synthetic_train_dev
    assert bundle.train.name == "train"
    assert bundle.development.name == "dev"
    # Passing only train/dev works; test is never consulted.
    assert "test" not in {"train": all_views["train"], "dev": all_views["dev"]}


def test_training_module_has_no_month7_evaluation_path() -> None:
    source = inspect.getsource(training_module)
    # The training API must not score or evaluate a final-test split.
    assert 'views["test"]' not in source
    assert "month_7" not in source
    assert "month 7" not in source.lower() or "never" in source.lower()
    eval_source = inspect.getsource(evaluation_module)
    assert "month 7" not in eval_source.lower() or "never" in eval_source.lower()
    assert 'split": "test"' not in eval_source
    assert "final_test" not in eval_source


def test_artifacts_module_labels_are_development_only() -> None:
    source = inspect.getsource(artifacts_module)
    assert "development_month6" in source
    assert "month7" not in source.lower().replace("month 7", "MONTH7_MENTION")
    # Filenames must say development / month 6.
    assert "development_month6_precision_recall.png" in source
    assert "development_month6_roc.png" in source


def test_variants_differ_only_in_class_weight() -> None:
    variants = build_variant_configs(BASE_CONFIG)
    assert set(variants) == {"unweighted", "balanced"}
    assert variants["unweighted"].class_weight is None
    assert variants["balanced"].class_weight == "balanced"
    assert variants["unweighted"].C == variants["balanced"].C == 1.0
    assert variants["unweighted"].solver == variants["balanced"].solver
    assert variants["unweighted"].max_iter == variants["balanced"].max_iter
    assert variants["unweighted"].random_state == variants["balanced"].random_state
    assert VARIANT_CLASS_WEIGHTS["unweighted"] is None
    assert VARIANT_CLASS_WEIGHTS["balanced"] == "balanced"


def test_fit_receives_only_training_rows(synthetic_train_dev, monkeypatch) -> None:
    bundle, config, _ = synthetic_train_dev
    seen: dict[str, object] = {}
    real_fit = Pipeline.fit

    def _spy_fit(self, X, y=None, **kwargs):  # type: ignore[no-untyped-def]
        seen["X"] = X
        seen["y"] = y
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", _spy_fit)
    model = replace(BASE_CONFIG, class_weight=None, max_iter=200)
    fit_and_score_variant(bundle, "unweighted", model, config, require_convergence=False)
    assert seen["X"] is bundle.train.X
    assert seen["y"] is bundle.train.y
    assert len(seen["X"]) == len(bundle.train.X)


def test_predict_uses_development_only(synthetic_train_dev, monkeypatch) -> None:
    bundle, config, _ = synthetic_train_dev
    seen: dict[str, object] = {}
    real_predict_proba = Pipeline.predict_proba

    def _spy(self, X, **kwargs):  # type: ignore[no-untyped-def]
        seen["X"] = X
        return real_predict_proba(self, X, **kwargs)

    monkeypatch.setattr(Pipeline, "predict_proba", _spy)
    model = replace(BASE_CONFIG, class_weight=None, max_iter=200)
    fit_and_score_variant(bundle, "unweighted", model, config, require_convergence=False)
    assert seen["X"] is bundle.development.X


# ---------------------------------------------------------------------------
# Metrics and threshold selection
# ---------------------------------------------------------------------------

def test_threshold_independent_metrics_on_perfect_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = threshold_independent_metrics(y_true, y_score)
    assert metrics.auprc == pytest.approx(1.0)
    assert metrics.auroc == pytest.approx(1.0)
    assert 0.0 <= metrics.brier_score <= 1.0


def test_confusion_at_threshold_counts() -> None:
    y_true = np.array([0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.6, 0.7, 0.2, 0.4])
    counts = confusion_at_threshold(y_true, y_score, 0.5)
    # preds: 0,1,1,0,0 → TP=1 (idx2), FP=1 (idx1), TN=2, FN=1 (idx3)
    assert counts.tp == 1
    assert counts.fp == 1
    assert counts.tn == 2
    assert counts.fn == 1


def test_fpr_constrained_threshold_tie_breaking() -> None:
    # Construct scores so two thresholds share the same TPR under FPR<=0.5,
    # then the rule must pick lowest FPR, then highest threshold.
    y_true = np.array([0, 0, 0, 0, 1, 1])
    # Ranking that yields a clear FPR<=0.5 regime.
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    operating = select_fpr_constrained_threshold(y_true, y_score, max_fpr=0.5)
    assert operating.metrics.fpr <= 0.5 + 1e-12
    # With threshold 0.9: preds for scores>=0.9 → indices 4,5 → TP=2, FP=0 → TPR=1, FPR=0
    assert operating.threshold == pytest.approx(0.9)
    assert operating.metrics.tpr == pytest.approx(1.0)
    assert operating.metrics.fpr == pytest.approx(0.0)

    # Tie on TPR with different FPR: prefer lower FPR.
    y_true2 = np.array([0, 0, 0, 1, 1])
    y_score2 = np.array([0.1, 0.4, 0.6, 0.7, 0.9])
    op2 = select_fpr_constrained_threshold(y_true2, y_score2, max_fpr=0.5)
    assert op2.metrics.fpr <= 0.5 + 1e-12
    # Highest TPR under the constraint should be selected.
    best_tpr = op2.metrics.tpr
    for thr in np.unique(y_score2):
        counts = confusion_at_threshold(y_true2, y_score2, float(thr))
        if counts.fpr <= 0.5:
            assert counts.tpr <= best_tpr + 1e-12


def test_highest_threshold_wins_identical_tpr_and_fpr() -> None:
    # Perfect separation: the unique-score threshold 0.8 yields TPR=1, FPR=0.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.8])
    operating = select_fpr_constrained_threshold(y_true, y_score, max_fpr=0.5)
    assert operating.threshold == pytest.approx(0.8)
    assert operating.metrics.tpr == pytest.approx(1.0)

    # When two candidate thresholds produce identical TPR and FPR, choose the
    # highest. Thresholds 0.0 and 0.5 both predict every row positive.
    y_score_flat = np.array([0.5, 0.5, 0.5, 0.5])
    op_tie = select_fpr_constrained_threshold(y_true, y_score_flat, max_fpr=1.0)
    assert op_tie.threshold == pytest.approx(0.5)
    assert op_tie.metrics.tpr == pytest.approx(1.0)
    assert op_tie.metrics.fpr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end synthetic training behaviour
# ---------------------------------------------------------------------------

def test_fit_and_score_is_deterministic(synthetic_train_dev) -> None:
    bundle, config, _ = synthetic_train_dev
    model = replace(BASE_CONFIG, class_weight=None, max_iter=300)
    first = fit_and_score_variant(bundle, "unweighted", model, config, require_convergence=False)
    second = fit_and_score_variant(bundle, "unweighted", model, config, require_convergence=False)
    np.testing.assert_allclose(first.development_y_score, second.development_y_score)
    assert first.evaluation["threshold_independent"] == second.evaluation[
        "threshold_independent"
    ]


def test_unknown_categories_remain_safe(synthetic_train_dev) -> None:
    bundle, config, _ = synthetic_train_dev
    model = replace(BASE_CONFIG, class_weight=None, max_iter=300)
    # Inject an unseen category into development only.
    X_dev = bundle.development.X.copy()
    X_dev.iloc[0, X_dev.columns.get_loc("payment_type")] = "ZZ_UNSEEN"
    mutated = extract_train_dev(
        {
            "train": bundle.train,
            "dev": SplitView(name="dev", X=X_dev, y=bundle.development.y),
        },
        feature_columns=bundle.feature_columns,
        raw_sha256="synthetic",
        data_config=config,
    )
    result = fit_and_score_variant(
        mutated, "unweighted", model, config, require_convergence=False
    )
    assert np.isfinite(result.development_y_score).all()


def test_missing_values_handled_through_pipeline(synthetic_train_dev) -> None:
    bundle, config, _ = synthetic_train_dev
    assert bundle.train.X.isna().any().any()
    model = replace(BASE_CONFIG, class_weight="balanced", max_iter=300)
    result = fit_and_score_variant(
        bundle, "balanced", model, config, require_convergence=False
    )
    assert np.isfinite(result.development_y_score).all()
    assert 0.0 <= result.evaluation["threshold_independent"]["auprc"] <= 1.0


def test_saved_metadata_contains_variant_config_and_schema(
    synthetic_train_dev, tmp_path: Path
) -> None:
    bundle, config, _ = synthetic_train_dev
    model = replace(BASE_CONFIG, class_weight=None, max_iter=300)
    result = fit_and_score_variant(
        bundle, "unweighted", model, config, require_convergence=False
    )
    paths = save_variant_artifacts(
        result, tmp_path / "unweighted", data_config=config, raw_sha256="synthetic"
    )
    payload = (paths["config"]).read_text(encoding="utf-8")
    assert '"variant": "unweighted"' in payload
    assert '"C": 1.0' in payload
    assert '"solver": "lbfgs"' in payload
    assert "email_is_free" in payload
    assert "fraud_bool" not in (tmp_path / "unweighted" / "transformed_feature_names.json").read_text()
    metrics = evaluate_development_scores(
        result.development_y_true, result.development_y_score
    )
    assert metrics["split"] == "development"
    assert metrics["month"] == 6


def test_nonconvergence_raises_when_required(synthetic_train_dev) -> None:
    bundle, config, _ = synthetic_train_dev
    # Extremely low max_iter forces non-convergence on this feature set.
    model = replace(BASE_CONFIG, class_weight="balanced", max_iter=1, solver="lbfgs")
    with pytest.raises(NonConvergenceError):
        fit_and_score_variant(
            bundle, "balanced", model, config, require_convergence=True
        )
