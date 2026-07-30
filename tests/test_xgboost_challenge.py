"""Tests for the bounded XGBoost improvement challenge protocol."""

from __future__ import annotations

import inspect
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baf_data.config import FROZEN_CONFIG
from baf_data.sentinels import normalise_sentinels
from baf_data.splitting import build_temporal_indices
from baf_data.views import SplitView
from baf_models.artifacts import (
    FROZEN_XGBOOST_BASELINE_DIR,
    FROZEN_XGBOOST_STABILITY_DIR,
    LOGISTIC_OUTPUT_ROOT,
    XGBOOST_CHALLENGE_OUTPUT_ROOT,
    XGBOOST_CHALLENGE_RUN_ID,
    ArtifactError,
    assert_challenge_output_isolated,
    run_directory,
)
from baf_models.challenge import (
    CANDIDATE_HYPERPARAMS,
    CANDIDATE_ORDER,
    CHALLENGE_SEEDS,
    FORMAL_RANDOM_STATE,
    CandidateSummary,
    ChallengeDataBundle,
    ChallengeError,
    assert_candidates_match_frozen_spec,
    build_candidate_config,
    fit_final_challenge_on_train_dev,
    fit_internal_with_early_stopping,
    resolve_final_config,
    select_candidate,
    summarise_candidate,
    validate_challenge_seeds,
)
import baf_models.challenge as challenge_module
import run_xgboost_challenge as challenge_cli


@pytest.fixture(autouse=True)
def forbid_real_dataset_and_git(monkeypatch: pytest.MonkeyPatch) -> None:
    real_read_csv = pd.read_csv

    def _guarded_read_csv(filepath_or_buffer, *args, **kwargs):  # type: ignore[no-untyped-def]
        path = str(filepath_or_buffer)
        if "Base.csv" in path or "raw/baf" in path:
            raise AssertionError("Challenge tests must not read Base.csv.")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _guarded_read_csv)

    real_run = subprocess.run

    def _forbid_git(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tokens = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if tokens and Path(str(tokens[0])).name == "git":
            raise AssertionError("Challenge tests must not execute Git.")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _forbid_git)


@pytest.fixture()
def synthetic_challenge_bundle(synthetic_frame, synthetic_config) -> ChallengeDataBundle:
    normalised, _ = normalise_sentinels(synthetic_frame, synthetic_config.sentinel_rules)
    indices = build_temporal_indices(normalised, synthetic_config)
    train_idx = indices["train"]
    months = normalised.loc[train_idx, synthetic_config.split_column]
    fit_idx = train_idx[months.isin((0, 1, 2, 3, 4))]
    val_idx = train_idx[months.isin((5,))]
    features = list(synthetic_config.feature_columns)
    target = synthetic_config.target_column

    def _view(name: str, idx) -> SplitView:
        return SplitView(
            name=name,
            X=normalised.loc[idx, features].copy(),
            y=normalised.loc[idx, target].copy(),
        )

    return ChallengeDataBundle(
        internal_fit=_view("internal_fit", fit_idx),
        internal_val=_view("internal_val", val_idx),
        development=_view("dev", indices["dev"]),
        feature_columns=synthetic_config.feature_columns,
        raw_sha256="synthetic",
    )


def test_candidates_and_seeds_are_frozen() -> None:
    assert_candidates_match_frozen_spec()
    assert CANDIDATE_ORDER == (
        "C0_current",
        "C1_shallow_regularised",
        "C2_medium_regularised",
        "C3_conservative_weighted",
    )
    assert validate_challenge_seeds(CHALLENGE_SEEDS) == (42, 43, 44)
    assert CANDIDATE_HYPERPARAMS["C3_conservative_weighted"]["scale_pos_weight"] == 3
    with pytest.raises(ChallengeError):
        validate_challenge_seeds([42, 43])


def test_internal_fit_and_val_months_only(synthetic_challenge_bundle, monkeypatch) -> None:
    bundle = synthetic_challenge_bundle
    assert set(range(5)).issuperset({0})  # sanity
    seen_fit = {}
    seen_val_predict = {}

    real_fit = None
    from xgboost import XGBClassifier

    real_xgb_fit = XGBClassifier.fit
    real_predict = XGBClassifier.predict_proba

    def _spy_fit(self, X, y=None, **kwargs):  # type: ignore[no-untyped-def]
        seen_fit["n"] = len(X)
        assert "eval_set" in kwargs
        eval_x, eval_y = kwargs["eval_set"][0]
        seen_val_predict["n_eval"] = len(eval_x)
        return real_xgb_fit(self, X, y, **kwargs)

    def _spy_predict(self, X, **kwargs):  # type: ignore[no-untyped-def]
        seen_val_predict["n_predict"] = len(X)
        return real_predict(self, X, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", _spy_fit)
    monkeypatch.setattr(XGBClassifier, "predict_proba", _spy_predict)

    config = build_candidate_config("C0_current", random_state=42)
    config = replace(config, n_estimators=20, verbosity=0)
    fit_internal_with_early_stopping(
        bundle, config, FROZEN_CONFIG, candidate_id="C0_current"
    )
    assert seen_fit["n"] == len(bundle.internal_fit.X)
    assert seen_val_predict["n_eval"] == len(bundle.internal_val.X)
    assert seen_val_predict["n_predict"] == len(bundle.internal_val.X)
    assert seen_fit["n"] != len(bundle.development.X)


def test_final_refit_uses_months_0_to_5(synthetic_challenge_bundle, monkeypatch) -> None:
    from sklearn.pipeline import Pipeline

    bundle = synthetic_challenge_bundle
    seen = {}
    real_fit = Pipeline.fit

    def _spy(self, X, y=None, **kwargs):  # type: ignore[no-untyped-def]
        seen["n"] = len(X)
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", _spy)
    resolved = resolve_final_config("C0_current", median_n_trees=10)
    resolved = replace(resolved, verbosity=0)
    result = fit_final_challenge_on_train_dev(bundle, resolved, FROZEN_CONFIG)
    assert seen["n"] == len(bundle.internal_fit.X) + len(bundle.internal_val.X)
    assert result.evaluation["month"] == 6
    assert result.evaluation["split"] == "development"


def test_month7_absent_from_challenge_sources() -> None:
    for module in (challenge_module, challenge_cli):
        source = inspect.getsource(module)
        assert 'views["test"]' not in source
        assert "month_7" not in source


def test_selection_rules_and_median() -> None:
    def _summary(
        candidate_id: str,
        mean_auprc: float,
        mean_tpr: float,
        mean_brier: float,
        max_depth: int,
        best_iterations: tuple[int, ...],
    ) -> CandidateSummary:
        n_trees = tuple(i + 1 for i in best_iterations)
        return CandidateSummary(
            candidate_id=candidate_id,
            mean_auprc=mean_auprc,
            mean_auroc=0.8,
            mean_brier=mean_brier,
            mean_tpr_at_fpr_le_5pct=mean_tpr,
            mean_precision_at_fpr_le_5pct=0.1,
            mean_review_rate_at_fpr_le_5pct=0.05,
            mean_fit_seconds=1.0,
            best_iterations=best_iterations,
            n_trees_at_best=n_trees,
            median_best_iteration=int(np.median(best_iterations)),
            median_n_trees=int(np.median(n_trees)),
            max_depth=max_depth,
            hyperparams=dict(CANDIDATE_HYPERPARAMS[candidate_id]),
        )

    # Clear AUPRC winner.
    summaries = [
        _summary("C0_current", 0.10, 0.50, 0.02, 6, (10, 20, 30)),
        _summary("C1_shallow_regularised", 0.12, 0.49, 0.02, 4, (11, 21, 31)),
        _summary("C2_medium_regularised", 0.11, 0.51, 0.02, 6, (12, 22, 32)),
        _summary("C3_conservative_weighted", 0.105, 0.52, 0.02, 4, (13, 23, 33)),
    ]
    selection = select_candidate(summaries)
    assert selection.selected_candidate_id == "C1_shallow_regularised"
    assert selection.median_n_trees == 22

    # AUPRC tie-band -> higher TPR.
    summaries = [
        _summary("C0_current", 0.120, 0.50, 0.02, 6, (10, 20, 30)),
        _summary("C1_shallow_regularised", 0.1205, 0.51, 0.02, 4, (10, 20, 30)),
        _summary("C2_medium_regularised", 0.1198, 0.49, 0.02, 6, (10, 20, 30)),
        _summary("C3_conservative_weighted", 0.110, 0.55, 0.02, 4, (10, 20, 30)),
    ]
    selection = select_candidate(summaries)
    assert selection.selected_candidate_id == "C1_shallow_regularised"

    # Complexity: prefer smaller depth.
    summaries = [
        _summary("C0_current", 0.12, 0.50, 0.012, 6, (10, 20, 30)),
        _summary("C1_shallow_regularised", 0.12, 0.50, 0.012, 4, (10, 20, 30)),
        _summary("C2_medium_regularised", 0.12, 0.50, 0.012, 6, (10, 20, 30)),
        _summary("C3_conservative_weighted", 0.12, 0.50, 0.012, 4, (10, 20, 30)),
    ]
    selection = select_candidate(summaries)
    assert selection.selected_candidate_id == "C1_shallow_regularised"


def test_final_seed_fixed_and_no_month6_retune() -> None:
    resolved = resolve_final_config("C2_medium_regularised", median_n_trees=123)
    assert resolved.random_state == FORMAL_RANDOM_STATE == 42
    assert resolved.n_estimators == 123
    source = inspect.getsource(challenge_cli.main)
    assert "GridSearch" not in source
    assert "Optuna" not in source
    # Month 6 is evaluated once after selection; no loop over candidates on month 6.
    assert source.count("fit_final_challenge_on_train_dev(") == 1


def test_output_isolation() -> None:
    challenge_dir = run_directory(
        XGBOOST_CHALLENGE_OUTPUT_ROOT, XGBOOST_CHALLENGE_RUN_ID
    )
    assert_challenge_output_isolated(challenge_dir)
    with pytest.raises(ArtifactError):
        assert_challenge_output_isolated(LOGISTIC_OUTPUT_ROOT / "x")
    with pytest.raises(ArtifactError):
        assert_challenge_output_isolated(FROZEN_XGBOOST_BASELINE_DIR)
    with pytest.raises(ArtifactError):
        assert_challenge_output_isolated(FROZEN_XGBOOST_STABILITY_DIR)


def test_end_to_end_synthetic_internal_and_final(synthetic_challenge_bundle) -> None:
    bundle = synthetic_challenge_bundle
    rows = []
    for candidate_id in CANDIDATE_ORDER:
        for seed in CHALLENGE_SEEDS:
            config = build_candidate_config(candidate_id, random_state=seed)
            config = replace(config, n_estimators=30, verbosity=0)
            # Keep early stopping rounds at protocol value but tiny data may stop early.
            rows.append(
                fit_internal_with_early_stopping(
                    bundle, config, FROZEN_CONFIG, candidate_id=candidate_id
                )
            )
    summaries = [
        summarise_candidate(cid, [r for r in rows if r.candidate_id == cid])
        for cid in CANDIDATE_ORDER
    ]
    selection = select_candidate(summaries)
    resolved = resolve_final_config(
        selection.selected_candidate_id, median_n_trees=selection.median_n_trees
    )
    resolved = replace(resolved, verbosity=0)
    final = fit_final_challenge_on_train_dev(bundle, resolved, FROZEN_CONFIG)
    assert final.evaluation["month"] == 6
    assert 0.0 <= final.evaluation["threshold_independent"]["auprc"] <= 1.0
