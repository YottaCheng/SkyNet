"""Bounded one-shot XGBoost improvement challenge (development only).

Internal selection uses months 0–4 fit and month 5 validation with early
stopping. A single selected configuration is then refit on months 0–5 and
evaluated once on month 6. Month 7 is never used.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.views import SplitView, validate_feature_schema
from baf_models.evaluation import DEFAULT_MAX_FPR, evaluate_development_scores
from baf_models.preprocessing import build_preprocessor
from baf_models.training import (
    TrainDevBundle,
    TrainingError,
    VariantFitResult,
    fit_and_score_pipeline,
)
from baf_models.xgboost_model import (
    CLASSIFIER_STEP,
    PREPROCESSING_STEP,
    XGBoostBaselineConfig,
    build_xgboost_classifier,
    build_xgboost_pipeline,
)

logger = logging.getLogger(__name__)

CHALLENGE_SEEDS: tuple[int, ...] = (42, 43, 44)
EARLY_STOPPING_ROUNDS = 50
FORMAL_RANDOM_STATE = 42

#: Absolute mean-AUPRC gap below which TPR tie-break applies.
AUPRC_TIE_EPS = 0.001
#: Absolute mean-TPR gap below which Brier tie-break applies.
TPR_TIE_EPS = 0.001
#: Absolute mean-Brier gap below which complexity tie-break applies.
BRIER_TIE_EPS = 0.001

#: Meaningful month-6 improvement thresholds versus frozen XGBoost seed 42.
IMPROVE_AUPRC_MIN_DELTA = 0.005
IMPROVE_TPR_MAX_DROP = 0.002
IMPROVE_BRIER_MAX_WORSEN = 0.001

#: Stretch challenge targets (not required for a positive decision).
STRETCH_AUPRC = 0.18
STRETCH_TPR = 0.53

SHARED_FIXED: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 1,
}

#: Frozen candidate hyperparameter blocks (excluding shared fixed fields).
CANDIDATE_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "C0_current": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "scale_pos_weight": 1,
    },
    "C1_shallow_regularised": {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 5,
        "scale_pos_weight": 1,
    },
    "C2_medium_regularised": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0,
        "reg_lambda": 5,
        "scale_pos_weight": 1,
    },
    "C3_conservative_weighted": {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0,
        "reg_lambda": 5,
        "scale_pos_weight": 3,
    },
}

CANDIDATE_ORDER: tuple[str, ...] = (
    "C0_current",
    "C1_shallow_regularised",
    "C2_medium_regularised",
    "C3_conservative_weighted",
)


class ChallengeError(RuntimeError):
    """Raised when the bounded challenge protocol is violated."""


@dataclass(frozen=True)
class ChallengeDataBundle:
    """Internal fit/val plus sealed development; no final-test handle."""

    internal_fit: SplitView
    internal_val: SplitView
    development: SplitView
    feature_columns: tuple[str, ...]
    raw_sha256: str


@dataclass(frozen=True)
class InternalSeedResult:
    """One candidate × seed run on month-5 validation."""

    candidate_id: str
    seed: int
    best_iteration: int
    n_trees_at_best: int
    auprc: float
    auroc: float
    brier_score: float
    tpr_at_fpr_le_5pct: float
    precision_at_fpr_le_5pct: float
    review_rate_at_fpr_le_5pct: float
    fpr_at_fpr_le_5pct: float
    threshold_at_fpr_le_5pct: float
    fit_seconds: float
    predict_seconds: float


@dataclass(frozen=True)
class CandidateSummary:
    """Aggregated month-5 metrics for one candidate across challenge seeds."""

    candidate_id: str
    mean_auprc: float
    mean_auroc: float
    mean_brier: float
    mean_tpr_at_fpr_le_5pct: float
    mean_precision_at_fpr_le_5pct: float
    mean_review_rate_at_fpr_le_5pct: float
    mean_fit_seconds: float
    best_iterations: tuple[int, ...]
    n_trees_at_best: tuple[int, ...]
    median_best_iteration: int
    median_n_trees: int
    max_depth: int
    hyperparams: dict[str, Any]


@dataclass(frozen=True)
class SelectionRecord:
    """Frozen-rule selection outcome (not a seed ranking)."""

    selected_candidate_id: str
    reason: str
    median_best_iteration: int
    median_n_trees: int
    comparisons: list[dict[str, Any]]


def validate_challenge_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(s) for s in seeds)
    if resolved != CHALLENGE_SEEDS:
        raise ChallengeError(
            f"Challenge seeds must be exactly {CHALLENGE_SEEDS}; got {resolved}."
        )
    if len(set(resolved)) != len(resolved):
        raise ChallengeError("Challenge seeds must not contain duplicates.")
    return resolved


def build_candidate_config(
    candidate_id: str, *, random_state: int
) -> XGBoostBaselineConfig:
    """Build a typed config for one frozen candidate and seed."""
    if candidate_id not in CANDIDATE_HYPERPARAMS:
        raise ChallengeError(f"Unknown challenge candidate {candidate_id!r}.")
    if candidate_id not in CANDIDATE_ORDER:
        raise ChallengeError(f"Candidate {candidate_id!r} missing from frozen order.")
    payload = {**SHARED_FIXED, **CANDIDATE_HYPERPARAMS[candidate_id]}
    return XGBoostBaselineConfig(
        objective=str(payload["objective"]),
        eval_metric=str(payload["eval_metric"]),
        n_estimators=int(payload["n_estimators"]),
        max_depth=int(payload["max_depth"]),
        learning_rate=float(payload["learning_rate"]),
        min_child_weight=float(payload["min_child_weight"]),
        subsample=float(payload["subsample"]),
        colsample_bytree=float(payload["colsample_bytree"]),
        gamma=float(payload["gamma"]),
        reg_alpha=float(payload["reg_alpha"]),
        reg_lambda=float(payload["reg_lambda"]),
        scale_pos_weight=float(payload["scale_pos_weight"]),
        tree_method=str(payload["tree_method"]),
        random_state=int(random_state),
        n_jobs=int(payload["n_jobs"]),
        verbosity=int(payload["verbosity"]),
    )


def assert_candidates_match_frozen_spec() -> None:
    """Fail fast if the in-code candidate table drifts from the protocol."""
    if tuple(CANDIDATE_HYPERPARAMS) != CANDIDATE_ORDER:
        raise ChallengeError("CANDIDATE_HYPERPARAMS keys must match CANDIDATE_ORDER.")
    expected = {
        "C0_current": (500, 6, 0.05, 1, 0.8, 0.8, 0.0, 0.0, 1.0, 1.0),
        "C1_shallow_regularised": (1000, 4, 0.03, 5, 0.8, 0.8, 0.0, 0.0, 5.0, 1.0),
        "C2_medium_regularised": (1000, 6, 0.03, 5, 0.8, 0.8, 0.1, 0.0, 5.0, 1.0),
        "C3_conservative_weighted": (1000, 4, 0.03, 5, 0.8, 0.8, 0.1, 0.0, 5.0, 3.0),
    }
    for candidate_id, values in expected.items():
        hp = CANDIDATE_HYPERPARAMS[candidate_id]
        observed = (
            int(hp["n_estimators"]),
            int(hp["max_depth"]),
            float(hp["learning_rate"]),
            float(hp["min_child_weight"]),
            float(hp["subsample"]),
            float(hp["colsample_bytree"]),
            float(hp["gamma"]),
            float(hp["reg_alpha"]),
            float(hp["reg_lambda"]),
            float(hp["scale_pos_weight"]),
        )
        if observed != values:
            raise ChallengeError(
                f"Candidate {candidate_id} drifted from frozen spec: "
                f"{observed} != {values}."
            )


def load_challenge_bundle(
    raw_path: Path,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> ChallengeDataBundle:
    """Load data and retain internal fit/val + development only.

    Drops any month-7 / test handle immediately after construction.
    """
    from baf_data import load_prepared_splits

    prepared = load_prepared_splits(raw_path, data_config)
    try:
        frame = prepared.normalised_frame
        train_idx = prepared.indices["train"]
        dev_idx = prepared.indices["dev"]
        months = frame.loc[train_idx, data_config.split_column]
        fit_mask = months.isin((0, 1, 2, 3, 4))
        val_mask = months.isin((5,))
        fit_idx = train_idx[fit_mask]
        val_idx = train_idx[val_mask]
        if len(fit_idx) + len(val_idx) != len(train_idx):
            raise ChallengeError(
                "Internal split of training months did not cover months 0–5 exactly."
            )
        if set(months[fit_mask].unique().tolist()) != {0, 1, 2, 3, 4}:
            raise ChallengeError("Internal fit must contain exactly months 0–4.")
        if set(months[val_mask].unique().tolist()) != {5}:
            raise ChallengeError("Internal validation must contain exactly month 5.")
        if not set(frame.loc[dev_idx, data_config.split_column].unique().tolist()) == {
            6
        }:
            raise ChallengeError("Development view must contain exactly month 6.")

        features = list(data_config.feature_columns)
        target = data_config.target_column

        def _view(name: str, idx: pd.Index) -> SplitView:
            X = frame.loc[idx, features].copy()
            y = frame.loc[idx, target].copy()
            validate_feature_schema(X, data_config)
            return SplitView(name=name, X=X, y=y)

        bundle = ChallengeDataBundle(
            internal_fit=_view("internal_fit", fit_idx),
            internal_val=_view("internal_val", val_idx),
            development=_view("dev", dev_idx),
            feature_columns=data_config.feature_columns,
            raw_sha256=prepared.raw_sha256,
        )
    finally:
        del prepared
    return bundle


def fit_internal_with_early_stopping(
    bundle: ChallengeDataBundle,
    model_config: XGBoostBaselineConfig,
    data_config: DataLayerConfig = FROZEN_CONFIG,
    *,
    candidate_id: str,
    max_fpr: float = DEFAULT_MAX_FPR,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> InternalSeedResult:
    """Fit on months 0–4; early-stop and evaluate on month 5 only."""
    if bundle.internal_fit.name != "internal_fit":
        raise ChallengeError("Internal fit view must be named 'internal_fit'.")
    if bundle.internal_val.name != "internal_val":
        raise ChallengeError("Internal validation view must be named 'internal_val'.")
    if early_stopping_rounds != EARLY_STOPPING_ROUNDS:
        raise ChallengeError(
            f"early_stopping_rounds must be {EARLY_STOPPING_ROUNDS}; "
            f"got {early_stopping_rounds}."
        )

    preprocessor = build_preprocessor(data_config)
    classifier = build_xgboost_classifier(model_config)
    classifier.set_params(early_stopping_rounds=early_stopping_rounds)

    X_fit = bundle.internal_fit.X
    y_fit = bundle.internal_fit.y
    X_val = bundle.internal_val.X
    y_val = bundle.internal_val.y

    logger.info(
        "Internal fit candidate=%s seed=%s X=%s; validate on month 5 X=%s.",
        candidate_id,
        model_config.random_state,
        X_fit.shape,
        X_val.shape,
    )
    t0 = time.perf_counter()
    preprocessor.fit(X_fit, y_fit)
    X_fit_t = preprocessor.transform(X_fit)
    X_val_t = preprocessor.transform(X_val)
    classifier.fit(X_fit_t, y_fit, eval_set=[(X_val_t, y_val)], verbose=False)
    fit_seconds = time.perf_counter() - t0

    best_iteration = int(classifier.best_iteration)
    n_trees = best_iteration + 1
    t1 = time.perf_counter()
    y_score = np.asarray(classifier.predict_proba(X_val_t)[:, 1], dtype=float)
    predict_seconds = time.perf_counter() - t1
    evaluation = evaluate_development_scores(
        y_val.to_numpy(dtype=int), y_score, max_fpr=max_fpr
    )
    ranking = evaluation["threshold_independent"]
    operating = evaluation["operating_point_fpr_le_max"]
    return InternalSeedResult(
        candidate_id=candidate_id,
        seed=int(model_config.random_state),
        best_iteration=best_iteration,
        n_trees_at_best=n_trees,
        auprc=float(ranking["auprc"]),
        auroc=float(ranking["auroc"]),
        brier_score=float(ranking["brier_score"]),
        tpr_at_fpr_le_5pct=float(operating["tpr"]),
        precision_at_fpr_le_5pct=float(operating["precision"]),
        review_rate_at_fpr_le_5pct=float(operating["review_rate"]),
        fpr_at_fpr_le_5pct=float(operating["fpr"]),
        threshold_at_fpr_le_5pct=float(operating["threshold"]),
        fit_seconds=float(fit_seconds),
        predict_seconds=float(predict_seconds),
    )


def summarise_candidate(
    candidate_id: str, seed_results: Sequence[InternalSeedResult]
) -> CandidateSummary:
    """Aggregate three internal seeds for one candidate."""
    if candidate_id not in CANDIDATE_ORDER:
        raise ChallengeError(f"Unknown candidate {candidate_id!r}.")
    rows = [r for r in seed_results if r.candidate_id == candidate_id]
    seeds = tuple(r.seed for r in rows)
    if seeds != CHALLENGE_SEEDS:
        raise ChallengeError(
            f"Candidate {candidate_id} seeds {seeds} != {CHALLENGE_SEEDS}."
        )
    best_iterations = tuple(r.best_iteration for r in rows)
    n_trees = tuple(r.n_trees_at_best for r in rows)
    median_best_iteration = int(np.median(np.asarray(best_iterations, dtype=int)))
    median_n_trees = int(np.median(np.asarray(n_trees, dtype=int)))
    if median_n_trees != median_best_iteration + 1:
        # Median of (bi+1) may differ from median(bi)+1 only on even counts;
        # with three seeds they must agree.
        raise ChallengeError(
            f"Inconsistent median tree count for {candidate_id}: "
            f"median_best_iteration={median_best_iteration}, "
            f"median_n_trees={median_n_trees}."
        )
    hp = dict(CANDIDATE_HYPERPARAMS[candidate_id])
    return CandidateSummary(
        candidate_id=candidate_id,
        mean_auprc=float(np.mean([r.auprc for r in rows])),
        mean_auroc=float(np.mean([r.auroc for r in rows])),
        mean_brier=float(np.mean([r.brier_score for r in rows])),
        mean_tpr_at_fpr_le_5pct=float(np.mean([r.tpr_at_fpr_le_5pct for r in rows])),
        mean_precision_at_fpr_le_5pct=float(
            np.mean([r.precision_at_fpr_le_5pct for r in rows])
        ),
        mean_review_rate_at_fpr_le_5pct=float(
            np.mean([r.review_rate_at_fpr_le_5pct for r in rows])
        ),
        mean_fit_seconds=float(np.mean([r.fit_seconds for r in rows])),
        best_iterations=best_iterations,
        n_trees_at_best=n_trees,
        median_best_iteration=median_best_iteration,
        median_n_trees=median_n_trees,
        max_depth=int(hp["max_depth"]),
        hyperparams=hp,
    )


def select_candidate(summaries: Sequence[CandidateSummary]) -> SelectionRecord:
    """Apply the frozen selection / tie-break rules (no seed picking)."""
    by_id = {s.candidate_id: s for s in summaries}
    if set(by_id) != set(CANDIDATE_ORDER):
        raise ChallengeError("Selection requires exactly candidates C0–C3.")
    ordered = [by_id[cid] for cid in CANDIDATE_ORDER]
    comparisons: list[dict[str, Any]] = []

    best_auprc = max(s.mean_auprc for s in ordered)
    contenders = [
        s for s in ordered if best_auprc - s.mean_auprc < AUPRC_TIE_EPS
    ]
    comparisons.append(
        {
            "step": "mean_auprc",
            "best": best_auprc,
            "contenders": [s.candidate_id for s in contenders],
        }
    )
    if len(contenders) == 1:
        chosen = contenders[0]
        reason = (
            f"Highest mean AUPRC={chosen.mean_auprc:.6f} "
            f"(no rival within {AUPRC_TIE_EPS})."
        )
        return SelectionRecord(
            selected_candidate_id=chosen.candidate_id,
            reason=reason,
            median_best_iteration=chosen.median_best_iteration,
            median_n_trees=chosen.median_n_trees,
            comparisons=comparisons,
        )

    best_tpr = max(s.mean_tpr_at_fpr_le_5pct for s in contenders)
    contenders = [
        s
        for s in contenders
        if best_tpr - s.mean_tpr_at_fpr_le_5pct < TPR_TIE_EPS
    ]
    comparisons.append(
        {
            "step": "mean_tpr",
            "best": best_tpr,
            "contenders": [s.candidate_id for s in contenders],
        }
    )
    if len(contenders) == 1:
        chosen = contenders[0]
        reason = (
            f"AUPRC tie-band then higher mean TPR@FPR<=5%="
            f"{chosen.mean_tpr_at_fpr_le_5pct:.6f}."
        )
        return SelectionRecord(
            selected_candidate_id=chosen.candidate_id,
            reason=reason,
            median_best_iteration=chosen.median_best_iteration,
            median_n_trees=chosen.median_n_trees,
            comparisons=comparisons,
        )

    best_brier = min(s.mean_brier for s in contenders)
    contenders = [
        s for s in contenders if s.mean_brier - best_brier < BRIER_TIE_EPS
    ]
    comparisons.append(
        {
            "step": "mean_brier",
            "best": best_brier,
            "contenders": [s.candidate_id for s in contenders],
        }
    )
    if len(contenders) == 1:
        chosen = contenders[0]
        reason = (
            f"TPR tie-band then lower mean Brier={chosen.mean_brier:.6f}."
        )
        return SelectionRecord(
            selected_candidate_id=chosen.candidate_id,
            reason=reason,
            median_best_iteration=chosen.median_best_iteration,
            median_n_trees=chosen.median_n_trees,
            comparisons=comparisons,
        )

    min_depth = min(s.max_depth for s in contenders)
    depth_contenders = [s for s in contenders if s.max_depth == min_depth]
    chosen = min(
        depth_contenders, key=lambda s: CANDIDATE_ORDER.index(s.candidate_id)
    )
    comparisons.append(
        {
            "step": "complexity",
            "min_depth": min_depth,
            "contenders": [s.candidate_id for s in depth_contenders],
            "chosen": chosen.candidate_id,
        }
    )
    reason = (
        f"Brier tie-band then smaller max_depth={chosen.max_depth} "
        f"(protocol order secondary)."
    )
    return SelectionRecord(
        selected_candidate_id=chosen.candidate_id,
        reason=reason,
        median_best_iteration=chosen.median_best_iteration,
        median_n_trees=chosen.median_n_trees,
        comparisons=comparisons,
    )


def resolve_final_config(
    candidate_id: str, *, median_n_trees: int
) -> XGBoostBaselineConfig:
    """Freeze the selected candidate with median tree count and seed 42."""
    if median_n_trees <= 0:
        raise ChallengeError(f"median_n_trees must be positive; got {median_n_trees}.")
    config = build_candidate_config(candidate_id, random_state=FORMAL_RANDOM_STATE)
    resolved = replace(config, n_estimators=int(median_n_trees))
    if resolved.random_state != FORMAL_RANDOM_STATE:
        raise ChallengeError("Final challenge config must use random_state=42.")
    return resolved


def fit_final_challenge_on_train_dev(
    bundle: ChallengeDataBundle,
    model_config: XGBoostBaselineConfig,
    data_config: DataLayerConfig = FROZEN_CONFIG,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> VariantFitResult:
    """Refit on months 0–5; score month 6 once. No early stopping."""
    if model_config.random_state != FORMAL_RANDOM_STATE:
        raise ChallengeError(
            f"Final refit requires random_state={FORMAL_RANDOM_STATE}; "
            f"got {model_config.random_state}."
        )
    pipeline = build_xgboost_pipeline(model_config, data_config)
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    if classifier.get_params().get("early_stopping_rounds") is not None:
        raise ChallengeError("Final month-6 evaluation must not use early stopping.")

    # Reconstruct full months 0–5 training matrix from internal views.
    X_train = pd.concat(
        [bundle.internal_fit.X, bundle.internal_val.X], axis=0
    )
    y_train = pd.concat(
        [bundle.internal_fit.y, bundle.internal_val.y], axis=0
    )
    if len(X_train) != len(bundle.internal_fit.X) + len(bundle.internal_val.X):
        raise ChallengeError("Failed to reconstruct months 0–5 training matrix.")
    train_dev = TrainDevBundle(
        train=SplitView(name="train", X=X_train, y=y_train),
        development=bundle.development,
        feature_columns=bundle.feature_columns,
        raw_sha256=bundle.raw_sha256,
    )
    if train_dev.development.name != "dev":
        raise ChallengeError("Final evaluation requires development name 'dev'.")

    return fit_and_score_pipeline(
        train_dev,
        "xgboost_challenge_final",
        pipeline,
        model_config,
        max_fpr=max_fpr,
        converged=True,
        n_iter=[],
        fit_log_extra=(
            f"candidate_final n_estimators={model_config.n_estimators}, "
            f"max_depth={model_config.max_depth}, "
            f"scale_pos_weight={model_config.scale_pos_weight}"
        ),
    )


def judge_improvement(
    challenge_eval: Mapping[str, Any],
    frozen_eval: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predefined meaningful-improvement gate."""
    c_rank = challenge_eval["threshold_independent"]
    f_rank = frozen_eval["threshold_independent"]
    c_op = challenge_eval["operating_point_fpr_le_max"]
    f_op = frozen_eval["operating_point_fpr_le_max"]
    delta_auprc = float(c_rank["auprc"]) - float(f_rank["auprc"])
    delta_tpr = float(c_op["tpr"]) - float(f_op["tpr"])
    delta_brier = float(c_rank["brier_score"]) - float(f_rank["brier_score"])
    fpr_ok = float(c_op["fpr"]) <= 0.05 + 1e-12
    meaningful = (
        delta_auprc >= IMPROVE_AUPRC_MIN_DELTA
        and delta_tpr >= -IMPROVE_TPR_MAX_DROP
        and delta_brier <= IMPROVE_BRIER_MAX_WORSEN
        and fpr_ok
    )
    return {
        "meaningful_improvement_candidate": meaningful,
        "delta_auprc": delta_auprc,
        "delta_tpr_at_fpr_le_5pct": delta_tpr,
        "delta_brier_score": delta_brier,
        "challenge_fpr_le_0_05": fpr_ok,
        "thresholds": {
            "min_delta_auprc": IMPROVE_AUPRC_MIN_DELTA,
            "max_tpr_drop": IMPROVE_TPR_MAX_DROP,
            "max_brier_worsen": IMPROVE_BRIER_MAX_WORSEN,
        },
        "stretch_targets": {
            "auprc": STRETCH_AUPRC,
            "tpr_at_fpr_le_5pct": STRETCH_TPR,
            "auprc_met": float(c_rank["auprc"]) >= STRETCH_AUPRC,
            "tpr_met": float(c_op["tpr"]) >= STRETCH_TPR,
        },
        "decision": (
            "adopt_as_improvement_candidate"
            if meaningful
            else "retain_frozen_xgboost"
        ),
        "note": (
            "Month 6 baseline metrics were previously observed by the researcher; "
            "this challenge is a bounded development improvement, not a blind "
            "evaluation. Final claims still require sealed month 7."
        ),
    }


def internal_result_to_dict(result: InternalSeedResult) -> dict[str, Any]:
    return asdict(result)


def candidate_summary_to_dict(summary: CandidateSummary) -> dict[str, Any]:
    return asdict(summary)


def selection_to_dict(selection: SelectionRecord) -> dict[str, Any]:
    return asdict(selection)
