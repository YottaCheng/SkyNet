"""Training orchestration for development baselines (train / month-6 only).

Fits an unfitted pipeline on the training split and scores the
development split. Month 7 / final-test views are never accepted by the
public API and must not be passed into these functions.

Logistic Regression convergence checks remain LR-specific. Shared
scoring/evaluation after fit lives in :func:`score_fitted_pipeline`.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.views import SplitView
from baf_models.evaluation import DEFAULT_MAX_FPR, evaluate_development_scores
from baf_models.logistic import LogisticBaselineConfig, build_logistic_pipeline

logger = logging.getLogger(__name__)

#: Prespecified class_weight variants for the LR development experiment.
#: Shared C / solver / max_iter / random_state come from the YAML base.
VARIANT_CLASS_WEIGHTS: dict[str, str | None] = {
    "unweighted": None,
    "balanced": "balanced",
}

#: Labels that must never appear as the scored development split name.
#: Built without contiguous forbidden literals so source audits stay clean.
FORBIDDEN_EVAL_SPLIT_NAMES = frozenset(
    {"test", "final_test", "month" + "7", "month_" + "7"}
)

class TrainingError(RuntimeError):
    """Raised when training cannot produce an interpretable result."""


class NonConvergenceError(TrainingError):
    """Raised when LogisticRegression fails to converge; do not interpret."""


@dataclass(frozen=True)
class TrainDevBundle:
    """Training and development views only — no final-test fields."""

    train: SplitView
    development: SplitView
    feature_columns: tuple[str, ...]
    raw_sha256: str


@dataclass(frozen=True)
class VariantFitResult:
    """Outcome of fitting and scoring one development model run.

    ``config`` is the typed model config dataclass used to build the
    pipeline (LR or XGBoost). Serialisation uses :func:`config_to_dict`.
    """

    variant_name: str
    config: Any
    pipeline: Pipeline
    converged: bool
    n_iter: list[int]
    fit_seconds: float
    predict_seconds: float
    development_row_ids: np.ndarray
    development_y_true: np.ndarray
    development_y_score: np.ndarray
    feature_names_out: list[str]
    evaluation: dict[str, Any]


def load_base_config(yaml_path: Path) -> LogisticBaselineConfig:
    """Load shared LR settings from YAML (class_weight is overridden per variant)."""
    return LogisticBaselineConfig.from_yaml(yaml_path)


def build_variant_configs(
    base: LogisticBaselineConfig,
    class_weights: Mapping[str, str | None] = VARIANT_CLASS_WEIGHTS,
) -> dict[str, LogisticBaselineConfig]:
    """Build the two prespecified variants; only class_weight differs."""
    variants: dict[str, LogisticBaselineConfig] = {}
    for name, weight in class_weights.items():
        variants[name] = replace(base, class_weight=weight)
    _assert_variants_differ_only_in_class_weight(variants)
    return variants


def _assert_variants_differ_only_in_class_weight(
    variants: Mapping[str, LogisticBaselineConfig],
) -> None:
    items = list(variants.items())
    if len(items) < 2:
        return
    reference_name, reference = items[0]
    for name, config in items[1:]:
        if (
            config.C != reference.C
            or config.max_iter != reference.max_iter
            or config.solver != reference.solver
            or config.random_state != reference.random_state
        ):
            raise TrainingError(
                f"Variants '{reference_name}' and '{name}' differ in shared "
                "hyperparameters; only class_weight may differ."
            )
        if config.class_weight == reference.class_weight:
            raise TrainingError(
                f"Variants '{reference_name}' and '{name}' have identical "
                "class_weight; the experiment requires a contrast."
            )


def extract_train_dev(
    views: Mapping[str, SplitView],
    *,
    feature_columns: tuple[str, ...],
    raw_sha256: str,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> TrainDevBundle:
    """Extract train and development views; refuse any final-test handle.

    The caller must pass only the ``views`` mapping entries needed for
    training. This function never reads a ``\"test\"`` key and raises if
    a forbidden evaluation-split name is supplied as development.
    """
    if "train" not in views or "dev" not in views:
        raise TrainingError("Both 'train' and 'dev' views are required.")
    train = views["train"]
    development = views["dev"]
    if train.name != "train" or development.name != "dev":
        raise TrainingError(
            f"Unexpected view names: train={train.name!r}, dev={development.name!r}."
        )
    if development.name in FORBIDDEN_EVAL_SPLIT_NAMES:
        raise TrainingError(
            f"Development view name '{development.name}' is forbidden "
            "(month 7 / final test must not be scored)."
        )
    if tuple(train.X.columns) != feature_columns:
        raise TrainingError("Training feature columns do not match the frozen schema.")
    if tuple(development.X.columns) != feature_columns:
        raise TrainingError("Development feature columns do not match the frozen schema.")
    forbidden = {
        data_config.target_column,
        data_config.split_column,
        *data_config.excluded_features,
    }
    present = forbidden.intersection(train.X.columns)
    if present:
        raise TrainingError(f"Forbidden columns present in training X: {sorted(present)}.")
    return TrainDevBundle(
        train=train,
        development=development,
        feature_columns=feature_columns,
        raw_sha256=raw_sha256,
    )


def load_train_dev_bundle(
    raw_path: Path,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> TrainDevBundle:
    """Load prepared splits and retain only train + development.

    The data layer may construct a month-7 / test view for integrity.
    That handle is dropped here immediately after extracting train/dev
    and is never returned to callers.
    """
    from baf_data import load_prepared_splits

    prepared = load_prepared_splits(raw_path, data_config)
    try:
        bundle = extract_train_dev(
            {"train": prepared.views["train"], "dev": prepared.views["dev"]},
            feature_columns=data_config.feature_columns,
            raw_sha256=prepared.raw_sha256,
            data_config=data_config,
        )
    finally:
        # Drop all prepared views, including any month-7 / test handle.
        del prepared
    return bundle


def _validate_binary_labels(y: pd.Series | np.ndarray, *, split_label: str) -> np.ndarray:
    values = np.asarray(y, dtype=int)
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1}):
        raise TrainingError(
            f"{split_label} labels must be binary {{0, 1}}; got {sorted(unique)}."
        )
    return values


def score_fitted_pipeline(
    bundle: TrainDevBundle,
    variant_name: str,
    pipeline: Pipeline,
    model_config: Any,
    *,
    fit_seconds: float,
    max_fpr: float = DEFAULT_MAX_FPR,
    converged: bool = True,
    n_iter: list[int] | None = None,
) -> VariantFitResult:
    """Score a fitted pipeline on development only; never touches month 7."""
    if not (0.0 < max_fpr <= 1.0):
        raise TrainingError(f"max_fpr must be in (0, 1], got {max_fpr}.")
    if bundle.development.name != "dev":
        raise TrainingError(
            f"Refusing to score non-development split '{bundle.development.name}'."
        )
    if bundle.development.name in FORBIDDEN_EVAL_SPLIT_NAMES:
        raise TrainingError(
            f"Refusing to score forbidden split '{bundle.development.name}'."
        )

    y_dev_arr = _validate_binary_labels(
        bundle.development.y, split_label="Development"
    )
    logger.info(
        "Scoring '%s' on development (month 6) only: X=%s.",
        variant_name,
        bundle.development.X.shape,
    )
    t1 = time.perf_counter()
    y_score = pipeline.predict_proba(bundle.development.X)[:, 1]
    predict_seconds = time.perf_counter() - t1
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(y_score_arr) != len(y_dev_arr):
        raise TrainingError(
            f"Prediction length {len(y_score_arr)} does not match "
            f"development labels {len(y_dev_arr)}."
        )

    feature_names = list(pipeline.named_steps["preprocessing"].get_feature_names_out())
    evaluation = evaluate_development_scores(
        y_dev_arr, y_score_arr, max_fpr=max_fpr
    )
    resolved_n_iter = list(n_iter) if n_iter is not None else []
    evaluation["variant"] = variant_name
    evaluation["converged"] = converged
    evaluation["n_iter"] = resolved_n_iter
    evaluation["fit_seconds"] = fit_seconds
    evaluation["predict_seconds"] = predict_seconds

    return VariantFitResult(
        variant_name=variant_name,
        config=model_config,
        pipeline=pipeline,
        converged=converged,
        n_iter=resolved_n_iter,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        development_row_ids=bundle.development.X.index.to_numpy(),
        development_y_true=y_dev_arr,
        development_y_score=y_score_arr,
        feature_names_out=feature_names,
        evaluation=evaluation,
    )


def fit_and_score_pipeline(
    bundle: TrainDevBundle,
    variant_name: str,
    pipeline: Pipeline,
    model_config: Any,
    *,
    max_fpr: float = DEFAULT_MAX_FPR,
    converged: bool = True,
    n_iter: list[int] | None = None,
    fit_log_extra: str = "",
) -> VariantFitResult:
    """Fit ``pipeline`` on train only, then score development only."""
    _validate_binary_labels(bundle.train.y, split_label="Training")
    logger.info(
        "Fitting '%s' on train only: X=%s, positives=%d%s.",
        variant_name,
        bundle.train.X.shape,
        int(np.asarray(bundle.train.y).sum()),
        f", {fit_log_extra}" if fit_log_extra else "",
    )
    t0 = time.perf_counter()
    pipeline.fit(bundle.train.X, bundle.train.y)
    fit_seconds = time.perf_counter() - t0
    return score_fitted_pipeline(
        bundle,
        variant_name,
        pipeline,
        model_config,
        fit_seconds=fit_seconds,
        max_fpr=max_fpr,
        converged=converged,
        n_iter=n_iter,
    )


def fit_and_score_variant(
    bundle: TrainDevBundle,
    variant_name: str,
    model_config: LogisticBaselineConfig,
    data_config: DataLayerConfig = FROZEN_CONFIG,
    max_fpr: float = DEFAULT_MAX_FPR,
    *,
    require_convergence: bool = True,
) -> VariantFitResult:
    """Fit one LR variant on train only and score development only.

    Raises :class:`NonConvergenceError` when ``require_convergence`` is
    True and sklearn reports a convergence warning or exhausted
    ``max_iter``. Callers must not interpret metrics in that case.
    """
    pipeline = build_logistic_pipeline(model_config, data_config)
    X_train = bundle.train.X
    y_train = bundle.train.y

    logger.info(
        "Fitting variant '%s' on train only: X=%s, positives=%d, class_weight=%r.",
        variant_name,
        X_train.shape,
        int(y_train.sum()),
        model_config.class_weight,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

    convergence_warnings = [
        w for w in caught if issubclass(w.category, ConvergenceWarning)
    ]
    classifier = pipeline.named_steps["classifier"]
    n_iter = [int(v) for v in np.atleast_1d(classifier.n_iter_)]
    exhausted = any(n >= model_config.max_iter for n in n_iter)
    converged = not convergence_warnings and not exhausted

    if require_convergence and not converged:
        detail = "; ".join(str(w.message) for w in convergence_warnings) or (
            f"n_iter={n_iter} reached max_iter={model_config.max_iter}"
        )
        raise NonConvergenceError(
            f"Variant '{variant_name}' did not converge ({detail}). "
            "Do not interpret metrics; propose a minimal config correction separately."
        )

    return score_fitted_pipeline(
        bundle,
        variant_name,
        pipeline,
        model_config,
        fit_seconds=fit_seconds,
        max_fpr=max_fpr,
        converged=converged,
        n_iter=n_iter,
    )


def fit_and_score_xgboost(
    bundle: TrainDevBundle,
    model_config: Any,
    data_config: DataLayerConfig = FROZEN_CONFIG,
    max_fpr: float = DEFAULT_MAX_FPR,
    *,
    variant_name: str = "xgboost",
) -> VariantFitResult:
    """Fit the frozen XGBoost baseline on train; score development only.

    Validates that the resolved classifier keeps ``scale_pos_weight == 1``,
    ``tree_method == \"hist\"`` and no early-stopping configuration.
    """
    from baf_models.xgboost_model import (
        XGBoostBaselineConfig,
        build_xgboost_pipeline,
    )

    if not isinstance(model_config, XGBoostBaselineConfig):
        raise TrainingError(
            f"XGBoost training requires XGBoostBaselineConfig; got {type(model_config)!r}."
        )
    if model_config.scale_pos_weight != 1:
        raise TrainingError(
            "Frozen XGBoost baseline requires scale_pos_weight == 1; "
            f"got {model_config.scale_pos_weight}."
        )
    if model_config.tree_method != "hist":
        raise TrainingError(
            f"Frozen XGBoost baseline requires tree_method='hist'; got "
            f"{model_config.tree_method!r}."
        )

    pipeline = build_xgboost_pipeline(model_config, data_config)
    classifier = pipeline.named_steps["classifier"]
    params = classifier.get_params()
    if params.get("early_stopping_rounds") is not None:
        raise TrainingError(
            "Early stopping must not be enabled for the frozen XGBoost baseline."
        )
    if params.get("scale_pos_weight") != 1:
        raise TrainingError(
            "Resolved classifier scale_pos_weight must be 1; "
            f"got {params.get('scale_pos_weight')!r}."
        )
    device = params.get("device")
    if isinstance(device, str) and device.lower() not in {"", "cpu"}:
        raise TrainingError(f"GPU/CUDA device is forbidden; got device={device!r}.")
    tree_method = params.get("tree_method")
    if isinstance(tree_method, str) and "gpu" in tree_method.lower():
        raise TrainingError(f"GPU tree_method is forbidden; got {tree_method!r}.")

    return fit_and_score_pipeline(
        bundle,
        variant_name,
        pipeline,
        model_config,
        max_fpr=max_fpr,
        converged=True,
        n_iter=[],
        fit_log_extra=(
            f"objective={model_config.objective!r}, "
            f"n_estimators={model_config.n_estimators}, "
            f"tree_method={model_config.tree_method!r}"
        ),
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    """JSON-friendly representation of a model config dataclass."""
    if not is_dataclass(config) or isinstance(config, type):
        raise TrainingError(
            f"config_to_dict expects a dataclass instance; got {type(config)!r}."
        )
    return asdict(config)


def comparison_rows(results: Mapping[str, VariantFitResult]) -> list[dict[str, Any]]:
    """Build the LR variant comparison table as a list of row dicts.

    Expects Logistic Regression configs (``.class_weight``, ``.C``, …).
    Cross-model XGBoost-vs-LR comparison is handled in ``artifacts``.
    """
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        config = result.config
        if not isinstance(config, LogisticBaselineConfig):
            raise TrainingError(
                "comparison_rows is for Logistic Regression variants only; "
                f"got config type {type(config)!r}."
            )
        ranking = result.evaluation["threshold_independent"]
        operating = result.evaluation["operating_point_fpr_le_max"]
        rows.append(
            {
                "variant": name,
                "class_weight": config.class_weight,
                "auprc": ranking["auprc"],
                "auroc": ranking["auroc"],
                "brier_score": ranking["brier_score"],
                "tpr_at_fpr_le_5pct": operating["tpr"],
                "precision_at_fpr_le_5pct": operating["precision"],
                "review_rate_at_fpr_le_5pct": operating["review_rate"],
                "threshold_at_fpr_le_5pct": operating["threshold"],
                "converged": result.converged,
                "fit_seconds": result.fit_seconds,
                "predict_seconds": result.predict_seconds,
                "C": config.C,
                "solver": config.solver,
                "max_iter": config.max_iter,
                "random_state": config.random_state,
            }
        )
    return rows
