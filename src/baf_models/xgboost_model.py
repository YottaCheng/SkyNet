"""Unfitted XGBoost baseline pipeline.

Defines a typed configuration object (loadable from YAML) and a builder
that assembles ``preprocessing + XGBClassifier`` into a single
scikit-learn Pipeline. Feature columns come exclusively from
:data:`baf_data.config.FROZEN_CONFIG` via the shared preprocessor.
Nothing here calls ``fit``, ``predict`` or ``predict_proba``, and no
data file is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_models.preprocessing import build_preprocessor

PREPROCESSING_STEP = "preprocessing"
CLASSIFIER_STEP = "classifier"


@dataclass(frozen=True)
class XGBoostBaselineConfig:
    """Typed, immutable configuration for the XGBoost baseline.

    Values mirror ``config/xgboost_baseline.yaml``. They are a-priori
    starting settings, not development-selected hyperparameters.
    """

    objective: str
    eval_metric: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    gamma: float
    reg_alpha: float
    reg_lambda: float
    scale_pos_weight: float
    tree_method: str
    random_state: int
    n_jobs: int
    verbosity: int

    def __post_init__(self) -> None:
        if self.objective != "binary:logistic":
            raise ValueError(
                f"objective must be 'binary:logistic', got {self.objective!r}."
            )
        if self.eval_metric != "aucpr":
            raise ValueError(f"eval_metric must be 'aucpr', got {self.eval_metric!r}.")
        if self.tree_method != "hist":
            raise ValueError(
                f"tree_method must be 'hist' (CPU); got {self.tree_method!r}."
            )
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {self.n_estimators}.")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {self.max_depth}.")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}."
            )
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample}.")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError(
                f"colsample_bytree must be in (0, 1], got {self.colsample_bytree}."
            )
        if self.scale_pos_weight <= 0:
            raise ValueError(
                f"scale_pos_weight must be positive, got {self.scale_pos_weight}."
            )
        if self.min_child_weight < 0:
            raise ValueError(
                f"min_child_weight must be non-negative, got {self.min_child_weight}."
            )
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}.")
        if self.reg_alpha < 0:
            raise ValueError(f"reg_alpha must be non-negative, got {self.reg_alpha}.")
        if self.reg_lambda < 0:
            raise ValueError(f"reg_lambda must be non-negative, got {self.reg_lambda}.")
        if self.verbosity < 0:
            raise ValueError(f"verbosity must be non-negative, got {self.verbosity}.")

    @classmethod
    def from_yaml(cls, path: Path) -> "XGBoostBaselineConfig":
        """Load the configuration from a YAML file with a ``model`` block."""
        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"{path} must contain a top-level 'model' mapping.")
        model = payload["model"]
        return cls(
            objective=str(model["objective"]),
            eval_metric=str(model["eval_metric"]),
            n_estimators=int(model["n_estimators"]),
            max_depth=int(model["max_depth"]),
            learning_rate=float(model["learning_rate"]),
            min_child_weight=float(model["min_child_weight"]),
            subsample=float(model["subsample"]),
            colsample_bytree=float(model["colsample_bytree"]),
            gamma=float(model["gamma"]),
            reg_alpha=float(model["reg_alpha"]),
            reg_lambda=float(model["reg_lambda"]),
            scale_pos_weight=float(model["scale_pos_weight"]),
            tree_method=str(model["tree_method"]),
            random_state=int(model["random_state"]),
            n_jobs=int(model["n_jobs"]),
            verbosity=int(model["verbosity"]),
        )


def build_xgboost_classifier(model_config: XGBoostBaselineConfig) -> XGBClassifier:
    """Construct an unfitted ``XGBClassifier`` from the typed config."""
    return XGBClassifier(
        objective=model_config.objective,
        eval_metric=model_config.eval_metric,
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        learning_rate=model_config.learning_rate,
        min_child_weight=model_config.min_child_weight,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        gamma=model_config.gamma,
        reg_alpha=model_config.reg_alpha,
        reg_lambda=model_config.reg_lambda,
        scale_pos_weight=model_config.scale_pos_weight,
        tree_method=model_config.tree_method,
        random_state=model_config.random_state,
        n_jobs=model_config.n_jobs,
        verbosity=model_config.verbosity,
    )


def build_xgboost_pipeline(
    model_config: XGBoostBaselineConfig,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> Pipeline:
    """Assemble the unfitted XGBoost baseline pipeline.

    Returns ``Pipeline([preprocessing, classifier])`` where preprocessing
    is the shared frozen-schema ColumnTransformer (identical input
    representation to the Logistic Regression baseline) and the
    classifier is an ``XGBClassifier`` parameterised entirely by
    ``model_config``. The returned pipeline has not been fitted.
    """
    return Pipeline(
        steps=[
            (PREPROCESSING_STEP, build_preprocessor(data_config)),
            (CLASSIFIER_STEP, build_xgboost_classifier(model_config)),
        ]
    )
