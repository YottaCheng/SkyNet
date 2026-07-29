"""Unfitted Logistic Regression baseline pipeline.

Defines a typed configuration object (loadable from YAML) and a builder
that assembles ``preprocessing + LogisticRegression`` into a single
scikit-learn Pipeline. Nothing here calls ``fit``, ``predict`` or
``predict_proba``, and no data file is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_models.preprocessing import build_preprocessor

#: Solvers accepted for the baseline; guards against silent typos in YAML.
SUPPORTED_SOLVERS = ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga")

PREPROCESSING_STEP = "preprocessing"
CLASSIFIER_STEP = "classifier"


@dataclass(frozen=True)
class LogisticBaselineConfig:
    """Typed, immutable configuration for the LR baseline.

    ``class_weight`` is deliberately configurable (``None``, ``"balanced"``
    or an explicit mapping); the shipped default is an a-priori initial
    setting, not an experimentally selected value.
    """

    C: float
    max_iter: int
    solver: str
    random_state: int
    class_weight: str | dict[int, float] | None

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}.")
        if self.solver not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{self.solver}'; expected one of {SUPPORTED_SOLVERS}."
            )
        if not (
            self.class_weight is None
            or self.class_weight == "balanced"
            or isinstance(self.class_weight, dict)
        ):
            raise ValueError(
                "class_weight must be None, 'balanced' or a class->weight mapping; "
                f"got {self.class_weight!r}."
            )

    @classmethod
    def from_yaml(cls, path: Path) -> "LogisticBaselineConfig":
        """Load the configuration from a YAML file with a ``model`` block."""
        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"{path} must contain a top-level 'model' mapping.")
        model = payload["model"]
        class_weight = model.get("class_weight")
        if isinstance(class_weight, dict):
            class_weight = {int(k): float(v) for k, v in class_weight.items()}
        return cls(
            C=float(model["C"]),
            max_iter=int(model["max_iter"]),
            solver=str(model["solver"]),
            random_state=int(model["random_state"]),
            class_weight=class_weight,
        )


def build_logistic_pipeline(
    model_config: LogisticBaselineConfig,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> Pipeline:
    """Assemble the unfitted LR baseline pipeline.

    Returns ``Pipeline([preprocessing, classifier])`` where preprocessing
    is the frozen-schema ColumnTransformer and the classifier is a
    LogisticRegression parameterised entirely by ``model_config``.
    The returned pipeline has not been fitted.
    """
    classifier = LogisticRegression(
        C=model_config.C,
        max_iter=model_config.max_iter,
        solver=model_config.solver,
        random_state=model_config.random_state,
        class_weight=model_config.class_weight,
    )
    return Pipeline(
        steps=[
            (PREPROCESSING_STEP, build_preprocessor(data_config)),
            (CLASSIFIER_STEP, classifier),
        ]
    )
