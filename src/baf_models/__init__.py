"""Statistical model package for the BAF experiments.

Provides the unfitted Logistic Regression and XGBoost scaffolds,
development-set evaluation helpers and training orchestration. Feature
lists are always derived from :mod:`baf_data.config`.
"""

from baf_models.evaluation import (
    OperatingPoint,
    select_fpr_constrained_threshold,
    threshold_independent_metrics,
)
from baf_models.logistic import LogisticBaselineConfig, build_logistic_pipeline
from baf_models.preprocessing import FeatureGroups, build_preprocessor, feature_groups
from baf_models.training import (
    TrainDevBundle,
    VariantFitResult,
    build_variant_configs,
    extract_train_dev,
    fit_and_score_variant,
    fit_and_score_xgboost,
    load_train_dev_bundle,
)
from baf_models.xgboost_model import XGBoostBaselineConfig, build_xgboost_pipeline

__all__ = [
    "FeatureGroups",
    "LogisticBaselineConfig",
    "OperatingPoint",
    "TrainDevBundle",
    "VariantFitResult",
    "XGBoostBaselineConfig",
    "build_logistic_pipeline",
    "build_preprocessor",
    "build_variant_configs",
    "build_xgboost_pipeline",
    "extract_train_dev",
    "feature_groups",
    "fit_and_score_variant",
    "fit_and_score_xgboost",
    "load_train_dev_bundle",
    "select_fpr_constrained_threshold",
    "threshold_independent_metrics",
]
