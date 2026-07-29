"""Statistical model architectures for the BAF experiments.

This package defines **unfitted** scikit-learn pipelines only. It never
reads Base.csv, never fits a transformer or model, and reuses the frozen
feature schema from :mod:`baf_data.config` instead of duplicating field
lists. Training, evaluation and threshold selection belong to a later,
separately logged task.
"""

from baf_models.logistic import LogisticBaselineConfig, build_logistic_pipeline
from baf_models.preprocessing import build_preprocessor, split_feature_kinds

__all__ = [
    "LogisticBaselineConfig",
    "build_logistic_pipeline",
    "build_preprocessor",
    "split_feature_kinds",
]
