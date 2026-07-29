"""Unfitted preprocessing definition for the Logistic Regression baseline.

Numeric columns: median imputation with missing indicators, then
standard scaling. Categorical columns: one-hot encoding that ignores
unseen categories at transform time.

Feature lists are derived from the frozen data-layer schema
(:data:`baf_data.config.FROZEN_CONFIG`); nothing is duplicated here.
Nothing in this module is fitted and no data file is read.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baf_data.config import FROZEN_CONFIG, DataLayerConfig


def split_feature_kinds(
    config: DataLayerConfig = FROZEN_CONFIG,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition the frozen feature columns into (numeric, categorical).

    Integer and float columns are numeric; string columns are
    categorical. The two tuples preserve raw file order and together
    cover every frozen feature column exactly once.
    """
    kinds = {spec.name: spec.kind for spec in config.raw_columns}
    numeric = tuple(
        name for name in config.feature_columns if kinds[name] in ("integer", "float")
    )
    categorical = tuple(
        name for name in config.feature_columns if kinds[name] == "string"
    )
    return numeric, categorical


def build_numeric_pipeline() -> Pipeline:
    """Unfitted numeric pipeline: median imputer with missing indicators,
    then standard scaling (indicator columns are scaled too, which is
    harmless for binary indicators and keeps the pipeline simple)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Unfitted categorical pipeline: one-hot encoding that ignores
    categories unseen during fitting instead of raising at transform."""
    return Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )


def build_preprocessor(config: DataLayerConfig = FROZEN_CONFIG) -> ColumnTransformer:
    """Construct the unfitted ColumnTransformer for the LR baseline.

    Expects an X that already satisfies the data layer's
    ``validate_feature_schema`` (exactly the frozen feature columns).
    This function only assembles the transformer; it never calls
    ``fit`` and never touches Base.csv.
    """
    numeric, categorical = split_feature_kinds(config)
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), list(numeric)),
            ("categorical", build_categorical_pipeline(), list(categorical)),
        ],
        remainder="drop",
    )
