"""Unfitted preprocessing definition for the Logistic Regression baseline.

Four explicit branches, all derived from the frozen data-layer schema
(:data:`baf_data.config.FROZEN_CONFIG`) so no field list is duplicated:

1. **continuous** — median imputation then standard scaling;
2. **missing indicators** — separate unscaled 0/1 indicators for exactly
   the frozen sentinel-rule columns (the only columns that can contain
   NaN after the data layer's normalisation; no missingness rule is
   invented here);
3. **binary** — validated 0/1 passthrough, never scaled;
4. **categorical** — unknown-safe one-hot encoding, never scaled.

Nothing in this module is fitted on BAF data and no data file is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baf_data.config import FROZEN_CONFIG, DataLayerConfig


@dataclass(frozen=True)
class FeatureGroups:
    """Disjoint preprocessing branches over the frozen feature columns.

    ``continuous``, ``binary`` and ``categorical`` partition the frozen
    feature columns exactly. ``missing_indicator_sources`` lists the
    columns (all continuous) whose missingness is additionally exposed
    as unscaled 0/1 indicator features.
    """

    continuous: tuple[str, ...]
    binary: tuple[str, ...]
    categorical: tuple[str, ...]
    missing_indicator_sources: tuple[str, ...]


def feature_groups(config: DataLayerConfig = FROZEN_CONFIG) -> FeatureGroups:
    """Derive the four preprocessing groups from the supplied config.

    Binary membership, kind information and sentinel rules all come from
    the data-layer configuration — the single executable source of
    truth. This module declares no field list of its own; the config's
    own ``validate()`` enforces that binary features are integer-kind
    feature columns.
    """
    kinds = {spec.name: spec.kind for spec in config.raw_columns}
    features = config.feature_columns

    binary = tuple(config.binary_features)
    categorical = tuple(n for n in features if kinds[n] == "string")
    continuous = tuple(
        n for n in features if kinds[n] in ("integer", "float") and n not in set(binary)
    )
    # Only the frozen sentinel-rule columns can contain NaN after the
    # data layer's normalisation; indicators are limited to exactly them.
    indicator_sources = tuple(
        rule.column for rule in config.sentinel_rules if rule.column in features
    )
    missing_from_continuous = set(indicator_sources) - set(continuous)
    if missing_from_continuous:
        raise ValueError(
            "Sentinel-rule columns must be continuous features; "
            f"got {sorted(missing_from_continuous)}."
        )
    return FeatureGroups(
        continuous=continuous,
        binary=binary,
        categorical=categorical,
        missing_indicator_sources=indicator_sources,
    )


class BinaryPassthrough(BaseEstimator, TransformerMixin):
    """Pass 0/1 columns through unchanged, validating their domain.

    Raises :class:`ValueError` at ``fit`` or ``transform`` time if any
    value is not exactly 0 or 1 (NaN included), so unexpected codes in
    later real-data use fail loudly instead of being silently scaled.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "BinaryPassthrough":
        self._validate(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._validate(X)
        return X.to_numpy(dtype="float64")

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return self.feature_names_in_

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        values = X.to_numpy(dtype="float64")
        if not np.isin(values, (0.0, 1.0)).all():
            bad = [
                str(column)
                for column, ok in zip(X.columns, np.isin(values, (0.0, 1.0)).all(axis=0))
                if not ok
            ]
            raise ValueError(
                f"Binary feature column(s) {bad} contain values other than 0/1."
            )


def build_continuous_pipeline() -> Pipeline:
    """Unfitted continuous pipeline: median imputation, then standard
    scaling. Missingness is exposed separately by the indicator branch,
    so the imputer does not append its own indicator columns."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
        ]
    )


def build_missing_indicator() -> MissingIndicator:
    """Unfitted 0/1 missing indicators for the sentinel-rule columns.

    ``features="all"`` keeps one indicator per listed column regardless
    of what happens to be missing in a particular fitting sample, so the
    output schema is deterministic. The indicators are never scaled."""
    return MissingIndicator(features="all")


def build_categorical_pipeline() -> Pipeline:
    """Unfitted categorical pipeline: one-hot encoding that ignores
    categories unseen during fitting instead of raising at transform."""
    return Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])


def build_preprocessor(config: DataLayerConfig = FROZEN_CONFIG) -> ColumnTransformer:
    """Construct the unfitted four-branch ColumnTransformer.

    Expects an X that already satisfies the data layer's
    ``validate_feature_schema`` (exactly the frozen feature columns).
    This function only assembles the transformer; it never calls
    ``fit`` and never touches Base.csv.
    """
    groups = feature_groups(config)
    return ColumnTransformer(
        transformers=[
            ("continuous", build_continuous_pipeline(), list(groups.continuous)),
            (
                "missing_indicator",
                build_missing_indicator(),
                list(groups.missing_indicator_sources),
            ),
            ("binary", BinaryPassthrough(), list(groups.binary)),
            ("categorical", build_categorical_pipeline(), list(groups.categorical)),
        ],
        remainder="drop",
    )
