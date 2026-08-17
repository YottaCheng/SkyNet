"""Tests for the D2-S v1.1 Isolation Forest aggregator.

Does not modify D2-S v1.0.  Uses the synthetic fixture only; does not
open Month 7 or call an LLM.
"""

from __future__ import annotations

from inspect import getsource

import numpy as np
import pandas as pd
import pytest

from baf_data.config import FROZEN_CONFIG
from d2.aggregation import aggregate_equal_mean
from d2.contract import RELATIONSHIP_IDS, SCORE_CONTRACT_ID
from d2.errors import D2ContractError, D2FitError
from d2.iforest_v11 import (
    FIXED_IFOREST_PARAMS,
    FROZEN_D2S_V10_FINGERPRINT,
    IFOREST_FEATURE_IDS,
    SCORE_CONTRACT_ID_V11,
    D2SV11IForestAggregator,
    collapse_to_iforest_features,
    fit_iforest_aggregator,
)
from d2.scoring import D2SScorer, fit_d2s_scorer


def _reference(synthetic_frame: pd.DataFrame) -> pd.DataFrame:
    return synthetic_frame.loc[
        synthetic_frame["month"].between(0, 5) & synthetic_frame["fraud_bool"].eq(0)
    ].copy()


def _relationship_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(20260816)
    data = {rid: rng.uniform(0.0, 1.0, n) for rid in RELATIONSHIP_IDS}
    return pd.DataFrame(data)


def test_v10_contract_id_unchanged() -> None:
    assert SCORE_CONTRACT_ID == "d2s-v1.0.0-pairwise8-20260816"
    assert SCORE_CONTRACT_ID_V11 != SCORE_CONTRACT_ID


def test_payment_channel_is_max_c01_c14() -> None:
    frame = pd.DataFrame(
        {
            "C01": [0.1, 0.9, 0.4],
            "C14": [0.8, 0.2, 0.4],
            "C13": [0.0, 0.5, 1.0],
            "C09": [0.1, 0.2, 0.3],
            "C03": [0.2, 0.3, 0.4],
            "C10": [0.0, 0.0, 0.1],
            "C11": [0.5, 0.5, 0.5],
            "C15": [0.7, 0.1, 0.0],
        }
    )
    features = collapse_to_iforest_features(frame)
    assert list(features.columns) == list(IFOREST_FEATURE_IDS)
    assert list(features["payment_channel"]) == pytest.approx([0.8, 0.9, 0.4])
    assert "C01" not in features.columns
    assert "C14" not in features.columns
    assert len(features.columns) == 7


def test_collapse_rejects_d1_score_column() -> None:
    frame = _relationship_frame(8)
    frame["d1_score"] = 0.4
    with pytest.raises(D2ContractError, match="Forbidden inference"):
        collapse_to_iforest_features(frame)


def test_fit_refuses_fraud_rows() -> None:
    frame = _relationship_frame(20)
    frame["fraud_bool"] = 0
    frame.loc[0, "fraud_bool"] = 1
    with pytest.raises(D2FitError, match="non-legitimate"):
        fit_iforest_aggregator(
            frame, v10_fingerprint=FROZEN_D2S_V10_FINGERPRINT, month7_opened=False
        )


def test_fit_refuses_month7_flag() -> None:
    with pytest.raises(D2FitError, match="Month 7"):
        fit_iforest_aggregator(
            _relationship_frame(20),
            v10_fingerprint=FROZEN_D2S_V10_FINGERPRINT,
            month7_opened=True,
        )


def test_fit_refuses_wrong_v10_fingerprint() -> None:
    with pytest.raises(D2ContractError, match="fingerprint"):
        fit_iforest_aggregator(
            _relationship_frame(20), v10_fingerprint="not-the-frozen-v10"
        )


def test_anomaly_score_is_negative_score_samples() -> None:
    frame = _relationship_frame(60)
    aggregator = fit_iforest_aggregator(
        frame, v10_fingerprint=FROZEN_D2S_V10_FINGERPRINT
    )
    features = collapse_to_iforest_features(frame)
    expected = -aggregator.model.score_samples(features.to_numpy(dtype="float64"))
    got = aggregator.score_relationship_frame(frame)
    np.testing.assert_allclose(got, expected)
    assert not np.array_equal(
        got, aggregator.model.predict(features.to_numpy(dtype="float64"))
    )


def test_score_source_does_not_call_predict() -> None:
    source = getsource(D2SV11IForestAggregator.score_features)
    assert ".predict(" not in source
    assert "score_samples" in source


def test_save_load_roundtrip(tmp_path) -> None:
    frame = _relationship_frame(50)
    fitted = fit_iforest_aggregator(frame, v10_fingerprint=FROZEN_D2S_V10_FINGERPRINT)
    model_path = tmp_path / "D2S_V11_IFOREST_MODEL.joblib"
    config_path = tmp_path / "D2S_V11_IFOREST_CONFIG.json"
    fitted.save(model_path, config_path)
    loaded = D2SV11IForestAggregator.load(model_path)
    np.testing.assert_allclose(
        loaded.score_relationship_frame(frame),
        fitted.score_relationship_frame(frame),
    )
    assert loaded.params == FIXED_IFOREST_PARAMS
    assert loaded.n_train == 50


def test_v10_scorer_still_equal_mean(synthetic_frame) -> None:
    reference = _reference(synthetic_frame)
    scorer = fit_d2s_scorer(reference, raw_sha256="synthetic")
    row = synthetic_frame.loc[synthetic_frame["month"].eq(6)].iloc[1]
    features = {name: row[name] for name in FROZEN_CONFIG.feature_columns}
    result = scorer.score(features)
    assert result["d2_score"] == pytest.approx(
        aggregate_equal_mean(result["relationship_scores"])
    )
    assert isinstance(scorer, D2SScorer)
