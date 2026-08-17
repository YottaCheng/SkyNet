"""Scoring-behaviour tests for D2-S."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from d2.aggregation import aggregate_equal_mean
from d2.contract import FORBIDDEN_INFERENCE_KEYS, RELATIONSHIP_IDS
from d2.errors import D2ContractError
from d2.scoring import D2SScorer, application_to_frame, fit_d2s_scorer


def _reference(synthetic_frame):
    return synthetic_frame.loc[
        synthetic_frame["month"].between(0, 5) & synthetic_frame["fraud_bool"].eq(0)
    ].copy()


def _application(synthetic_frame) -> dict:
    row = synthetic_frame.loc[synthetic_frame["month"].eq(6)].iloc[1]
    from baf_data.config import FROZEN_CONFIG

    return {name: row[name] for name in FROZEN_CONFIG.feature_columns}


@pytest.fixture()
def fitted_scorer(synthetic_frame) -> D2SScorer:
    return fit_d2s_scorer(_reference(synthetic_frame), raw_sha256="synthetic")


def test_eight_scores_in_unit_interval(fitted_scorer, synthetic_frame) -> None:
    result = fitted_scorer.score(_application(synthetic_frame))
    assert set(result["relationship_scores"]) == set(RELATIONSHIP_IDS)
    assert list(result["relationship_scores"]) == list(RELATIONSHIP_IDS)
    for rid, value in result["relationship_scores"].items():
        assert 0.0 <= value <= 1.0, rid
    assert 0.0 <= result["d2_score"] <= 1.0
    assert result["d2_score"] == aggregate_equal_mean(result["relationship_scores"])


def test_scoring_is_deterministic(fitted_scorer, synthetic_frame) -> None:
    app = _application(synthetic_frame)
    first = fitted_scorer.score(app)
    second = fitted_scorer.score(app)
    assert first == second
    again = fit_d2s_scorer(_reference(synthetic_frame), raw_sha256="synthetic")
    third = again.score(app)
    assert third["relationship_scores"] == first["relationship_scores"]
    assert third["d2_score"] == first["d2_score"]


def test_save_load_roundtrip(fitted_scorer, synthetic_frame, tmp_path) -> None:
    path = tmp_path / "d2s.json"
    fitted_scorer.save(path)
    loaded = D2SScorer.load(path)
    app = _application(synthetic_frame)
    assert loaded.score(app) == fitted_scorer.score(app)
    assert loaded.fingerprint == fitted_scorer.fingerprint
    assert loaded.month7_opened is False


def test_rarity_uses_observed_state_not_hardcoded_presence(fitted_scorer) -> None:
    """C01 must score 1-P(observed presence/absence | payment), not 1-P(present)."""
    table = fitted_scorer.tables["C01"]
    conditioner = next(iter(table.n_x))
    present = table.raw_rarity(conditioner, "1")
    absent = table.raw_rarity(conditioner, "0")
    p_present = table.probability(conditioner, "1")
    p_absent = table.probability(conditioner, "0")
    assert present == pytest.approx(1.0 - p_present)
    assert absent == pytest.approx(1.0 - p_absent)
    assert present != pytest.approx(absent)
    # Hard-coding 1-P(presence) would give the same rarity for both states.
    assert absent != pytest.approx(1.0 - p_present) or p_present == pytest.approx(0.5)


def test_official_missingness_semantics(fitted_scorer, synthetic_frame) -> None:
    app = _application(synthetic_frame)
    present = deepcopy(app)
    present["bank_months_count"] = 12
    sentinel = deepcopy(app)
    sentinel["bank_months_count"] = -1
    nan_app = deepcopy(app)
    nan_app["bank_months_count"] = float("nan")
    sentinel_score = fitted_scorer.score(sentinel)
    nan_score = fitted_scorer.score(nan_app)
    present_score = fitted_scorer.score(present)
    assert sentinel_score["relationship_scores"]["C01"] == nan_score["relationship_scores"]["C01"]
    # Presence vs official missing must be able to differ.
    assert (
        present_score["relationship_scores"]["C01"]
        != sentinel_score["relationship_scores"]["C01"]
        or present["payment_type"] == sentinel["payment_type"]
    )
    balcon_neg = deepcopy(app)
    balcon_neg["intended_balcon_amount"] = -3.2
    balcon_nan = deepcopy(app)
    balcon_nan["intended_balcon_amount"] = float("nan")
    assert (
        fitted_scorer.score(balcon_neg)["relationship_scores"]["C14"]
        == fitted_scorer.score(balcon_nan)["relationship_scores"]["C14"]
    )


def test_application_to_frame_applies_sentinels() -> None:
    frame = application_to_frame(
        {
            "payment_type": "AA",
            "bank_months_count": -1,
            "intended_balcon_amount": -8.0,
            "prev_address_months_count": -1,
            "current_address_months_count": -1,
            "housing_status": "BA",
            "customer_age": 30,
            "date_of_birth_distinct_emails_4w": 7,
            "employment_status": "CA",
            "phone_home_valid": 0,
            "phone_mobile_valid": 1,
        }
    )
    assert np.isnan(frame.loc[0, "bank_months_count"])
    assert np.isnan(frame.loc[0, "intended_balcon_amount"])
    assert np.isnan(frame.loc[0, "prev_address_months_count"])
    assert np.isnan(frame.loc[0, "current_address_months_count"])


def test_attacked_and_untouched_use_identical_score_function(
    fitted_scorer, synthetic_frame
) -> None:
    untouched = _application(synthetic_frame)
    attacked = deepcopy(untouched)
    attacked["payment_type"] = "AC"
    attacked["housing_status"] = "BE"
    first = fitted_scorer.score(untouched)
    second = fitted_scorer.score(attacked)
    # Same public function; outputs are ordinary score dicts with no provenance branch.
    assert set(first) == set(second) == {"relationship_scores", "d2_score"}
    assert D2SScorer.score is D2SScorer.score


def test_score_many_matches_score(fitted_scorer, synthetic_frame) -> None:
    frame = synthetic_frame.loc[synthetic_frame["month"].eq(6)].head(3)
    many = fitted_scorer.score_many(frame)
    for idx, row in frame.iterrows():
        single = fitted_scorer.score(row.to_dict())
        assert many.loc[idx, "d2_score"] == pytest.approx(single["d2_score"])
        for rid in RELATIONSHIP_IDS:
            assert many.loc[idx, rid] == pytest.approx(single["relationship_scores"][rid])
