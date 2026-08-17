"""Month-7 seal and sampling isolation for D2-L."""

from __future__ import annotations

from pathlib import Path

import pytest

from d2l.calibrate import draw_disjoint_samples, thresholds_for_budgets
from d2l.errors import D2LDataError
from d2l.isolation import assert_months_allowed, assert_not_month7_path


def test_month7_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(D2LDataError):
        assert_not_month7_path(tmp_path / "month7_scores.csv")
    with pytest.raises(D2LDataError):
        assert_not_month7_path(Path("/tmp/month_7/data.csv"))


def test_sealed_month_request_is_rejected() -> None:
    with pytest.raises(D2LDataError):
        assert_months_allowed([7])
    with pytest.raises(D2LDataError):
        assert_months_allowed([6, 7])
    assert assert_months_allowed([6]) == (6,)


def test_disjoint_samples_are_deterministic(synthetic_frame) -> None:
    legit = synthetic_frame.loc[
        synthetic_frame["month"].eq(6) & synthetic_frame["fraud_bool"].eq(0)
    ].copy()
    legit = legit.rename_axis("source_row_id").reset_index()
    first = draw_disjoint_samples(legit, n_cal=2, n_val=2, n_sanity=1)
    second = draw_disjoint_samples(legit, n_cal=2, n_val=2, n_sanity=1)
    assert list(first["calibration"]["source_row_id"]) == list(
        second["calibration"]["source_row_id"]
    )
    cal = set(first["calibration"]["source_row_id"])
    val = set(first["validation"]["source_row_id"])
    sanity = set(first["sanity"]["source_row_id"])
    assert cal.isdisjoint(val)
    assert cal.isdisjoint(sanity)
    assert val.isdisjoint(sanity)


def test_d2l_load_columns_cover_all_core_fields() -> None:
    from d2l.contract import APPLICATION_FIELDS
    from d2l.data import _LOAD_COLUMNS

    assert set(APPLICATION_FIELDS) <= set(_LOAD_COLUMNS)
    assert "fraud_bool" in _LOAD_COLUMNS
    assert "month" in _LOAD_COLUMNS
    assert "credit_risk_score" not in _LOAD_COLUMNS


def test_constant_scores_cannot_realise_partial_budgets() -> None:
    import numpy as np

    from d2l.calibrate import score_collapse_report, thresholds_for_budgets

    scores = np.full(20, 85.0)
    table = thresholds_for_budgets(scores, budgets=(0.05, 0.10, 0.15))
    assert (table["empirical_review_rate"] == 1.0).all()
    assert (table["threshold"] == 85.0).all()
    collapse = score_collapse_report(scores)
    assert collapse["collapsed"] is True
    assert collapse["n_unique"] == 1


def test_thresholds_record_empirical_rates() -> None:
    import numpy as np

    scores = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    table = thresholds_for_budgets(scores, budgets=(0.10, 0.20))
    ten = table.loc[table["budget"].eq(0.10)].iloc[0]
    assert ten["n_review"] >= 1
    assert 0.0 < float(ten["empirical_review_rate"]) <= 1.0
    assert float(ten["threshold"]) == float(np.quantile(scores, 0.90))
