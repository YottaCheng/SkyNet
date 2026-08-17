"""Data-boundary tests: Months 0–5 fit, Month 6 not for fit, Month 7 sealed."""

from __future__ import annotations

import pytest

from d2.data import (
    assert_months_allowed,
    load_d2_frame,
    load_month6_applications,
    load_reference_legitimate,
)
from d2.errors import D2DataError
from d2.scoring import fit_d2s_scorer


def test_sealed_month_request_raises() -> None:
    with pytest.raises(D2DataError, match="sealed"):
        assert_months_allowed([7], allow_calibration=True)
    with pytest.raises(D2DataError, match="sealed"):
        assert_months_allowed([0, 7], allow_calibration=False)


def test_month6_not_allowed_for_reference_load() -> None:
    with pytest.raises(D2DataError, match="not permitted"):
        assert_months_allowed([6], allow_calibration=False)


def test_loader_keeps_only_requested_months(synthetic_raw_layout) -> None:
    raw_path, config = synthetic_raw_layout
    loaded = load_reference_legitimate(raw_path, config=config, verify_hash=True)
    assert loaded.month7_opened is False
    assert set(loaded.frame["month"].unique()) <= {0, 1, 2, 3, 4, 5}
    assert set(loaded.frame["fraud_bool"].unique()) == {0}
    assert 7 not in set(loaded.frame["month"].unique())
    assert loaded.n_sealed_rows_skipped > 0


def test_month6_load_excludes_month7(synthetic_raw_layout) -> None:
    raw_path, config = synthetic_raw_layout
    loaded = load_month6_applications(
        raw_path, fraud_bool=None, config=config, verify_hash=True
    )
    assert set(loaded.months) == {6}
    assert loaded.month7_opened is False
    assert loaded.n_sealed_rows_skipped > 0


def test_direct_month7_load_raises(synthetic_raw_layout) -> None:
    raw_path, config = synthetic_raw_layout
    with pytest.raises(D2DataError, match="sealed"):
        load_d2_frame(
            raw_path,
            (7,),
            allow_calibration=True,
            fraud_bool=None,
            config=config,
            verify_hash=True,
        )


def test_fit_rejects_month6_and_fraud_rows(synthetic_frame) -> None:
    month6 = synthetic_frame.loc[synthetic_frame["month"].eq(6)].copy()
    with pytest.raises(Exception, match="non-training months"):
        fit_d2s_scorer(month6, raw_sha256="x", month7_opened=False)

    train_with_fraud = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    with pytest.raises(Exception, match="fraud_bool"):
        fit_d2s_scorer(train_with_fraud, raw_sha256="x", month7_opened=False)


def test_fit_uses_only_months_0_5_legitimate(synthetic_frame) -> None:
    reference = synthetic_frame.loc[
        synthetic_frame["month"].between(0, 5) & synthetic_frame["fraud_bool"].eq(0)
    ].copy()
    # Poison month 6 / 7 / fraud with a payment type that must not enter the table.
    poisoned = synthetic_frame.copy()
    poisoned.loc[poisoned["month"].eq(6), "payment_type"] = "ZZ"
    poisoned.loc[poisoned["month"].eq(7), "payment_type"] = "WW"
    poisoned.loc[poisoned["fraud_bool"].eq(1), "payment_type"] = "YY"
    scorer = fit_d2s_scorer(reference, raw_sha256="synthetic", month7_opened=False)
    c01_levels = set(scorer.tables["C01"].n_x)
    assert "ZZ" not in c01_levels
    assert "WW" not in c01_levels
    assert "YY" not in c01_levels
    assert scorer.reference_months == (0, 1, 2, 3, 4, 5)
    assert scorer.month7_opened is False
    assert scorer.bins.current_address_edges[0] >= 0.0


def test_development_phase_still_seals_month7() -> None:
    with pytest.raises(D2DataError, match="sealed"):
        assert_months_allowed([7], allow_calibration=True, phase="development")


def test_final_phase_requests_month7_only() -> None:
    assert assert_months_allowed([7], allow_calibration=False, phase="final") == (7,)
    with pytest.raises(D2DataError, match="Month 7 only"):
        assert_months_allowed([6], allow_calibration=True, phase="final")
    with pytest.raises(D2DataError, match="Month 7 only"):
        assert_months_allowed([0, 7], allow_calibration=False, phase="final")


def test_final_phase_synthetic_load_retains_only_month7(synthetic_raw_layout) -> None:
    raw_path, config = synthetic_raw_layout
    loaded = load_d2_frame(
        raw_path,
        (7,),
        allow_calibration=False,
        fraud_bool=None,
        config=config,
        verify_hash=True,
        phase="final",
    )
    assert loaded.month7_opened is True
    assert set(loaded.months) == {7}
