"""Unit tests for Month-6 calibration helpers (no real Month-6/7 load)."""

from __future__ import annotations

import numpy as np
import pytest

from d2.calibrate import extract_d1_pass_features, security_rows, thresholds_for_budgets
from d2.data import assert_months_allowed
from d2.errors import D2DataError


def test_thresholds_match_fixed_budgets() -> None:
    scores = np.linspace(0.0, 1.0, 1001)
    table = thresholds_for_budgets(scores, budgets=(0.01, 0.05, 0.10))
    assert list(table["budget"]) == [0.01, 0.05, 0.10]
    for _, row in table.iterrows():
        assert row["n_legitimate_d1_pass"] == 1001
        assert 0.0 <= row["threshold"] <= 1.0
        assert row["n_review"] >= 1
        assert abs(row["benign_review_rate"] - row["budget"]) < 0.02


def test_security_rows_interception_and_bypass() -> None:
    budget = thresholds_for_budgets(np.array([0.1, 0.2, 0.3, 0.4]), budgets=(0.25,))
    security = security_rows(
        budget_table=budget,
        attacker_scores={
            "A0": np.array([0.05, 0.9]),
            "A1": np.array([0.95]),
            "A2": np.array([0.0, 0.0, 0.0]),
            "A3": np.array([0.8, 0.85]),
        },
    )
    row = security.iloc[0]
    assert row["A0_n_d1_pass"] == 2
    assert row["A0_interception"] + row["A0_full_bypass"] == pytest.approx(1.0)
    assert row["A2_interception"] in {0.0}


def test_extract_ignores_invalid_or_block() -> None:
    episode = {
        "steps": [
            {
                "internal_defence": {"decision": "PASS"},
                "validity": {"is_valid": False, "candidate_features": {"a": 1}},
            },
            {
                "internal_defence": {"decision": "BLOCK"},
                "validity": {"is_valid": True, "candidate_features": {"a": 2}},
            },
        ]
    }
    assert extract_d1_pass_features(episode) == []


def test_calibration_month_allowed_only_with_flag() -> None:
    assert assert_months_allowed([6], allow_calibration=True) == (6,)
    with pytest.raises(D2DataError):
        assert_months_allowed([6], allow_calibration=False)
