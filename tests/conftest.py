"""Shared fixtures: a synthetic BAF-shaped frame and a matching config.

The synthetic frame reproduces the raw schema (32 columns, correct dtype
kinds) at a small scale, including sentinel values, valid negative
velocity_6h values and negative credit_risk_score values, so pure
functions can be tested quickly without the external drive.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baf_data.config import FROZEN_CONFIG, DataLayerConfig  # noqa: E402

ROWS_PER_MONTH = 6  # months 0-7 -> 48 rows in total


def make_synthetic_frame() -> pd.DataFrame:
    """Deterministic small frame matching the frozen raw schema."""
    rng = np.random.default_rng(20260728)
    n = ROWS_PER_MONTH * 8
    month = np.repeat(np.arange(8), ROWS_PER_MONTH)
    fraud = np.zeros(n, dtype="int64")
    fraud[::ROWS_PER_MONTH] = 1  # one fraud case per month

    df = pd.DataFrame(
        {
            "fraud_bool": fraud,
            "income": rng.uniform(0.1, 0.9, n),
            "name_email_similarity": rng.uniform(0, 1, n),
            "prev_address_months_count": np.where(
                np.arange(n) % 3 == 0, -1, rng.integers(0, 380, n)
            ).astype("int64"),
            "current_address_months_count": np.where(
                np.arange(n) % 8 == 0, -1, rng.integers(0, 420, n)
            ).astype("int64"),
            "customer_age": rng.integers(10, 90, n).astype("int64"),
            "days_since_request": rng.uniform(0, 78, n),
            "intended_balcon_amount": np.where(
                np.arange(n) % 2 == 0, rng.uniform(-16, -0.01, n), rng.uniform(0, 112, n)
            ),
            "payment_type": np.array(["AA", "AB", "AC", "AD", "AE"])[np.arange(n) % 5],
            "zip_count_4w": rng.integers(1, 6700, n).astype("int64"),
            # First two rows carry valid negative velocity values that must survive.
            "velocity_6h": np.concatenate(
                [np.array([-170.6, -5.5]), rng.uniform(0, 16000, n - 2)]
            ),
            "velocity_24h": rng.uniform(1300, 9500, n),
            "velocity_4w": rng.uniform(2825, 6994, n),
            "bank_branch_count_8w": rng.integers(0, 2385, n).astype("int64"),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 39, n).astype("int64"),
            "employment_status": np.array(["CA", "CB", "CC", "CD", "CE", "CF", "CG"])[
                np.arange(n) % 7
            ],
            # Include -1 and other negatives that must all be preserved.
            "credit_risk_score": np.concatenate(
                [np.array([-1, -170, -42]), rng.integers(0, 389, n - 3)]
            ).astype("int64"),
            "email_is_free": rng.integers(0, 2, n).astype("int64"),
            "housing_status": np.array(["BA", "BB", "BC", "BD", "BE", "BF", "BG"])[
                np.arange(n) % 7
            ],
            "phone_home_valid": rng.integers(0, 2, n).astype("int64"),
            "phone_mobile_valid": rng.integers(0, 2, n).astype("int64"),
            "bank_months_count": np.where(
                np.arange(n) % 4 == 0, -1, rng.integers(0, 32, n)
            ).astype("int64"),
            "has_other_cards": rng.integers(0, 2, n).astype("int64"),
            "proposed_credit_limit": rng.choice([190.0, 200.0, 500.0, 2100.0], n),
            "foreign_request": rng.integers(0, 2, n).astype("int64"),
            "source": np.array(["INTERNET", "TELEAPP"])[np.arange(n) % 2],
            "session_length_in_minutes": np.where(
                np.arange(n) % 6 == 0, -1.0, rng.uniform(0, 85, n)
            ),
            "device_os": np.array(["windows", "linux", "macintosh", "x11", "other"])[
                np.arange(n) % 5
            ],
            "keep_alive_session": rng.integers(0, 2, n).astype("int64"),
            "device_distinct_emails_8w": np.where(
                np.arange(n) % 12 == 0, -1, rng.integers(0, 3, n)
            ).astype("int64"),
            "device_fraud_count": np.zeros(n, dtype="int64"),
            "month": month.astype("int64"),
        }
    )
    assert tuple(df.columns) == FROZEN_CONFIG.raw_column_names
    return df


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture()
def synthetic_config(synthetic_frame: pd.DataFrame) -> DataLayerConfig:
    """Frozen config adjusted only for the synthetic frame's size/hash."""
    return replace(
        FROZEN_CONFIG,
        expected_rows=len(synthetic_frame),
        expected_sha256="unused-for-in-memory-tests",
    )


@pytest.fixture()
def synthetic_raw_layout(tmp_path: Path, synthetic_frame: pd.DataFrame):
    """A raw/baf/Base.csv layout on disk plus a config with the real hash."""
    from baf_data.integrity import sha256_of_file

    raw_dir = tmp_path / "raw" / "baf"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "Base.csv"
    synthetic_frame.to_csv(raw_path, index=False)
    config = replace(
        FROZEN_CONFIG,
        expected_rows=len(synthetic_frame),
        expected_sha256=sha256_of_file(raw_path),
    )
    return raw_path, config
