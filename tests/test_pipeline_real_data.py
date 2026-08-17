"""Integration test against the real immutable Base.csv.

Skipped automatically when the external drive is not mounted. Expected
counts are pinned to the 23 July 2026 read-only audit evidence
(`documentation/baf_audit/`), so any drift in the source file or the
frozen rules fails loudly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from baf_data.config import FROZEN_CONFIG
from baf_data.pipeline import run_pipeline

RAW_PATH = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")

pytestmark = pytest.mark.skipif(
    not RAW_PATH.is_file(), reason="External drive with Base.csv is not mounted."
)
# This module reads the monolithic Base.csv, including Month-7 rows, for
# historical split-count inventory. Exclude it from pre-Month-7 hardening:
#   pytest --ignore=tests/test_pipeline_real_data.py

EXPECTED_SPLITS = {
    "train": {"row_count": 794_989, "fraud_count": 8_151, "months": [0, 1, 2, 3, 4, 5]},
    "dev": {"row_count": 108_168, "fraud_count": 1_450, "months": [6]},
    "test": {"row_count": 96_843, "fraud_count": 1_428, "months": [7]},
}

EXPECTED_SENTINEL_CONVERSIONS = {
    "prev_address_months_count": 712_920,
    "current_address_months_count": 4_254,
    "intended_balcon_amount": 742_523,
    "bank_months_count": 253_635,
    "session_length_in_minutes": 2_015,
    "device_distinct_emails_8w": 359,
}


@pytest.fixture(scope="module")
def real_result(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("baf_real_outputs")
    return run_pipeline(RAW_PATH, output_dir, FROZEN_CONFIG)


def test_source_hash_verified_before_and_after(real_result) -> None:
    # run_pipeline itself raises if either verification fails; confirm the
    # recorded digest matches the frozen expectation.
    assert real_result.prepared.raw_sha256 == FROZEN_CONFIG.expected_sha256
    assert real_result.split_manifest["source"]["sha256"] == FROZEN_CONFIG.expected_sha256


def test_raw_file_is_not_writable() -> None:
    assert not os.access(RAW_PATH, os.W_OK)


def test_all_rows_belong_to_exactly_one_split(real_result) -> None:
    manifest = real_result.split_manifest
    assert manifest["total_rows"] == 1_000_000
    assert sum(entry["row_count"] for entry in manifest["splits"].values()) == 1_000_000
    assert manifest["total_fraud"] == 11_029


def test_split_counts_match_audited_evidence(real_result) -> None:
    for name, expected in EXPECTED_SPLITS.items():
        entry = real_result.split_manifest["splits"][name]
        assert entry["row_count"] == expected["row_count"], name
        assert entry["fraud_count"] == expected["fraud_count"], name
        assert entry["months"] == expected["months"], name


def test_splits_are_month_pure(real_result) -> None:
    frame = real_result.prepared.normalised_frame
    months = frame["month"]
    for name, expected in EXPECTED_SPLITS.items():
        observed = set(months.loc[real_result.prepared.indices[name]].unique())
        assert observed == set(expected["months"]), name


def test_x_views_exclude_all_forbidden_columns(real_result) -> None:
    for view in real_result.prepared.views.values():
        assert view.X.shape[1] == 27
        for forbidden in (
            "fraud_bool",
            "month",
            "device_fraud_count",
            "days_since_request",
            "credit_risk_score",
        ):
            assert forbidden not in view.X.columns


def test_sentinel_conversion_counts_match_audit(real_result) -> None:
    assert real_result.prepared.conversion_counts == EXPECTED_SENTINEL_CONVERSIONS


def test_valid_negative_velocity_values_survive(real_result) -> None:
    frame = real_result.prepared.normalised_frame
    assert int((frame["velocity_6h"] < 0).sum()) == 44


def test_no_rows_dropped_and_targets_unchanged(real_result) -> None:
    frame = real_result.prepared.normalised_frame
    assert len(frame) == 1_000_000
    total_positives = sum(int(v.y.sum()) for v in real_result.prepared.views.values())
    assert total_positives == 11_029


def test_no_output_inside_raw_directory(real_result) -> None:
    raw_dir = RAW_PATH.parent.resolve()
    for path in real_result.written_paths.values():
        assert raw_dir not in path.resolve().parents
