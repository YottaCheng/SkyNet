"""Fail-closed experimental phase/month data-access contract."""

from __future__ import annotations

import pytest

from baf_data.errors import ProtocolAccessError
from baf_data.protocol_access import (
    DEVELOPMENT_MONTHS,
    load_dataset_for_protocol,
    validate_phase_months,
)


def test_development_cannot_request_month7() -> None:
    with pytest.raises(ProtocolAccessError, match="cannot request Month 7"):
        validate_phase_months("development", [0, 1, 2, 3, 4, 5, 6, 7])
    with pytest.raises(ProtocolAccessError, match="cannot request Month 7"):
        validate_phase_months("development", [7])


def test_final_cannot_fall_back_to_month6() -> None:
    with pytest.raises(ProtocolAccessError, match="exactly"):
        validate_phase_months("final", [6])
    with pytest.raises(ProtocolAccessError, match="exactly"):
        validate_phase_months("final", [0, 1, 2, 3, 4, 5, 6, 7])
    with pytest.raises(ProtocolAccessError, match="exactly"):
        validate_phase_months("final", [7, 6])


def test_unknown_phase_fails_closed() -> None:
    with pytest.raises(ProtocolAccessError, match="Unknown experimental phase"):
        validate_phase_months("dev", DEVELOPMENT_MONTHS)


def test_development_load_drops_month7_before_scientific_ops(
    synthetic_raw_layout,
) -> None:
    raw_path, config = synthetic_raw_layout
    loaded = load_dataset_for_protocol(
        raw_path,
        phase="development",
        allowed_months=DEVELOPMENT_MONTHS,
        config=config,
    )
    months = set(int(m) for m in loaded.frame["month"].unique())
    assert months <= set(DEVELOPMENT_MONTHS)
    assert 7 not in months
    assert loaded.month7_rows_retained is False
    assert loaded.month_filter_was_first_semantic_operation is True
    assert "test" not in loaded.views


def test_development_month6_only(synthetic_raw_layout) -> None:
    raw_path, config = synthetic_raw_layout
    loaded = load_dataset_for_protocol(
        raw_path, phase="development", allowed_months=(6,), config=config
    )
    assert set(int(m) for m in loaded.frame["month"].unique()) == {6}
    assert loaded.views["dev"].X.shape[0] == len(loaded.frame)


def test_final_load_on_synthetic_fixture_does_not_use_development_months(
    synthetic_raw_layout,
) -> None:
    raw_path, config = synthetic_raw_layout
    loaded = load_dataset_for_protocol(
        raw_path, phase="final", allowed_months=(7,), config=config
    )
    assert set(int(m) for m in loaded.frame["month"].unique()) == {7}
    assert loaded.month7_rows_retained is True
    assert "train" not in loaded.views
    assert "dev" not in loaded.views


def test_loader_rejects_development_month7_even_if_file_contains_it(
    synthetic_raw_layout,
) -> None:
    raw_path, config = synthetic_raw_layout
    with pytest.raises(ProtocolAccessError, match="cannot request Month 7"):
        load_dataset_for_protocol(
            raw_path, phase="development", allowed_months=(7,), config=config
        )
