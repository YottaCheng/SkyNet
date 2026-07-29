"""Integration tests for the full pipeline on a synthetic raw layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from baf_data.errors import OutputPathError, RawSourceIntegrityError
from baf_data.manifests import FEATURE_MANIFEST_NAME, SPLIT_MANIFEST_NAME
from baf_data.pipeline import RUN_LOG_NAME, run_pipeline


def test_full_pipeline_writes_manifests_and_run_log(synthetic_raw_layout, tmp_path: Path) -> None:
    raw_path, config = synthetic_raw_layout
    output_dir = tmp_path / "outputs"
    result = run_pipeline(raw_path, output_dir, config)

    assert (output_dir / SPLIT_MANIFEST_NAME).is_file()
    assert (output_dir / FEATURE_MANIFEST_NAME).is_file()
    assert (output_dir / RUN_LOG_NAME).is_file()

    manifest = result.split_manifest
    assert manifest["total_rows"] == config.expected_rows
    assert manifest["source"]["sha256"] == config.expected_sha256
    assert set(manifest["splits"]) == {"train", "dev", "test"}
    assert result.feature_manifest["feature_count"] == 27

    # No output landed inside the raw directory.
    assert list(raw_path.parent.iterdir()) == [raw_path]


def test_repeated_runs_produce_identical_manifests(synthetic_raw_layout, tmp_path: Path) -> None:
    raw_path, config = synthetic_raw_layout
    first_dir = tmp_path / "run1"
    second_dir = tmp_path / "run2"
    run_pipeline(raw_path, first_dir, config)
    run_pipeline(raw_path, second_dir, config)
    for name in (SPLIT_MANIFEST_NAME, FEATURE_MANIFEST_NAME):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_pipeline_refuses_tampered_source(synthetic_raw_layout, tmp_path: Path) -> None:
    raw_path, config = synthetic_raw_layout
    raw_path.write_text(raw_path.read_text() + "tampered\n")
    with pytest.raises(RawSourceIntegrityError, match="SHA-256 mismatch"):
        run_pipeline(raw_path, tmp_path / "outputs", config)
    assert not (tmp_path / "outputs").exists()


def test_pipeline_refuses_output_inside_raw_directory(synthetic_raw_layout) -> None:
    raw_path, config = synthetic_raw_layout
    with pytest.raises(OutputPathError):
        run_pipeline(raw_path, raw_path.parent / "outputs", config)
