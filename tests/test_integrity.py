"""Unit tests for hashing, source verification and output-path guards."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from baf_data.errors import OutputPathError, RawSourceIntegrityError
from baf_data.integrity import (
    ensure_output_path_allowed,
    sha256_of_file,
    verify_raw_source,
)


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"deterministic contents\n"
    path = tmp_path / "sample.bin"
    path.write_bytes(payload)
    assert sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_verify_raw_source_accepts_matching_hash(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_bytes(b"a,b\n1,2\n")
    expected = sha256_of_file(path)
    assert verify_raw_source(path, expected) == expected


def test_verify_raw_source_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_bytes(b"a,b\n1,2\n")
    with pytest.raises(RawSourceIntegrityError, match="SHA-256 mismatch"):
        verify_raw_source(path, "0" * 64)


def test_verify_raw_source_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RawSourceIntegrityError, match="not found"):
        verify_raw_source(tmp_path / "missing.csv", "0" * 64)


@pytest.fixture()
def raw_path(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "data" / "raw" / "baf"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "Base.csv"
    path.write_bytes(b"x\n")
    return path


def test_output_guard_rejects_raw_directory_itself(raw_path: Path) -> None:
    with pytest.raises(OutputPathError):
        ensure_output_path_allowed(raw_path.parent, raw_path)


def test_output_guard_rejects_subdirectory_of_raw(raw_path: Path) -> None:
    with pytest.raises(OutputPathError):
        ensure_output_path_allowed(raw_path.parent / "outputs", raw_path)


def test_output_guard_rejects_ancestor_of_raw(raw_path: Path) -> None:
    with pytest.raises(OutputPathError):
        ensure_output_path_allowed(raw_path.parents[2], raw_path)


def test_output_guard_accepts_sibling_directory(raw_path: Path) -> None:
    sibling = raw_path.parents[2] / "splits" / "baf_base"
    ensure_output_path_allowed(sibling, raw_path)  # must not raise
