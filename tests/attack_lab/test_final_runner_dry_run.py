"""Dry-run of the sealed final runner: no Month 7, no live API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack_lab.final_experiment import (
    FORBIDDEN_EXECUTE_WITHOUT_FLAG,
    FinalRunnerError,
    load_protocol,
    preflight_final,
    refuse_overwrite,
    run_dry_run,
    run_execute_final_preflight_only,
    write_status,
)
from attack_lab.final_protocol import DEFAULT_PROTOCOL_PATH
from attack_lab.paths import EXPERIMENTS_ROOT
from attack_lab.first_success import extract_first_successful_d1_pass


def test_preflight_requires_explicit_mode() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    errors = preflight_final(protocol=protocol, dry_run=False, execute_final=False)
    assert any("Refusing to open Month 7" in e or "dry-run" in e for e in errors)


def test_execute_final_fails_closed_before_month7() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    with pytest.raises(FinalRunnerError, match="Paired Month-7 anchors"):
        run_execute_final_preflight_only(protocol)


def test_non_overwrite_complete_run(tmp_path: Path) -> None:
    existing = tmp_path / "run"
    existing.mkdir()
    write_status(existing, "COMPLETE")
    with pytest.raises(FinalRunnerError, match="completed"):
        refuse_overwrite(existing)


def test_dry_run_executes_logical_chain(tmp_path: Path) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    parent = EXPERIMENTS_ROOT / "final_month7"
    result = run_dry_run(protocol=protocol, output_parent=parent)
    run_dir = Path(result["run_dir"])
    assert result["status"] == "COMPLETE"
    assert (run_dir / "RUN_STATUS.json").is_file()
    status = json.loads((run_dir / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETE"
    assert status["month7_accessed"] is False
    assert status["live_api_calls"] == 0
    assert (run_dir / "FINAL_RUN_MANIFEST.json").is_file()
    assert (run_dir / "raw_attack_trajectories.json").is_file()
    assert (run_dir / "first_success_d1_pass.json").is_file()
    assert (run_dir / "d2_offline.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    manifest = json.loads((run_dir / "FINAL_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["thinking"] == "OFF"
    assert manifest["K"] == 10
    assert manifest["Q"] == 5
    assert manifest["m"] == 2
    assert manifest["live_api_calls"] == 0
    submissions = json.loads(
        (run_dir / "first_success_d1_pass.json").read_text(encoding="utf-8")
    )
    assert submissions
    assert extract_first_successful_d1_pass(
        json.loads(
            next(run_dir.glob("trajectories/*/*/episode_result.json")).read_text(
                encoding="utf-8"
            )
        )
    )
    with pytest.raises(FinalRunnerError):
        refuse_overwrite(run_dir)


def test_first_success_skips_later_passes() -> None:
    episode = {
        "steps": [
            {
                "attempt": 1,
                "internal_defence": {"decision": "BLOCK"},
                "validity": {"is_valid": True, "candidate_features": {"income": 0.8}},
            },
            {
                "attempt": 2,
                "internal_defence": {"decision": "PASS"},
                "validity": {"is_valid": True, "candidate_features": {"income": 0.1}},
            },
            {
                "attempt": 3,
                "internal_defence": {"decision": "PASS"},
                "validity": {"is_valid": True, "candidate_features": {"income": 0.05}},
            },
        ]
    }
    found = extract_first_successful_d1_pass(episode)
    assert found is not None
    assert found["attempt"] == 2
    assert found["features"]["income"] == 0.1
