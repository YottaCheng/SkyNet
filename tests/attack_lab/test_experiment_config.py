"""Formal configuration and manifest contract tests."""

from __future__ import annotations

import json

import pytest

from attack_lab.candidate_identity import CANDIDATE_IDENTITY_VERSION
from attack_lab.experiment_config import (
    FORMAL_CONFIG_SCHEMA_VERSION,
    FormalExperimentConfig,
)


def _payload() -> dict:
    return {
        "schema_version": FORMAL_CONFIG_SCHEMA_VERSION,
        "status": "FROZEN_READY_TO_RUN",
        "experiment_id": "test",
        "data_split": "dev_month6_reserved_formal_comparison",
        "month7_opened": False,
        "anchors": {"identifier": "x", "path": "/tmp/a", "sha256": "abc", "sample_size": 2},
        "experiment_seeds": [1, 2],
        "budget": {"Q": 5, "m": 2},
        "reference_pool": {"K": 10, "seed": 9},
        "candidate_identity_version": CANDIDATE_IDENTITY_VERSION,
        "governance": {"version": "g", "fingerprint": "gf"},
        "constraint_profile": {"id": "none", "fingerprint": None},
        "d1": {"version": "d", "artifact_sha256": {"model": "h"}},
        "attackers": {
            "a0": {"version": "a0-v"},
            "a1": {"version": "a1-v"},
            "a2": {"version": "a2-v"},
            "a3": {"version": "a3-v", "prompt_version": "p", "model": "m", "temperature": 0.2, "config_hash": "c"},
        },
        "aggregation": {"primary": "ASR@5"},
        "output_schema_version": "v1",
    }


def test_config_round_trip_and_manifest(tmp_path) -> None:
    config = FormalExperimentConfig(_payload())
    config.validate()
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    loaded = FormalExperimentConfig.load(path)
    assert loaded.config_hash == config.config_hash
    manifest = loaded.episode_manifest(
        attacker_id="a3",
        anchor_id="1",
        seed=1,
        reference_pool_fingerprint="pool",
        code_revision={"git_head": "head"},
    )
    assert manifest["attacker"] == "a3"
    assert manifest["prompt_version"] == "p"
    assert manifest["Q"] == 5
    assert manifest["candidate_identity_version"] == CANDIDATE_IDENTITY_VERSION


def test_config_rejects_month7_or_tampered_hash(tmp_path) -> None:
    payload = _payload()
    payload["month7_opened"] = True
    with pytest.raises(ValueError, match="month7_opened"):
        FormalExperimentConfig(payload).validate()

    valid = FormalExperimentConfig(_payload()).to_dict()
    valid["budget"]["Q"] = 9
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        FormalExperimentConfig.load(path)
