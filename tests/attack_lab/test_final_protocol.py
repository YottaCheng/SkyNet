"""Frozen final Month-7 protocol fail-closed checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack_lab.final_protocol import (
    DEFAULT_PROTOCOL_PATH,
    FinalProtocolConfig,
    FinalProtocolError,
    protocol_role_statement,
)


def test_frozen_protocol_loads_and_pins() -> None:
    config = FinalProtocolConfig.load(DEFAULT_PROTOCOL_PATH)
    assert config.phase == "final"
    assert config.month == 7
    assert config.k == 10
    assert config.m_max == 2
    assert config.q_max == 5
    assert config.payload["require_reference_provenance"] is True
    assert config.payload["attackers"]["A1"]["model"] == "deepseek-v4-pro"
    assert config.payload["attackers"]["A1"]["thinking"] == "OFF"
    assert config.payload["attackers"]["A3"]["thinking_disabled"] is True
    assert config.payload["d2"]["primary"]["role"] == "PRIMARY"
    assert config.payload["d2"]["secondary_prespecified"]["role"] == "SECONDARY_PRESPECIFIED"
    assert "D2-L" in config.payload["d2"]["excluded"]
    roles = protocol_role_statement()
    assert roles["selection_after_month7_forbidden"] is True


def test_protocol_rejects_month6_phase(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["phase"] = "development"
    payload["evaluation_month"] = 6
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalProtocolError):
        FinalProtocolConfig.load(path)


def test_protocol_rejects_thinking_on(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["attackers"]["A1"]["thinking"] = "ON"
    payload["attackers"]["A1"]["thinking_disabled"] = False
    path = tmp_path / "think.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalProtocolError):
        FinalProtocolConfig.load(path)
