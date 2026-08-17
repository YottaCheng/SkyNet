"""Fail-closed Pro returned-model identity for development sanity gates."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_dev_model_selection as runner  # noqa: E402


def test_pro_sanity_returned_identity_accepts_pro_alias_or_version_not_flash() -> None:
    assert runner.returned_model_identifies_as_pro("deepseek-v4-pro")
    assert runner.returned_model_identifies_as_pro("DeepSeek-V4-Pro-0813")
    assert not runner.returned_model_identifies_as_pro("deepseek-v4-flash")
    assert not runner.returned_model_identifies_as_pro("DeepSeek-V4-Flash-0731")
    assert not runner.returned_model_identifies_as_pro("")

    def _row(*, returned: list[str]) -> dict:
        return {
            "success": True,
            "attempts_to_success": 1,
            "Q_violations": 0,
            "m_violations": 0,
            "hidden_exposure": 0,
            "raw_proxy_exposure": 0,
            "non_reference_backed": 0,
            "post_freeze_adaptation": 0,
            "reflection_timing_ok": True,
            "local_repair_reflection_pin_ok": True,
            "stable_action_slot_mapping_ok": True,
            "catalogue_has_all_abstract_proxies": True,
            "transport_requested_models": ["deepseek-v4-pro"],
            "transport_returned_models": returned,
        }

    pro_rows = [_row(returned=["DeepSeek-V4-Pro-0813"]) for _ in range(15)]
    assert runner.sanity_gates("A1-Pro", pro_rows) == []
    assert runner.sanity_gates("A3-Pro", pro_rows) == []

    empty_errors = runner.sanity_gates("A1-Pro", [_row(returned=[]) for _ in range(15)])
    assert any("returned_models empty" in item for item in empty_errors)

    flash_errors = runner.sanity_gates(
        "A3-Pro", [_row(returned=["deepseek-v4-flash"]) for _ in range(15)]
    )
    assert any("not Pro-compatible" in item for item in flash_errors)


def test_runner_exception_not_counted_as_q_or_catalogue_violation() -> None:
    good = {
        "success": True,
        "attempts_to_success": 1,
        "Q_violations": 0,
        "m_violations": 0,
        "hidden_exposure": 0,
        "raw_proxy_exposure": 0,
        "non_reference_backed": 0,
        "post_freeze_adaptation": 0,
        "reflection_timing_ok": True,
        "local_repair_reflection_pin_ok": True,
        "stable_action_slot_mapping_ok": True,
        "catalogue_has_all_abstract_proxies": True,
        "transport_requested_models": ["deepseek-v4-pro"],
        "transport_returned_models": ["deepseek-v4-pro"],
    }
    rows = [dict(good) for _ in range(14)]
    rows.append(
        {
            "success": False,
            "runner_exception": True,
            "stop_reason": "runner_exception",
            "Q_violations": None,
            "catalogue_has_all_abstract_proxies": None,
            "transport_requested_models": [],
            "transport_returned_models": [],
        }
    )
    integ = runner.condition_integrity(rows)
    assert integ["runner_exceptions"] == 1
    assert integ["completed"] == 14
    assert integ["Q_violations"] == 0
    assert integ["catalogue_all_abstract_proxies"] is True
    errors = runner.sanity_gates("A3-Pro", rows)
    assert any("runner_exceptions=1" in item for item in errors)
    assert not any("Q violations" in item for item in errors)
    assert not any("abstract proxy catalogue incomplete" in item for item in errors)
