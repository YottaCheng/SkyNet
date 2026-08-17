"""Authoritative N=25 thinking-mode runner pins (no DeepSeek calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_dev_model_selection as model_sel  # noqa: E402
import run_dev_thinking_mode_n25 as thinking  # noqa: E402
from attack_lab.benchmark_pins import (  # noqa: E402
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A3_PROMPT_VERSION,
    REASONING_EFFORT_MAX,
    BenchmarkPinError,
)


def test_authoritative_pins_are_v4_4_and_v2_4() -> None:
    assert PINNED_A1_PROMPT_VERSION == "a1_oneshot_v4_4_adversarial_objective"
    assert PINNED_A3_PROMPT_VERSION == (
        "a3_episodic_reflective_v2_4_adversarial_objective"
    )
    assert model_sel.PINNED_A1_PROMPT_VERSION == PINNED_A1_PROMPT_VERSION
    assert model_sel.PINNED_A3_PROMPT_VERSION == PINNED_A3_PROMPT_VERSION
    assert thinking.expected_pins()["a1_prompt_version"] == PINNED_A1_PROMPT_VERSION
    assert thinking.expected_pins()["a3_prompt_version"] == PINNED_A3_PROMPT_VERSION


def test_thinking_cells_register_four_pro_conditions() -> None:
    manifests = thinking.preflight_thinking_cells()
    assert [m["condition_id"] for m in manifests] == [
        "A1-Pro-ThinkOff",
        "A1-Pro-ThinkOn",
        "A3-Pro-ThinkOff",
        "A3-Pro-ThinkOn",
    ]
    by_id = {m["condition_id"]: m for m in manifests}
    assert by_id["A1-Pro-ThinkOff"]["prompt_version"] == PINNED_A1_PROMPT_VERSION
    assert by_id["A3-Pro-ThinkOn"]["prompt_version"] == PINNED_A3_PROMPT_VERSION
    assert by_id["A1-Pro-ThinkOff"]["thinking_enabled"] is False
    assert by_id["A1-Pro-ThinkOff"]["reasoning_effort"] is None
    assert by_id["A1-Pro-ThinkOn"]["thinking_enabled"] is True
    assert by_id["A1-Pro-ThinkOn"]["reasoning_effort"] == REASONING_EFFORT_MAX
    assert by_id["A3-Pro-ThinkOn"]["model"] == MODEL_PRO
    assert all(m["prompt_hash"] for m in manifests)


def test_legacy_v4_3_v2_3_pins_are_rejected_by_benchmark_helpers() -> None:
    from attack_lab.benchmark_pins import assert_pinned_a1, assert_pinned_a3

    with pytest.raises(BenchmarkPinError, match="prompt_version"):
        assert_pinned_a1(
            prompt_version="a1_oneshot_v4_3_public_reference_view",
            require_reference_provenance=True,
        )
    with pytest.raises(BenchmarkPinError, match="prompt_version"):
        assert_pinned_a3(
            prompt_version="a3_episodic_reflective_v2_3_public_reference_view",
            require_reference_provenance=True,
        )


def test_condition_manifest_helper_includes_required_metadata() -> None:
    class _Probe:
        prompt_version = PINNED_A1_PROMPT_VERSION
        gower_policy = None
        model = MODEL_PRO
        thinking_disabled = False
        reasoning_effort = REASONING_EFFORT_MAX

        def config_hash(self) -> str:
            return "cfg123"

    manifest = model_sel.attacker_condition_manifest(
        condition_id="A1-Pro-ThinkOn",
        attacker_kind="a1",
        llm_model=MODEL_PRO,
        attacker=_Probe(),
        thinking_disabled=False,
        reasoning_effort=REASONING_EFFORT_MAX,
    )
    for key in (
        "attacker_version",
        "prompt_version",
        "prompt_hash",
        "model",
        "thinking_disabled",
        "thinking_enabled",
        "reasoning_effort",
        "config_hash",
    ):
        assert key in manifest
        assert manifest[key] is not None
    assert manifest["prompt_version"] == PINNED_A1_PROMPT_VERSION
    assert manifest["config_hash"] == "cfg123"
    assert manifest["thinking_enabled"] is True
