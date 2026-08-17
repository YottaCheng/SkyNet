"""Fail-closed benchmark/formal pins and DeepSeek Pro cost dispatch."""

from __future__ import annotations

import pytest

from attack_lab.attackers.a1_planner import (
    DEFAULT_MODEL,
    FORMAL_A1_MODEL_CONFIG,
    PROMPT_VERSION_V4,
    estimate_flash_cost_usd,
)
from attack_lab.attackers.a2_search import GOWER_POLICY_LEGACY_V1
from attack_lab.benchmark_pins import (
    MODEL_FLASH,
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
    PINNED_GOVERNANCE_FINGERPRINT,
    REASONING_EFFORT_MAX,
    BenchmarkPinError,
    assert_benchmark_match_pins,
    assert_pinned_defence_identity,
    assert_thinking_cell_config,
    condition_manifest,
    estimate_deepseek_cost_usd,
    pinned_attacker_summary,
    preflight_formal_payload,
    require_supported_llm_model,
)


def test_library_defaults_are_not_silently_rewritten() -> None:
    assert DEFAULT_MODEL == MODEL_FLASH == "deepseek-v4-flash"
    assert FORMAL_A1_MODEL_CONFIG.model == MODEL_FLASH
    assert FORMAL_A1_MODEL_CONFIG.prompt_version == PROMPT_VERSION_V4
    assert PINNED_A1_PROMPT_VERSION == "a1_oneshot_v4_3_public_reference_view"
    assert PINNED_A2_GOWER_POLICY == "a2_public_reference_gower_v2"
    assert PINNED_A3_PROMPT_VERSION == (
        "a3_episodic_reflective_v2_3_public_reference_view"
    )


def test_flash_cost_dispatch_matches_historical_estimator() -> None:
    kwargs = {
        "prompt_tokens": 12_345,
        "completion_tokens": 678,
        "cached_tokens": 1_000,
    }
    historical = estimate_flash_cost_usd(**kwargs)
    dispatched = estimate_deepseek_cost_usd(model=MODEL_FLASH, **kwargs)
    assert dispatched == historical
    pro = estimate_deepseek_cost_usd(model=MODEL_PRO, **kwargs)
    assert pro > historical


def test_pins_fail_closed_on_legacy_versions() -> None:
    with pytest.raises(BenchmarkPinError, match="prompt_version"):
        assert_benchmark_match_pins(
            attacker_id="a1",
            prompt_version=PROMPT_VERSION_V4,
            gower_policy=None,
            require_reference_provenance=True,
            llm_model=MODEL_FLASH,
        )
    with pytest.raises(BenchmarkPinError, match="prompt_version"):
        assert_benchmark_match_pins(
            attacker_id="a1",
            prompt_version="a1_oneshot_v4_4_adversarial_objective",
            gower_policy=None,
            require_reference_provenance=True,
            llm_model=MODEL_PRO,
        )
    with pytest.raises(BenchmarkPinError, match="gower_policy"):
        assert_benchmark_match_pins(
            attacker_id="a2",
            prompt_version=None,
            gower_policy=GOWER_POLICY_LEGACY_V1,
            require_reference_provenance=True,
        )
    with pytest.raises(BenchmarkPinError, match="require_reference_provenance"):
        assert_benchmark_match_pins(
            attacker_id="a3",
            prompt_version=PINNED_A3_PROMPT_VERSION,
            gower_policy=None,
            require_reference_provenance=False,
            llm_model=MODEL_PRO,
        )
    with pytest.raises(BenchmarkPinError, match="Unsupported llm_model"):
        require_supported_llm_model("deepseek-chat")


def test_pinned_flash_and_pro_are_both_accepted() -> None:
    assert_benchmark_match_pins(
        attacker_id="a1",
        prompt_version=PINNED_A1_PROMPT_VERSION,
        gower_policy=None,
        require_reference_provenance=True,
        llm_model=MODEL_FLASH,
    )
    assert_benchmark_match_pins(
        attacker_id="A1-Pro-ThinkOn",
        prompt_version=PINNED_A1_PROMPT_VERSION,
        gower_policy=None,
        require_reference_provenance=True,
        llm_model=MODEL_PRO,
    )
    assert_benchmark_match_pins(
        attacker_id="A3-Pro-ThinkOff",
        prompt_version=PINNED_A3_PROMPT_VERSION,
        gower_policy=None,
        require_reference_provenance=True,
        llm_model=MODEL_PRO,
    )
    assert_benchmark_match_pins(
        attacker_id="a2",
        prompt_version=None,
        gower_policy=PINNED_A2_GOWER_POLICY,
        require_reference_provenance=True,
    )


def test_thinking_and_defence_preflight_fail_closed() -> None:
    assert_thinking_cell_config(
        thinking_disabled=True,
        reasoning_effort=None,
        expect_thinking_disabled=True,
    )
    assert_thinking_cell_config(
        thinking_disabled=False,
        reasoning_effort=REASONING_EFFORT_MAX,
        expect_thinking_disabled=False,
    )
    with pytest.raises(BenchmarkPinError, match="thinking_disabled mismatch"):
        assert_thinking_cell_config(
            thinking_disabled=True,
            reasoning_effort=None,
            expect_thinking_disabled=False,
        )
    with pytest.raises(BenchmarkPinError, match="reasoning_effort"):
        assert_thinking_cell_config(
            thinking_disabled=False,
            reasoning_effort=None,
            expect_thinking_disabled=False,
        )
    assert_pinned_defence_identity(
        d1_artefact_id=PINNED_D1_ARTEFACT_ID,
        governance_fingerprint=PINNED_GOVERNANCE_FINGERPRINT,
        require_reference_provenance=True,
        month7_path_fragment="/final_month6/artefacts",
    )
    with pytest.raises(BenchmarkPinError, match="D1 artefact"):
        assert_pinned_defence_identity(
            d1_artefact_id="wrong",
            governance_fingerprint=PINNED_GOVERNANCE_FINGERPRINT,
            require_reference_provenance=True,
        )
    with pytest.raises(BenchmarkPinError, match="Month-7"):
        assert_pinned_defence_identity(
            d1_artefact_id=PINNED_D1_ARTEFACT_ID,
            governance_fingerprint=PINNED_GOVERNANCE_FINGERPRINT,
            require_reference_provenance=True,
            month7_path_fragment="/path/month7/leak",
        )


def test_condition_manifest_records_required_fields() -> None:
    manifest = condition_manifest(
        condition_id="A1-Pro-ThinkOn",
        attacker_kind="a1",
        prompt_version=PINNED_A1_PROMPT_VERSION,
        model=MODEL_PRO,
        thinking_disabled=False,
        reasoning_effort=REASONING_EFFORT_MAX,
        config_hash="abc",
        prompt_hash="def",
    )
    assert manifest["prompt_version"] == PINNED_A1_PROMPT_VERSION
    assert manifest["thinking_enabled"] is True
    assert manifest["reasoning_effort"] == "max"
    assert manifest["pins"] == pinned_attacker_summary()


def test_formal_payload_preflight_requires_explicit_pins() -> None:
    payload = {
        "require_reference_provenance": True,
        "attackers": {
            "a1": {"prompt_version": PINNED_A1_PROMPT_VERSION},
            "a2": {"gower_policy": PINNED_A2_GOWER_POLICY},
            "a3": {"prompt_version": PINNED_A3_PROMPT_VERSION},
        },
    }
    assert preflight_formal_payload(payload) == []
    bad = {
        "require_reference_provenance": False,
        "attackers": {
            "a1": {"prompt_version": "a1_oneshot_v4_4_adversarial_objective"},
            "a2": {"gower_policy": GOWER_POLICY_LEGACY_V1},
            "a3": {"prompt_version": "a3_episodic_reflective_v2_4_adversarial_objective"},
        },
    }
    errors = preflight_formal_payload(bad)
    assert any("require_reference_provenance" in item for item in errors)
    assert any("A1 prompt_version" in item for item in errors)
    assert any("A2 gower_policy" in item for item in errors)
    assert any("A3 prompt_version" in item for item in errors)
