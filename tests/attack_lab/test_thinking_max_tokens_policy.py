"""Thinking-enabled max_tokens policy (800 → 2000); ThinkOff unchanged."""

from __future__ import annotations

from attack_lab.attackers.a1_planner import (
    DEFAULT_MAX_TOKENS,
    THINKING_ENABLED_MAX_TOKENS,
    resolve_max_tokens,
)
from attack_lab.benchmark_pins import PINNED_A3_PROMPT_VERSION


def test_thinking_on_maps_default_800_to_2000() -> None:
    assert DEFAULT_MAX_TOKENS == 800
    assert THINKING_ENABLED_MAX_TOKENS == 2000
    assert (
        resolve_max_tokens(thinking_disabled=False, max_tokens=800)
        == 2000
    )


def test_thinking_off_keeps_800() -> None:
    assert resolve_max_tokens(thinking_disabled=True, max_tokens=800) == 800


def test_thinking_on_preserves_explicit_higher_budget() -> None:
    assert resolve_max_tokens(thinking_disabled=False, max_tokens=4000) == 4000


def test_global_a3_benchmark_pin_remains_v2_3() -> None:
    # Active stack pin after archiving V2.4 adversarial-objective leaf.
    assert PINNED_A3_PROMPT_VERSION == (
        "a3_episodic_reflective_v2_3_public_reference_view"
    )
