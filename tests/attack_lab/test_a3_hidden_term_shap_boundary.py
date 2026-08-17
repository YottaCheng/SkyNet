"""Regression: SHAP guards match real SHAP mentions, not English 'shape'."""

from __future__ import annotations

import pytest

from attack_lab.archive.contracts.a1_v4_1_contract import (
    classify_attacker_visible_term_context,
    prompt_contains_hidden_term,
    scan_attacker_visible_hidden_mentions,
)
from attack_lab.archive.contracts.a3_v2_2_contract import (
    ATTACK_OBJECTIVE_V2_2,
    PROMPT_VERSION_A3_V2_2,
    REFLECTION_MODE_DEFINITIONS_V2_2,
    REFLECTION_PURPOSE_V2_2,
    SELECTIONS_VS_HYPOTHESIS_NOTE,
    STATIC_RULE_DISCLOSURE_V2_2,
    assert_a3_v2_2_prompt_hard_contract,
)


def _minimal_v2_2_payload(*, hypothesis: str) -> dict:
    return {
        "prompt_version": PROMPT_VERSION_A3_V2_2,
        "attack_objective": ATTACK_OBJECTIVE_V2_2,
        "task": "Submit one candidate and obtain PASS within residual_m.",
        "reflection_purpose": REFLECTION_PURPOSE_V2_2,
        "episode_static_rules": STATIC_RULE_DISCLOSURE_V2_2,
        "selections_vs_hypothesis": SELECTIONS_VS_HYPOTHESIS_NOTE,
        "budget": {
            "residual_m": 2,
            "maximum_submitted_action_selections_this_query": 2,
        },
        "reflection_mode_definitions": dict(REFLECTION_MODE_DEFINITIONS_V2_2),
        "episodic_memory": [
            {
                "query_index": 4,
                "reflection_update": {
                    "mode": "ABANDON",
                    "hypothesis": hypothesis,
                },
            }
        ],
        "output_schema": {"properties": {"selections": {"maxProperties": 2}}},
    }


@pytest.mark.parametrize(
    ("text", "expect_match"),
    [
        ("Inspect SHAP next.", True),
        ("uses shap", True),
        ("SHAP values indicate risk", True),
        ("shap attribution over fields", True),
        ("shap_value", True),
        ('"feature_importance_or_shap"', True),
        ("feature_importance_or_shap", True),
        ("application shape", False),
        ("shaping a profile", False),
        ("profile reshaping", False),
        ("more plausible application shape.", False),
    ],
)
def test_prompt_contains_hidden_term_shap_match_matrix(
    text: str, expect_match: bool
) -> None:
    assert prompt_contains_hidden_term(text, "shap") is expect_match


def test_a3_v2_2_hard_contract_allows_shape_blocks_shap() -> None:
    assert_a3_v2_2_prompt_hard_contract(
        _minimal_v2_2_payload(
            hypothesis=(
                "These are less explored and can complete a more plausible "
                "application shape."
            )
        )
    )
    with pytest.raises(ValueError, match="hidden term 'shap'"):
        assert_a3_v2_2_prompt_hard_contract(
            _minimal_v2_2_payload(hypothesis="Inspect SHAP values next.")
        )
    with pytest.raises(ValueError, match="hidden term 'shap'"):
        assert_a3_v2_2_prompt_hard_contract(
            _minimal_v2_2_payload(hypothesis="Use shap attribution cues.")
        )


def test_audit_scanner_application_shape_no_hidden_exposure() -> None:
    text = "These edits complete a more plausible application shape."
    assert classify_attacker_visible_term_context(text, "shap") == "D_not_found"
    assert scan_attacker_visible_hidden_mentions(text) == []


def test_audit_scanner_shap_values_hidden_exposure() -> None:
    text = "Inspect SHAP values before the next query."
    assert classify_attacker_visible_term_context(text, "shap") == (
        "B_hidden_field_name"
    )
    findings = scan_attacker_visible_hidden_mentions(text)
    assert {"term": "shap", "class": "B_hidden_field_name"} in findings


def test_audit_scanner_feature_importance_or_shap_prohibition_wording() -> None:
    text = (
        '"unavailable_information": {"explicitly_unavailable": true, '
        '"feature_importance_or_shap": true}'
    )
    assert prompt_contains_hidden_term(text, "shap")
    assert classify_attacker_visible_term_context(text, "shap").startswith("C_")
