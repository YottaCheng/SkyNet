"""Contract-level invariants for D2-L V1."""

from __future__ import annotations

from d2l.contract import (
    APPLICATION_FIELDS,
    FIELD_DESCRIPTIONS,
    FORBIDDEN_INPUT_KEYS,
    MODEL_ID,
    PROMPT_VERSION,
    THINKING_DISABLED,
    contract_payload,
)
from d2l.prompt import prompt_text, system_prompt


def test_model_and_thinking_pin() -> None:
    assert MODEL_ID == "deepseek-v4-pro"
    assert THINKING_DISABLED is True
    payload = contract_payload()
    assert payload["receives_d1_numeric_score"] is False
    assert payload["receives_d2s_relationships"] is False
    assert payload["llm_does_not_choose_threshold"] is True
    assert payload["month7_opened"] is False
    assert payload["prompt_version"] == PROMPT_VERSION


def test_field_dictionary_covers_core_fields_only() -> None:
    assert tuple(FIELD_DESCRIPTIONS) == APPLICATION_FIELDS
    assert "credit_risk_score" not in FIELD_DESCRIPTIONS
    assert "days_since_request" not in FIELD_DESCRIPTIONS
    assert "device_fraud_count" not in FIELD_DESCRIPTIONS
    assert "fraud_bool" not in FIELD_DESCRIPTIONS
    assert "month" not in FIELD_DESCRIPTIONS


def test_forbidden_keys_include_d1_and_attack_channels() -> None:
    required = {
        "fraud_bool",
        "d1_score",
        "d1_probability",
        "d1_threshold",
        "shap",
        "d2_score",
        "attacker_kind",
        "changed_field_mask",
        "successful_query",
        "provenance",
        "reference_pool_membership",
    }
    assert required <= FORBIDDEN_INPUT_KEYS


def test_prompt_omits_d2s_relationships_and_d1_score() -> None:
    text = system_prompt() + prompt_text()
    assert "consistency_risk_score" in text
    assert "CLEAR" in text
    assert "Do not output a final CLEAR" in text or "do not output a final CLEAR" in text.lower()
