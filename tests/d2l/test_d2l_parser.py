"""Parser and prompt-boundary tests for D2-L."""

from __future__ import annotations

import pytest

from d2l.errors import D2LContractError, D2LParseError
from d2l.parser import parse_reviewer_output
from d2l.prompt import assert_prompt_has_no_forbidden_substrings, system_prompt


def test_parse_valid_json() -> None:
    text = (
        '{"consistency_risk_score": 12, "reason_codes": ["age/housing mismatch"], '
        '"summary": "The profile is mostly coherent. One contact field is odd."}'
    )
    parsed = parse_reviewer_output(text)
    assert parsed["consistency_risk_score"] == 12
    assert parsed["reason_codes"] == ["age/housing mismatch"]
    assert "mostly coherent" in parsed["summary"]


def test_parse_rejects_non_integer_and_out_of_range() -> None:
    with pytest.raises(D2LParseError):
        parse_reviewer_output(
            '{"consistency_risk_score": 12.4, "reason_codes": [], "summary": "x."}'
        )
    with pytest.raises(D2LParseError):
        parse_reviewer_output(
            '{"consistency_risk_score": 101, "reason_codes": [], "summary": "x."}'
        )
    with pytest.raises(D2LParseError):
        parse_reviewer_output(
            '{"consistency_risk_score": -1, "reason_codes": [], "summary": "x."}'
        )


def test_parse_accepts_integer_valued_float() -> None:
    parsed = parse_reviewer_output(
        '{"consistency_risk_score": 0.0, "reason_codes": [], "summary": "Coherent."}'
    )
    assert parsed["consistency_risk_score"] == 0


def test_prompt_rejects_relationship_leakage() -> None:
    with pytest.raises(D2LContractError):
        assert_prompt_has_no_forbidden_substrings(system_prompt() + "\nC01 rarity table")
    assert_prompt_has_no_forbidden_substrings(system_prompt())
