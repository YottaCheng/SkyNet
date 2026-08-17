"""Prove D2-L application views ignore leakage channels."""

from __future__ import annotations

from copy import deepcopy

from baf_data.config import FROZEN_CONFIG
from d2l.application_view import application_view, serialize_application_view
from d2l.contract import APPLICATION_FIELDS, FORBIDDEN_INPUT_KEYS
from d2l.prompt import build_messages


def _application(synthetic_frame) -> dict:
    row = synthetic_frame.loc[synthetic_frame["month"].eq(6)].iloc[1]
    return {name: row[name] for name in FROZEN_CONFIG.feature_columns}


def test_view_drops_leakage_keys(synthetic_frame) -> None:
    base = _application(synthetic_frame)
    leaked = deepcopy(base)
    leaked.update(
        {
            "fraud_bool": 1,
            "d1_score": 0.012,
            "d1_probability": 0.012,
            "d1_threshold": 0.047,
            "shap": {"income": 1.0},
            "d2_score": 0.9,
            "attacker_kind": "a3",
            "changed_field_mask": {"income": 1},
            "successful_query": 2,
            "provenance": {"pool": "secret"},
            "reference_pool_membership": True,
            "month": 7,
            "y_score": 0.01,
        }
    )
    clean = application_view(base)
    dirty = application_view(leaked)
    assert clean == dirty
    assert list(clean) == list(APPLICATION_FIELDS)
    for key in FORBIDDEN_INPUT_KEYS:
        assert key not in clean


def test_nested_application_constructor_matches_flat(synthetic_frame) -> None:
    base = _application(synthetic_frame)
    nested = {
        "application": base,
        "attack_metadata": {"attacker_kind": "a0", "successful_query": 1},
        "defender_metadata": {"d1_score": 0.02, "d1_threshold": 0.047},
        "record_id": "A0:1:seed_1:q1",
    }
    assert application_view(nested) == application_view(base)


def test_messages_do_not_contain_forbidden_keys(synthetic_frame) -> None:
    base = _application(synthetic_frame)
    nested = {
        "application": base,
        "defender_metadata": {"d1_score": 0.02, "d1_decision": "PASS"},
        "attack_metadata": {"attacker_kind": "a2"},
    }
    view = application_view(nested)
    blob = serialize_application_view(view)
    messages = build_messages(view)
    joined = blob + messages[0]["content"] + messages[1]["content"]
    for key in (
        "d1_score",
        "fraud_bool",
        "attacker_kind",
        "changed_field_mask",
        "d2_score",
        "shap",
    ):
        assert f'"{key}"' not in joined
        if key not in {"d1_score"}:
            assert key not in messages[0]["content"].lower()
