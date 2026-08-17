"""External DeepSeek payload allowlist tests."""

from __future__ import annotations

import pytest

from attack_lab.outbound_payload import (
    OutboundPayloadError,
    audit_outbound_payload,
    sanitise_reference_pool,
    temporary_episode_id,
)


def test_sanitise_reference_pool_replaces_source_linkable_ids() -> None:
    payload = sanitise_reference_pool(
        {
            "anchor_id": "839418",
            "K": 1,
            "generation_seed": 4,
            "pool_fingerprint": "secret-provenance",
            "context_fields": ["income"],
            "action_fields": ["income"],
            "profiles": [
                {"profile_id": "raw-row-123", "generation_seed": 9, "fields": {"income": 0.2}}
            ],
        },
        temporary_anchor_id=temporary_episode_id(7),
        allowed_fields={"income"},
    )
    assert payload["anchor_id"].startswith("dev-anchor-")
    assert payload["profiles"][0]["profile_id"] == "ref-01"
    blob = str(payload)
    assert "839418" not in blob
    assert "raw-row-123" not in blob
    assert "secret-provenance" not in blob
    assert "generation_seed" not in blob


def test_outbound_audit_rejects_forbidden_key_path_and_raw_id() -> None:
    allowed_top = ("anchor",)
    with pytest.raises(OutboundPayloadError, match="Denied outbound key"):
        audit_outbound_payload(
            {"anchor": {"case_id": temporary_episode_id(1), "fraud_bool": 1}},
            allowed_top_level_keys=allowed_top,
            allowed_feature_fields={"income"},
        )
    with pytest.raises(OutboundPayloadError, match="absolute path"):
        audit_outbound_payload(
            {"anchor": {"case_id": temporary_episode_id(1), "note": "/Users/x/a"}},
            allowed_top_level_keys=allowed_top,
            allowed_feature_fields={"income"},
        )
    with pytest.raises(OutboundPayloadError, match="Non-temporary"):
        audit_outbound_payload(
            {"anchor": {"case_id": "839418", "visible_fields": {"income": 0.2}}},
            allowed_top_level_keys=allowed_top,
            allowed_feature_fields={"income"},
        )


def test_outbound_audit_records_exact_field_allowlist() -> None:
    payload = {
        "anchor": {
            "case_id": temporary_episode_id(1),
            "visible_fields": {"income": 0.2},
        },
        "episode_memory": [{"public_label": "BLOCK", "changes": {"income": 0.3}}],
    }
    audit = audit_outbound_payload(
        payload,
        allowed_top_level_keys=("anchor", "episode_memory"),
        allowed_feature_fields={"income"},
    )
    assert audit["preflight"] == "PASS"
    assert audit["external_feature_fields"] == ["income"]
    assert audit["public_feedback_labels_present"] == ["BLOCK"]
