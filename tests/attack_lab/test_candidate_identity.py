"""Cross-attacker candidate identity contract."""

from __future__ import annotations

from attack_lab.candidate_identity import (
    CANDIDATE_IDENTITY_VERSION,
    canonical_candidate_fingerprint,
)


def test_same_projected_application_has_same_identity_despite_raw_representation() -> None:
    projected_a = {"income": 0.2, "customer_age": 40, "ignored_context": "x"}
    projected_b = {"customer_age": 40, "income": 0.2, "ignored_context": "y"}
    fields = ("income", "customer_age")
    first = canonical_candidate_fingerprint(
        anchor_id="anchor-1", projected_candidate=projected_a, action_fields=fields
    )
    second = canonical_candidate_fingerprint(
        anchor_id="anchor-1", projected_candidate=projected_b, action_fields=reversed(fields)
    )
    assert first == second


def test_candidate_identity_changes_with_projected_state_or_anchor() -> None:
    base = canonical_candidate_fingerprint(
        anchor_id="a", projected_candidate={"income": 0.2}
    )
    changed = canonical_candidate_fingerprint(
        anchor_id="a", projected_candidate={"income": 0.3}
    )
    other_anchor = canonical_candidate_fingerprint(
        anchor_id="b", projected_candidate={"income": 0.2}
    )
    assert len(base) == 64
    assert base != changed
    assert base != other_anchor
    assert CANDIDATE_IDENTITY_VERSION == "projected-action-state-v1"
