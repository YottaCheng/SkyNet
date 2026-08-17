"""Contract-level invariants for D2-S V1."""

from __future__ import annotations

from pathlib import Path

from d2.contract import (
    AGGREGATION_FORMULA,
    CALIBRATION_MONTHS,
    FORBIDDEN_INFERENCE_KEYS,
    REFERENCE_MONTHS,
    RELATIONSHIP_IDS,
    SCORE_CONTRACT_ID,
    SEALED_MONTHS,
    score_contract_payload,
)
from d2.relationships import pair_labels


def test_exactly_eight_pairwise_relationships() -> None:
    assert RELATIONSHIP_IDS == (
        "C01",
        "C14",
        "C13",
        "C09",
        "C03",
        "C10",
        "C11",
        "C15",
    )
    payload = score_contract_payload()
    assert payload["higher_order_relationships"] == []
    assert payload["threshold_in_scorer"] is False
    assert payload["editability_is_weight"] is False


def test_month_boundaries() -> None:
    assert REFERENCE_MONTHS == (0, 1, 2, 3, 4, 5)
    assert CALIBRATION_MONTHS == (6,)
    assert SEALED_MONTHS == (7,)


def test_leakage_keys_are_declared() -> None:
    required = {
        "fraud_bool",
        "d1_score",
        "d1_probability",
        "shap",
        "attacker_id",
        "attack_history",
        "changed_field_mask",
        "provenance",
        "reference_pool_membership",
        "attack_success",
    }
    assert required <= FORBIDDEN_INFERENCE_KEYS


def test_no_three_or_four_way_in_d2_source() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "d2"
    forbidden_tokens = (
        "three_way",
        "four_way",
        "3-way",
        "4-way",
        "I(age;hous,emp)",
        "triple_score",
    )
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{path.name} contains {token!r}"


def test_pair_labels_rejects_unknown_relationship(synthetic_frame) -> None:
    from d2.errors import D2FitError
    from d2.relationships import fit_binning

    bins = fit_binning(synthetic_frame)
    try:
        pair_labels(synthetic_frame, "C99", bins)
    except D2FitError as exc:
        assert "Unknown pairwise" in str(exc)
    else:
        raise AssertionError("expected D2FitError")


def test_aggregation_formula_is_equal_mean() -> None:
    assert " / 8" in AGGREGATION_FORMULA
    assert SCORE_CONTRACT_ID.startswith("d2s-v1.0.0")
