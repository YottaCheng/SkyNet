"""V2 discrete-rubric unit tests."""

from __future__ import annotations

import json

from d2l.v2_contract import DIMENSION_IDS
from d2l.v2_parser import parse_batch_scores, parse_rubric
from d2l.v2_score import (
    provisional_threshold,
    sanity_decision,
    sentinel_stability,
    total_score,
)


def _judgments(*labels: str) -> dict[str, str]:
    return {dim_id: labels[i] for i, dim_id in enumerate(DIMENSION_IDS)}


def test_label_mapping_and_total() -> None:
    all_good = _judgments(*(["GOOD"] * 8))
    all_bad = _judgments(*(["BAD"] * 8))
    mixed = _judgments("GOOD", "UNCERTAIN", "BAD", "GOOD", "GOOD", "GOOD", "GOOD", "GOOD")
    assert total_score(all_good) == 0
    assert total_score(all_bad) == 16
    assert total_score(mixed) == 3


def test_parse_rubric_requires_eight_fixed_ids() -> None:
    dimensions = [
        {
            "id": dim_id,
            "title": dim_id,
            "good": "coherent",
            "uncertain": "unclear",
            "bad": "incoherent",
        }
        for dim_id in DIMENSION_IDS
    ]
    parsed = parse_rubric(json.dumps({"dimensions": dimensions}))
    assert [row["id"] for row in parsed["dimensions"]] == list(DIMENSION_IDS)
    assert parsed["weights"] is None


def test_parse_batch_scores_accepts_nested_or_flat_labels() -> None:
    app_id = "L0001"
    payload = {
        "results": [
            {
                "id": app_id,
                "judgments": {
                    dim_id: {"judgment": "GOOD", "evidence": "n/a"}
                    for dim_id in DIMENSION_IDS
                },
            }
        ]
    }
    found = parse_batch_scores(json.dumps(payload), [app_id])
    assert found[app_id][DIMENSION_IDS[0]] == "GOOD"


def test_provisional_threshold_does_not_split_ties() -> None:
    scores = [2] * 90 + [6] * 10
    info = provisional_threshold(scores, target=0.10)
    assert info["random_split_of_ties"] is False
    assert info["final_month6_operating_point"] is False
    assert info["empirical_review_rate"] == 0.10
    assert info["threshold"] == 6


def test_spread_without_attack_shift_fails_discrimination() -> None:
    legit = [i % 10 for i in range(100)]
    attacks = [3] * 20
    info = provisional_threshold(legit)
    decision = sanity_decision(
        legit_scores=legit,
        attack_scores=attacks,
        threshold_info=info,
        sentinel={"acceptable": True},
    )
    assert info["empirical_review_rate"] not in (0.0, 1.0)
    assert decision["conclusion"] == "FAIL_NO_PRELIMINARY_DISCRIMINATION"


def test_upward_attack_shift_can_pass() -> None:
    legit = [i % 10 for i in range(100)]
    attacks = [12] * 20
    info = provisional_threshold(legit)
    decision = sanity_decision(
        legit_scores=legit,
        attack_scores=attacks,
        threshold_info=info,
        sentinel={"acceptable": True},
    )
    assert decision["conclusion"] == "PASS_TO_LARGER_MONTH6_CALIBRATION"


def test_constant_scores_are_collapse() -> None:
    scores = [8] * 100
    info = provisional_threshold(scores)
    decision = sanity_decision(
        legit_scores=scores,
        attack_scores=[12] * 20,
        threshold_info=info,
        sentinel={"acceptable": True},
    )
    assert decision["conclusion"] == "FAIL_SCORE_COLLAPSE"


def test_scoring_payload_has_no_attack_metadata() -> None:
    from d2l.v2_prompt import packed_applications, scoring_messages

    packed = packed_applications([("A0001", {"income": 0.5, "customer_age": 30})])
    blob = json.dumps(packed)
    assert "attacker" not in blob
    assert "d1_score" not in blob
    assert "condition_id" not in blob
    messages = scoring_messages(
        {
            "dimensions": [
                {"id": dim_id, "title": dim_id, "good": "g", "uncertain": "u", "bad": "b"}
                for dim_id in DIMENSION_IDS
            ]
        },
        [("A0001", {"income": 0.5, "customer_age": 30})],
    )
    joined = messages[0]["content"] + messages[1]["content"]
    assert "attacker" not in joined.lower()
    assert "fraud_bool" not in joined


def test_sentinel_material_difference() -> None:
    a = {"L0001": _judgments(*(["GOOD"] * 8))}
    b = {"L0001": _judgments(*(["BAD"] * 8))}
    report = sentinel_stability(a, b)
    assert report["acceptable"] is False
    assert report["rows"][0]["materially_different"] is True
