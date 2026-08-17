"""Prove D2-S inference ignores leakage channels."""

from __future__ import annotations

from copy import deepcopy
from inspect import getsource

from d2.calibrate import extract_d1_pass_features
from d2.contract import FORBIDDEN_INFERENCE_KEYS
from d2.scoring import D2SScorer, application_to_frame, fit_d2s_scorer


def _reference(synthetic_frame):
    return synthetic_frame.loc[
        synthetic_frame["month"].between(0, 5) & synthetic_frame["fraud_bool"].eq(0)
    ].copy()


def _application(synthetic_frame) -> dict:
    row = synthetic_frame.loc[synthetic_frame["month"].eq(6)].iloc[1]
    from baf_data.config import FROZEN_CONFIG

    return {name: row[name] for name in FROZEN_CONFIG.feature_columns}


def test_forbidden_keys_do_not_change_score(synthetic_frame) -> None:
    scorer = fit_d2s_scorer(_reference(synthetic_frame), raw_sha256="synthetic")
    base = _application(synthetic_frame)
    clean = scorer.score(base)
    leaked = deepcopy(base)
    leaked.update(
        {
            "fraud_bool": 1,
            "d1_score": 0.99,
            "d1_probability": 0.99,
            "risk_score": 0.99,
            "shap": {"income": 1.0},
            "attacker_id": "A3",
            "attack_history": ["x"],
            "changed_field_mask": {"income": 1},
            "provenance": {"pool": "secret"},
            "reference_pool_membership": True,
            "attack_success": True,
            "success": True,
            "month": 7,
        }
    )
    dirty = scorer.score(leaked)
    assert dirty == clean


def test_application_to_frame_drops_leakage_keys(synthetic_frame) -> None:
    app = _application(synthetic_frame)
    app["fraud_bool"] = 1
    app["d1_score"] = 0.4
    frame = application_to_frame(app)
    assert "fraud_bool" not in frame.columns
    assert "d1_score" not in frame.columns
    assert "month" not in frame.columns


def test_score_source_does_not_read_forbidden_keys() -> None:
    source = getsource(D2SScorer.score) + getsource(application_to_frame)
    for key in sorted(FORBIDDEN_INFERENCE_KEYS):
        if key in {"month"}:
            continue
        assert f'features["{key}"]' not in source
        assert f"features['{key}']" not in source


def test_fit_refuses_if_month7_opened_flag_set(synthetic_frame) -> None:
    from d2.errors import D2FitError

    reference = _reference(synthetic_frame)
    try:
        fit_d2s_scorer(reference, raw_sha256="x", month7_opened=True)
    except D2FitError as exc:
        assert "Month 7" in str(exc)
    else:
        raise AssertionError("expected D2FitError")


def test_extract_d1_pass_features_uses_candidate_only() -> None:
    episode = {
        "steps": [
            {
                "internal_defence": {"decision": "BLOCK", "risk_score": 0.9},
                "validity": {
                    "is_valid": True,
                    "candidate_features": {"payment_type": "AA"},
                },
            },
            {
                "internal_defence": {"decision": "PASS", "risk_score": 0.01},
                "validity": {
                    "is_valid": True,
                    "candidate_features": {
                        "payment_type": "AC",
                        "bank_months_count": None,
                    },
                },
                "research_meta": {"attacker_id": "A0", "success": True},
            },
        ]
    }
    found = extract_d1_pass_features(episode)
    assert len(found) == 1
    assert found[0]["payment_type"] == "AC"
    assert "research_meta" not in found[0]
    assert "risk_score" not in found[0]
