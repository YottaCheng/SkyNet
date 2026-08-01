"""Tests for defender defaults, frozen surface, and label_only feedback."""

from __future__ import annotations

from pathlib import Path

import pytest

from attack_lab.cli import build_parser
from attack_lab.defender import AttackLabDefenderError, FrozenXGBoostDefender
from attack_lab.feedback import FeedbackPolicy
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR
from attack_lab.types import InternalDefenceResult, ValidityResult


def test_d1_is_default_defence() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--case-id",
            "1",
            "--mutable-fields",
            "income",
            "--max-attempts",
            "3",
        ]
    )
    assert args.defence == "d1"
    assert args.attacker == "human"
    assert args.feedback == "label_only"


def test_frozen_defender_has_no_training_refit_path() -> None:
    # Construct without loading artefacts by injecting a stub pipeline object.
    class _StubPipe:
        def predict_proba(self, frame):  # noqa: ANN001
            import numpy as np

            return np.array([[0.2, 0.8]])

    defender = FrozenXGBoostDefender(
        pipeline=_StubPipe(),  # type: ignore[arg-type]
        threshold=0.5,
        artefact_id="stub",
        feature_columns=("income",),
        artefact_dir=Path("/tmp"),
        config_payload={},
    )
    with pytest.raises(AttackLabDefenderError, match="no training/refit"):
        defender.fit()
    with pytest.raises(AttackLabDefenderError, match="no training/refit"):
        defender.refit()


def test_label_only_does_not_expose_score_or_threshold() -> None:
    policy = FeedbackPolicy(mode="label_only")
    internal = InternalDefenceResult(
        risk_score=0.123456,
        threshold=0.04724566638469696,
        decision="BLOCK",
        runtime_ms=1.0,
        defender_name="d1",
        artefact_id="x",
    )
    public = policy.for_scored(internal, attempt=1, remaining_attempts=2)
    payload = public.__dict__
    assert "risk_score" not in payload
    assert "threshold" not in payload
    assert "0.123456" not in public.message
    assert "0.047" not in public.message
    assert public.label == "BLOCK"

    invalid = policy.for_invalid(
        ValidityResult(False, ("immutable",), None),
        attempt=1,
        remaining_attempts=2,
    )
    assert invalid.label == "INVALID"
    assert "0.123456" not in invalid.message


def test_frozen_c1_artefacts_present() -> None:
    pipeline = DEFAULT_C1_ARTEFACT_DIR / "fitted_pipeline.joblib"
    threshold = DEFAULT_C1_ARTEFACT_DIR / "development_month6_threshold_selection.json"
    assert pipeline.is_file()
    assert threshold.is_file()
