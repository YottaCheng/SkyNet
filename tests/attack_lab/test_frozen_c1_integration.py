"""Optional live integration against the serialised frozen C1 artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack_lab.defender import FrozenXGBoostDefender
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR
from baf_data.config import FROZEN_CONFIG


def _artefacts_ready() -> bool:
    d = DEFAULT_C1_ARTEFACT_DIR
    return (d / "fitted_pipeline.joblib").is_file() and (
        d / "development_month6_threshold_selection.json"
    ).is_file()


@pytest.mark.skipif(not _artefacts_ready(), reason="Frozen C1 artefacts not present")
def test_load_and_score_frozen_c1(baseline_features) -> None:
    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    threshold_path = (
        DEFAULT_C1_ARTEFACT_DIR / "development_month6_threshold_selection.json"
    )
    expected = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
    assert defender.threshold == pytest.approx(expected)
    assert defender.name == "d1_frozen_c1_xgboost"

    # Replace categorical values with fitted vocabulary entries.
    vocab = defender.categorical_vocabularies()
    features = dict(baseline_features)
    for name, categories in vocab.items():
        features[name] = categories[0]

    result = defender.score_application(features)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.threshold == pytest.approx(expected)
    assert result.decision in {"PASS", "BLOCK"}
    assert set(features) == set(FROZEN_CONFIG.feature_columns)

    with pytest.raises(Exception, match="no training/refit"):
        defender.fit([[0]])
