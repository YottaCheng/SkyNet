"""Month-6 starting-case discovery for the development attack laboratory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from attack_lab.defender import FrozenArtefactPaths, FrozenXGBoostDefender
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR
from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.pipeline import load_prepared_splits

DEFAULT_RAW_PATH = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")


class AttackLabCaseError(RuntimeError):
    """Raised when a development starting case cannot be resolved."""


FORBIDDEN_SPLIT_NAMES = frozenset({"test", "month7", "month_7", "holdout_test"})


@dataclass(frozen=True)
class StartingCase:
    """One reproducible month-6 true-positive application under frozen D1."""

    case_id: str
    source_row_id: int
    label: int
    features: dict[str, Any]
    initial_score: float
    initial_decision: str
    data_split: str = "dev_month6"


def assert_month6_only(split_name: str) -> None:
    """Refuse any attempt to select the sealed month-7 test split."""
    normalised = split_name.strip().lower()
    if (
        normalised in FORBIDDEN_SPLIT_NAMES
        or "month7" in normalised
        or "month_7" in normalised
    ):
        raise AttackLabCaseError(
            "Month 7 / sealed test split cannot be selected in the attack laboratory."
        )
    if normalised not in {"dev", "dev_month6", "month6", "month_6", "development"}:
        raise AttackLabCaseError(
            f"Unsupported split {split_name!r}; only month-6 development is permitted."
        )


def load_month6_feature_frame(
    raw_path: Path,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> pd.DataFrame:
    """Load prepared month-6 features and immediately discard the test split."""
    assert_month6_only("dev_month6")
    prepared = load_prepared_splits(raw_path, data_config)
    try:
        return prepared.views["dev"].X.copy()
    finally:
        # Never retain month 7 / test handles for attack-lab case selection.
        del prepared


def discover_true_positive_case_ids(
    artefact_dir: Path | None = None,
    *,
    threshold: float | None = None,
) -> list[int]:
    """Return source row_ids that are fraud and BLOCKED under frozen D1.

    Prefers the frozen development score CSV so discovery does not rescore
    unless the CSV is absent.
    """
    paths = FrozenArtefactPaths.from_dir(artefact_dir or DEFAULT_C1_ARTEFACT_DIR)
    paths.require_present()
    scores = pd.read_csv(paths.scores_path)
    required = {"row_id", "y_true", "y_score"}
    missing = required - set(scores.columns)
    if missing:
        raise AttackLabCaseError(
            f"Score CSV missing columns {sorted(missing)}: {paths.scores_path}"
        )
    if threshold is None:
        payload = json.loads(paths.threshold_path.read_text(encoding="utf-8"))
        threshold = float(payload["threshold"])
    mask = (scores["y_true"].astype(int) == 1) & (scores["y_score"] >= threshold)
    row_ids = scores.loc[mask, "row_id"].astype(int).tolist()
    if not row_ids:
        raise AttackLabCaseError("No true-positive BLOCKED cases found on month 6.")
    return row_ids


def load_starting_case(
    case_id: str | int,
    *,
    raw_path: Path,
    defender: FrozenXGBoostDefender | None = None,
    artefact_dir: Path | None = None,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> StartingCase:
    """Load a month-6 TP case by stable source row_id."""
    assert_month6_only("dev_month6")
    row_id = int(case_id)
    tp_ids = set(discover_true_positive_case_ids(artefact_dir))
    if row_id not in tp_ids:
        raise AttackLabCaseError(
            f"Case {row_id} is not a frozen-D1 true positive (fraud + initially BLOCKED) "
            "on month 6. Naturally missed fraud cases are not valid starting cases."
        )

    frame = load_month6_feature_frame(raw_path, data_config)
    if row_id not in frame.index:
        raise AttackLabCaseError(
            f"Case row_id {row_id} not found in prepared month-6 feature frame."
        )
    row = frame.loc[row_id]
    features = {name: _to_python(row[name]) for name in data_config.feature_columns}

    if defender is None:
        defender = FrozenXGBoostDefender.from_artefact_dir(artefact_dir, data_config)
    internal = defender.score_application(features)
    if internal.decision != "BLOCK":
        raise AttackLabCaseError(
            f"Case {row_id} did not BLOCK under the loaded frozen defender "
            f"(decision={internal.decision}, score={internal.risk_score})."
        )
    return StartingCase(
        case_id=str(row_id),
        source_row_id=row_id,
        label=1,
        features=features,
        initial_score=internal.risk_score,
        initial_decision=internal.decision,
        data_split="dev_month6",
    )


def _to_python(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    return value
