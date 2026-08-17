"""Legitimate-sample drawing and threshold derivation for frozen D2-L."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from d2l.contract import (
    CALIBRATION_SAMPLE_N,
    CALIBRATION_SAMPLE_SEED,
    PRIMARY_REVIEW_BUDGET,
    REVIEW_BUDGETS,
    SANITY_SAMPLE_N,
    VALIDATION_SAMPLE_N,
)
from d2l.errors import D2LDataError


def sort_legitimate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_row_id" not in frame.columns:
        raise D2LDataError("Legitimate frame missing source_row_id.")
    return frame.sort_values("source_row_id").reset_index(drop=True)


def draw_disjoint_samples(
    frame: pd.DataFrame,
    *,
    seed: int = CALIBRATION_SAMPLE_SEED,
    n_cal: int = CALIBRATION_SAMPLE_N,
    n_val: int = VALIDATION_SAMPLE_N,
    n_sanity: int = SANITY_SAMPLE_N,
) -> dict[str, pd.DataFrame]:
    """Deterministic samples from Month-6 legitimate D1-PASS rows."""
    ordered = sort_legitimate_frame(frame)
    needed = n_cal + n_val + n_sanity
    if len(ordered) < needed:
        raise D2LDataError(
            f"Need {needed} legitimate D1-PASS rows; found {len(ordered)}."
        )
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(ordered))
    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal : n_cal + n_val]
    sanity_idx = perm[n_cal + n_val : n_cal + n_val + n_sanity]
    return {
        "calibration": ordered.iloc[cal_idx].reset_index(drop=True),
        "validation": ordered.iloc[val_idx].reset_index(drop=True),
        "sanity": ordered.iloc[sanity_idx].reset_index(drop=True),
    }


def sample_id_hash(row_ids: list[int]) -> str:
    payload = json.dumps([int(x) for x in row_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def thresholds_for_budgets(
    scores: np.ndarray,
    budgets: tuple[float, ...] = REVIEW_BUDGETS,
) -> pd.DataFrame:
    values = np.asarray(scores, dtype="float64")
    n = int(values.size)
    if n == 0:
        raise D2LDataError("Cannot derive thresholds from an empty score vector.")
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        threshold = float(np.quantile(values, 1.0 - float(budget)))
        n_review = int((values >= threshold).sum())
        rows.append(
            {
                "budget": float(budget),
                "label": "PRIMARY" if budget == PRIMARY_REVIEW_BUDGET else "SENSITIVITY",
                "threshold": threshold,
                "n_legitimate_d1_pass_sample": n,
                "n_review": n_review,
                "empirical_review_rate": n_review / n,
            }
        )
    return pd.DataFrame(rows)


def apply_thresholds(
    scores: np.ndarray,
    threshold_table: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(scores, dtype="float64")
    n = int(values.size)
    out: dict[str, dict[str, float | int]] = {}
    for record in threshold_table.to_dict(orient="records"):
        threshold = float(record["threshold"])
        n_review = int((values >= threshold).sum())
        key = f"{int(round(float(record['budget']) * 100))}pct"
        out[key] = {
            "budget": float(record["budget"]),
            "threshold": threshold,
            "n": n,
            "n_review": n_review,
            "n_clear": n - n_review,
            "empirical_review_rate": (n_review / n) if n else float("nan"),
        }
    return out


def score_collapse_report(scores: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype="float64")
    unique, counts = np.unique(values, return_counts=True)
    return {
        "n": int(values.size),
        "n_unique": int(unique.size),
        "min": float(values.min()) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
        "score_counts": {str(int(k) if float(k).is_integer() else k): int(v) for k, v in zip(unique, counts)},
        "collapsed": bool(unique.size <= 1),
    }


def sample_manifest(
    samples: Mapping[str, pd.DataFrame],
    *,
    seed: int,
    population_n: int,
) -> dict[str, Any]:
    cal_ids = [int(x) for x in samples["calibration"]["source_row_id"].tolist()]
    val_ids = [int(x) for x in samples["validation"]["source_row_id"].tolist()]
    sanity_ids = [int(x) for x in samples["sanity"]["source_row_id"].tolist()]
    overlap = set(cal_ids) & set(val_ids)
    if overlap:
        raise D2LDataError(f"Calibration/validation overlap: {len(overlap)} ids.")
    if set(cal_ids) & set(sanity_ids) or set(val_ids) & set(sanity_ids):
        raise D2LDataError("Sanity sample overlaps a calibration/validation id.")
    return {
        "seed": int(seed),
        "population_n": int(population_n),
        "population_filter": "month=6 AND fraud_bool=0 AND D1=PASS",
        "calibration_n": len(cal_ids),
        "validation_n": len(val_ids),
        "sanity_n": len(sanity_ids),
        "calibration_source_row_ids": cal_ids,
        "validation_source_row_ids": val_ids,
        "sanity_source_row_ids": sanity_ids,
        "calibration_ids_sha256": sample_id_hash(cal_ids),
        "validation_ids_sha256": sample_id_hash(val_ids),
        "sanity_ids_sha256": sample_id_hash(sanity_ids),
        "month7_opened": False,
    }
