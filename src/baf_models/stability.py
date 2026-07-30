"""Fixed-seed stability helpers for the frozen XGBoost baseline.

Loops over a pre-specified seed list by changing only ``random_state``.
This module does not select a best seed, does not tune hyperparameters,
and never scores month 7.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

import numpy as np

from baf_models.training import VariantFitResult
from baf_models.xgboost_model import XGBoostBaselineConfig

#: Prespecified seeds for the stability check. Order is fixed; no duplicates.
STABILITY_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

AGGREGATED_METRICS: tuple[str, ...] = (
    "auprc",
    "auroc",
    "tpr_at_fpr_le_5pct",
)


class StabilityError(RuntimeError):
    """Raised when the stability protocol is violated."""


def validate_stability_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Accept only the frozen seed list, with no duplicates or reordering."""
    resolved = tuple(int(s) for s in seeds)
    if resolved != STABILITY_SEEDS:
        raise StabilityError(
            f"Stability seeds must be exactly {STABILITY_SEEDS}; got {resolved}."
        )
    if len(set(resolved)) != len(resolved):
        raise StabilityError("Stability seeds must not contain duplicates.")
    return resolved


def config_with_seed(
    base: XGBoostBaselineConfig, seed: int
) -> XGBoostBaselineConfig:
    """Return a config identical to ``base`` except for ``random_state``."""
    if seed not in STABILITY_SEEDS:
        raise StabilityError(
            f"Seed {seed} is not in the frozen stability list {STABILITY_SEEDS}."
        )
    modified = replace(base, random_state=seed)
    assert_only_random_state_differs(base, modified, seed)
    return modified


def assert_only_random_state_differs(
    base: XGBoostBaselineConfig,
    modified: XGBoostBaselineConfig,
    expected_seed: int,
) -> None:
    """Fail if any field other than ``random_state`` changed."""
    base_dict = asdict(base)
    modified_dict = asdict(modified)
    if modified_dict["random_state"] != expected_seed:
        raise StabilityError(
            f"Expected random_state={expected_seed}, got "
            f"{modified_dict['random_state']}."
        )
    base_dict.pop("random_state")
    modified_dict.pop("random_state")
    if base_dict != modified_dict:
        changed = {
            key: (base_dict[key], modified_dict[key])
            for key in base_dict
            if base_dict[key] != modified_dict[key]
        }
        raise StabilityError(
            "Stability configs may differ only in random_state; "
            f"unexpected changes: {changed}."
        )


def metrics_row_from_result(result: VariantFitResult, seed: int) -> dict[str, Any]:
    """Extract the per-seed development metrics record."""
    ranking = result.evaluation["threshold_independent"]
    operating = result.evaluation["operating_point_fpr_le_max"]
    return {
        "seed": int(seed),
        "auprc": float(ranking["auprc"]),
        "auroc": float(ranking["auroc"]),
        "brier_score": float(ranking["brier_score"]),
        "tpr_at_fpr_le_5pct": float(operating["tpr"]),
        "fpr_at_fpr_le_5pct": float(operating["fpr"]),
        "precision_at_fpr_le_5pct": float(operating["precision"]),
        "review_rate_at_fpr_le_5pct": float(operating["review_rate"]),
        "threshold_at_fpr_le_5pct": float(operating["threshold"]),
        "tp": int(operating["tp"]),
        "fp": int(operating["fp"]),
        "tn": int(operating["tn"]),
        "fn": int(operating["fn"]),
        "fit_seconds": float(result.fit_seconds),
        "predict_seconds": float(result.predict_seconds),
        "split": "development",
        "month": 6,
        "random_state": int(seed),
    }


def summarise_numeric(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample standard deviation (ddof=1), min and max."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise StabilityError("Cannot aggregate an empty metric series.")
    if arr.size == 1:
        sample_std = 0.0
    else:
        sample_std = float(np.std(arr, ddof=1))
    return {
        "mean": float(np.mean(arr)),
        "sample_std": sample_std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def deltas_versus_lr(
    seed_rows: Sequence[Mapping[str, Any]],
    lr_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Per-seed differences against saved unweighted LR development metrics."""
    if lr_metrics.get("split") != "development" or lr_metrics.get("month") != 6:
        raise StabilityError("LR comparison metrics must be development / month 6.")
    lr_ranking = lr_metrics["threshold_independent"]
    lr_operating = lr_metrics["operating_point_fpr_le_max"]
    rows: list[dict[str, Any]] = []
    for row in seed_rows:
        rows.append(
            {
                "seed": int(row["seed"]),
                "delta_auprc": float(row["auprc"]) - float(lr_ranking["auprc"]),
                "delta_auroc": float(row["auroc"]) - float(lr_ranking["auroc"]),
                "delta_brier_score": (
                    float(row["brier_score"]) - float(lr_ranking["brier_score"])
                ),
                "delta_tpr_at_fpr_le_5pct": (
                    float(row["tpr_at_fpr_le_5pct"]) - float(lr_operating["tpr"])
                ),
                "delta_precision_at_fpr_le_5pct": (
                    float(row["precision_at_fpr_le_5pct"])
                    - float(lr_operating["precision"])
                ),
                "delta_review_rate_at_fpr_le_5pct": (
                    float(row["review_rate_at_fpr_le_5pct"])
                    - float(lr_operating["review_rate"])
                ),
                "xgb_auprc_gt_lr_auprc": bool(
                    float(row["auprc"]) > float(lr_ranking["auprc"])
                ),
            }
        )
    return rows


def build_stability_summary(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    lr_metrics: Mapping[str, Any],
    base_random_state: int,
) -> dict[str, Any]:
    """Aggregate seed rows; never ranks or selects a preferred seed."""
    validate_stability_seeds([int(row["seed"]) for row in seed_rows])
    if base_random_state != 42:
        raise StabilityError(
            "Formal candidate random_state must remain 42; "
            f"got {base_random_state}."
        )
    aggregates = {
        metric: summarise_numeric([float(row[metric]) for row in seed_rows])
        for metric in AGGREGATED_METRICS
    }
    deltas = deltas_versus_lr(seed_rows, lr_metrics)
    seed_42 = next(row for row in seed_rows if int(row["seed"]) == 42)
    return {
        "split": "development",
        "month": 6,
        "seeds": list(STABILITY_SEEDS),
        "formal_candidate_random_state": 42,
        "note": (
            "Fixed-seed stability check only. Seeds are not ranked and the "
            "formal candidate remains random_state=42. FPR<=5% is a "
            "literature-linked experimental operating point, not a real-bank "
            "tolerance. Month 7 was not used."
        ),
        "aggregates": aggregates,
        "seed_42_within_observed_range": {
            metric: bool(
                aggregates[metric]["min"]
                <= float(seed_42[metric])
                <= aggregates[metric]["max"]
            )
            for metric in AGGREGATED_METRICS
        },
        "all_seeds_auprc_gt_lr": all(row["xgb_auprc_gt_lr_auprc"] for row in deltas),
        "deltas_versus_unweighted_lr": deltas,
        "per_seed": list(seed_rows),
    }
