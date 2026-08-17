"""Code-side V2 scoring: label maps, totals, provisional threshold, sanity decision."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from d2l.v2_contract import (
    COLLAPSE_MAX_UNIQUE,
    COLLAPSE_MODAL_SHARE,
    DIMENSION_IDS,
    DISCRIMINATION_REVIEW_LIFT,
    LABEL_TO_POINTS,
    MAX_TOTAL,
    MIN_TOTAL,
    TARGET_REVIEW_BUDGET,
)


def dimension_points(judgments: Mapping[str, str]) -> dict[str, int]:
    return {dim_id: int(LABEL_TO_POINTS[judgments[dim_id]]) for dim_id in DIMENSION_IDS}


def total_score(judgments: Mapping[str, str]) -> int:
    points = dimension_points(judgments)
    total = int(sum(points.values()))
    if total < MIN_TOTAL or total > MAX_TOTAL:
        raise ValueError(f"total_score {total} outside {MIN_TOTAL}-{MAX_TOTAL}.")
    return total


def score_histogram(scores: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(x) for x in scores)
    return {str(k): int(counts[k]) for k in range(MIN_TOTAL, MAX_TOTAL + 1)}


def score_summary(scores: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(list(scores), dtype="int64")
    n = int(values.size)
    unique = sorted(int(x) for x in np.unique(values))
    counts = Counter(int(x) for x in values)
    modal = max(counts.values()) / n if n else float("nan")
    return {
        "n": n,
        "n_unique": len(unique),
        "unique_scores": unique,
        "min": int(values.min()) if n else None,
        "q1": float(np.quantile(values, 0.25)) if n else None,
        "median": float(np.median(values)) if n else None,
        "q3": float(np.quantile(values, 0.75)) if n else None,
        "max": int(values.max()) if n else None,
        "mean": float(values.mean()) if n else None,
        "modal_share": float(modal),
        "histogram": score_histogram(values.tolist()),
    }


def provisional_threshold(
    scores: Sequence[int],
    *,
    target: float = TARGET_REVIEW_BUDGET,
) -> dict[str, Any]:
    values = np.asarray(list(scores), dtype="int64")
    n = int(values.size)
    ranked: list[tuple[float, int, int, float]] = []
    for threshold in range(MIN_TOTAL, MAX_TOTAL + 1):
        n_review = int((values >= threshold).sum())
        rate = n_review / n if n else float("nan")
        ranked.append((abs(rate - float(target)), -threshold, n_review, rate))
    interior = [row for row in ranked if 0.0 < row[3] < 1.0]
    pool = interior if interior else ranked
    pool.sort()
    distance, neg_t, n_review, rate = pool[0]
    threshold = -neg_t
    n_tie = int((values == threshold).sum())
    return {
        "target_budget": float(target),
        "threshold": int(threshold),
        "decision_rule": "REVIEW if total_score >= threshold else CLEAR",
        "n_legitimate": n,
        "n_review": int(n_review),
        "empirical_review_rate": float(rate),
        "abs_distance_from_target": float(distance),
        "tie_size_at_threshold": n_tie,
        "status": "provisional_sanity_only",
        "final_month6_operating_point": False,
        "random_split_of_ties": False,
    }


def sentinel_stability(
    first: Mapping[str, Mapping[str, str]],
    second: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    material_failures = 0
    for app_id in sorted(set(first) & set(second)):
        a = first[app_id]
        b = second[app_id]
        changed = [dim_id for dim_id in DIMENSION_IDS if a[dim_id] != b[dim_id]]
        total_a = total_score(a)
        total_b = total_score(b)
        identical = not changed
        material = (total_a != total_b) or (len(changed) > 1)
        if material:
            material_failures += 1
        rows.append(
            {
                "id": app_id,
                "identical": identical,
                "materially_different": material,
                "changed_dimensions": changed,
                "total_first": total_a,
                "total_second": total_b,
            }
        )
    return {
        "n_sentinels": len(rows),
        "n_identical": sum(1 for row in rows if row["identical"]),
        "n_materially_different": material_failures,
        "acceptable": material_failures == 0,
        "rows": rows,
    }


def sanity_decision(
    *,
    legit_scores: Sequence[int],
    attack_scores: Sequence[int],
    threshold_info: Mapping[str, Any],
    sentinel: Mapping[str, Any],
) -> dict[str, Any]:
    summary = score_summary(legit_scores)
    rate = float(threshold_info["empirical_review_rate"])
    collapsed = (
        int(summary["n_unique"]) <= COLLAPSE_MAX_UNIQUE
        or float(summary["modal_share"]) >= COLLAPSE_MODAL_SHARE
        or rate <= 0.0
        or rate >= 1.0
        or not bool(sentinel.get("acceptable"))
    )
    if collapsed:
        conclusion = "FAIL_SCORE_COLLAPSE"
    else:
        legit_median = float(np.median(np.asarray(list(legit_scores), dtype="int64")))
        attack_median = float(np.median(np.asarray(list(attack_scores), dtype="int64")))
        attack_rate = float(
            np.mean(
                np.asarray(list(attack_scores), dtype="int64")
                >= int(threshold_info["threshold"])
            )
        )
        shifted = (attack_median > legit_median) or (
            (attack_rate - rate) >= DISCRIMINATION_REVIEW_LIFT
        )
        conclusion = (
            "PASS_TO_LARGER_MONTH6_CALIBRATION"
            if shifted
            else "FAIL_NO_PRELIMINARY_DISCRIMINATION"
        )
    return {
        "conclusion": conclusion,
        "legitimate_summary": summary,
        "attack_summary": score_summary(attack_scores),
        "provisional_review_rate": rate,
        "sentinel_acceptable": bool(sentinel.get("acceptable")),
        "gates": {
            "collapse_max_unique": COLLAPSE_MAX_UNIQUE,
            "collapse_modal_share": COLLAPSE_MODAL_SHARE,
            "discrimination_review_lift": DISCRIMINATION_REVIEW_LIFT,
        },
    }
