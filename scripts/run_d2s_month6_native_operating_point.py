#!/usr/bin/env python3
"""Native Month-6 D2-S operating curve and Youden-J threshold.

Uses only native Month-6 applications and the frozen D1 / D2-S artefacts.
Does not load attacker outcomes, does not refit D2-S, and does not open Month 7.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parents[1]
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.paths import SCRATCH_CALIBRATION_ROOT, new_run_directory  # noqa: E402
from d2.calibrate import (  # noqa: E402
    REVIEW_BUDGETS,
    load_month6_d1_scores,
    load_month6_d1_threshold,
    thresholds_for_budgets,
)
from d2.contract import CALIBRATION_MONTHS, SCORE_CONTRACT_ID, SEALED_MONTHS  # noqa: E402
from d2.data import DEFAULT_RAW_PATH, load_month6_applications  # noqa: E402
from d2.errors import D2DataError  # noqa: E402
from d2.scoring import D2SScorer  # noqa: E402

TIE_BREAK_RULE = (
    "Among thresholds that attain the maximum Youden J, select the highest "
    "threshold. This is the lowest legitimate review burden (lowest FPR) "
    "among the tied max-J operating points."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-parent", type=Path, default=SCRATCH_CALIBRATION_ROOT)
    return parser.parse_args()


def _assert_month6_only(frame: pd.DataFrame) -> None:
    months = set(int(m) for m in frame["month"].unique())
    if months - set(CALIBRATION_MONTHS):
        raise D2DataError(f"Non-Month-6 rows present: {sorted(months)}")
    if frame["month"].isin(list(SEALED_MONTHS)).any():
        raise D2DataError("Sealed-month rows present; aborting.")


def d1_confusion(scores: pd.DataFrame, threshold: float) -> dict[str, int]:
    y_true = scores["y_true"].astype("int64")
    blocked = scores["y_score"].astype("float64") >= float(threshold)
    tp = int(((y_true == 1) & blocked).sum())
    fp = int(((y_true == 0) & blocked).sum())
    tn = int(((y_true == 0) & ~blocked).sum())
    fn = int(((y_true == 1) & ~blocked).sum())
    return {
        "n_month6_scored": int(len(scores)),
        "d1_threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "positive_prediction": "D1 BLOCK (y_score >= threshold)",
        "ground_truth_positive": "fraud_bool == 1",
    }


def metrics_at_threshold(
    legit_scores: np.ndarray,
    fraud_scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    n_legit = int(legit_scores.size)
    n_fraud = int(fraud_scores.size)
    legit_review = int((legit_scores >= threshold).sum())
    fraud_review = int((fraud_scores >= threshold).sum())
    legit_clear = n_legit - legit_review
    fraud_clear = n_fraud - fraud_review
    tpr = fraud_review / n_fraud if n_fraud else float("nan")
    fpr = legit_review / n_legit if n_legit else float("nan")
    return {
        "threshold": float(threshold),
        "n_legitimate_d1_pass": n_legit,
        "n_fraud_d1_pass": n_fraud,
        "legitimate_review_count": legit_review,
        "legitimate_review_rate": fpr,
        "legitimate_auto_clear_count": legit_clear,
        "legitimate_auto_clear_rate": 1.0 - fpr if n_legit else float("nan"),
        "fraud_review_count": fraud_review,
        "fraud_recovery_rate": tpr,
        "residual_fraud_clear_count": fraud_clear,
        "residual_fraud_clear_rate": 1.0 - tpr if n_fraud else float("nan"),
        "specificity": 1.0 - fpr if n_legit else float("nan"),
        "youden_j": tpr - fpr if n_legit and n_fraud else float("nan"),
        "youden_j_integer_score": int(fraud_review * n_legit - legit_review * n_fraud),
    }


def youden_curve(legit_scores: np.ndarray, fraud_scores: np.ndarray) -> pd.DataFrame:
    """Evaluate every distinct observed D2-S score as a REVIEW threshold."""
    candidates = np.unique(np.concatenate([legit_scores, fraud_scores]))
    candidates = np.sort(candidates)[::-1]
    n_legit = int(legit_scores.size)
    n_fraud = int(fraud_scores.size)
    legit_sorted = np.sort(legit_scores)
    fraud_sorted = np.sort(fraud_scores)
    rows: list[dict[str, float | int]] = []
    for threshold in candidates:
        legit_review = n_legit - int(np.searchsorted(legit_sorted, threshold, side="left"))
        fraud_review = n_fraud - int(np.searchsorted(fraud_sorted, threshold, side="left"))
        tpr = fraud_review / n_fraud
        fpr = legit_review / n_legit
        rows.append(
            {
                "threshold": float(threshold),
                "n_legitimate_d1_pass": n_legit,
                "n_fraud_d1_pass": n_fraud,
                "legitimate_review_count": legit_review,
                "legitimate_review_rate": fpr,
                "legitimate_auto_clear_count": n_legit - legit_review,
                "legitimate_auto_clear_rate": 1.0 - fpr,
                "fraud_review_count": fraud_review,
                "fraud_recovery_rate": tpr,
                "residual_fraud_clear_count": n_fraud - fraud_review,
                "residual_fraud_clear_rate": 1.0 - tpr,
                "specificity": 1.0 - fpr,
                "youden_j": tpr - fpr,
                "youden_j_integer_score": int(fraud_review * n_legit - legit_review * n_fraud),
            }
        )
    return pd.DataFrame(rows)


def select_max_youden(curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    max_score = int(curve["youden_j_integer_score"].max())
    tied = curve.loc[curve["youden_j_integer_score"] == max_score].copy()
    tied = tied.sort_values(["threshold", "legitimate_review_rate"], ascending=[False, True])
    selected = tied.iloc[0]
    return tied, selected


def plot_roc(curve: pd.DataFrame, selected: pd.Series, checkpoints: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ordered = curve.sort_values("legitimate_review_rate")
    ax.plot(
        ordered["legitimate_review_rate"],
        ordered["fraud_recovery_rate"],
        color="#1f4e79",
        linewidth=1.6,
        label="Native Month-6 D1-PASS",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#888888", linestyle="--", linewidth=1.0, label="Chance")
    ax.scatter(
        [float(selected["legitimate_review_rate"])],
        [float(selected["fraud_recovery_rate"])],
        color="#c0392b",
        s=42,
        zorder=5,
        label="Max Youden J",
    )
    highlight = checkpoints.loc[checkpoints["budget"].isin([0.05, 0.10, 0.15])]
    ax.scatter(
        highlight["legitimate_review_rate"],
        highlight["fraud_recovery_rate"],
        color="#d4a017",
        s=36,
        zorder=4,
        label="5% / 10% / 15% review budgets",
    )
    ax.set_xlabel("Legitimate review rate (FPR)")
    ax.set_ylabel("Fraud recovery rate (TPR)")
    ax.set_title("Native Month-6 D2-S operating curve (D1-PASS only)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_balance(curve: pd.DataFrame, selected: pd.Series, checkpoints: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ordered = curve.sort_values("legitimate_auto_clear_rate")
    ax.plot(
        ordered["legitimate_auto_clear_rate"],
        ordered["fraud_recovery_rate"],
        color="#1f4e79",
        linewidth=1.6,
        label="Native Month-6 D1-PASS",
    )
    ax.scatter(
        [float(selected["legitimate_auto_clear_rate"])],
        [float(selected["fraud_recovery_rate"])],
        color="#c0392b",
        s=42,
        zorder=5,
        label="Max Youden J",
    )
    highlight = checkpoints.loc[checkpoints["budget"].isin([0.05, 0.10, 0.15])]
    ax.scatter(
        highlight["legitimate_auto_clear_rate"],
        highlight["fraud_recovery_rate"],
        color="#d4a017",
        s=36,
        zorder=4,
        label="5% / 10% / 15% review budgets",
    )
    ax.set_xlabel("Legitimate auto-clear rate")
    ax.set_ylabel("Fraud recovery rate")
    ax.set_title("Native Month-6 D2-S operational balance (D1-PASS only)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    scorer = D2SScorer.load(args.reference)
    if scorer.month7_opened:
        raise SystemExit("Refusing a scorer that opened Month 7.")
    if "month7" in str(args.reference).lower() or "month_7" in str(args.reference).lower():
        raise SystemExit("Refusing a Month-7 reference path.")

    d1_scores = load_month6_d1_scores()
    d1_threshold = load_month6_d1_threshold()
    confusion = d1_confusion(d1_scores, d1_threshold)

    loaded = load_month6_applications(args.raw, fraud_bool=None, verify_hash=True)
    if loaded.month7_opened:
        raise D2DataError("Month-6 load reported month7_opened=True.")
    _assert_month6_only(loaded.frame)

    merged = loaded.frame.merge(
        d1_scores,
        left_on="source_row_id",
        right_on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if int((merged["y_true"] != merged["fraud_bool"]).sum()):
        raise D2DataError("D1 score join disagreed with fraud_bool.")
    if int(len(merged)) != int(len(d1_scores)):
        raise D2DataError(
            f"Month-6 join size {len(merged)} != D1 score rows {len(d1_scores)}."
        )

    passed = merged.loc[merged["y_score"] < d1_threshold].copy()
    legit = passed.loc[passed["fraud_bool"] == 0]
    fraud = passed.loc[passed["fraud_bool"] == 1]
    if int(len(legit)) != confusion["TN"]:
        raise D2DataError("D1-PASS legitimate count != D1 TN.")
    if int(len(fraud)) != confusion["FN"]:
        raise D2DataError("D1-PASS fraud count != D1 FN.")

    legit_scored = scorer.score_many(legit)["d2_score"].to_numpy(dtype="float64")
    fraud_scored = scorer.score_many(fraud)["d2_score"].to_numpy(dtype="float64")

    curve = youden_curve(legit_scored, fraud_scored)
    tied, selected = select_max_youden(curve)

    budget_defs = thresholds_for_budgets(legit_scored, REVIEW_BUDGETS)
    checkpoint_rows = []
    for row in budget_defs.itertuples(index=False):
        metrics = metrics_at_threshold(legit_scored, fraud_scored, float(row.threshold))
        metrics["budget"] = float(row.budget)
        checkpoint_rows.append(metrics)
    checkpoints = pd.DataFrame(checkpoint_rows)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"d2s_month6_native_operating_point_{SCORE_CONTRACT_ID}_{stamp}",
        parent=args.output_parent,
        stage="scratch",
    )

    curve.to_csv(run_dir / "native_youden_curve.csv", index=False)
    tied.to_csv(run_dir / "max_youden_tied_thresholds.csv", index=False)
    pd.DataFrame([selected.to_dict()]).to_csv(run_dir / "max_youden_operating_point.csv", index=False)
    checkpoints.to_csv(run_dir / "review_budget_checkpoints.csv", index=False)
    pd.DataFrame([
        {"source_row_id": int(i), "fraud_bool": 0, "d2_score": float(s)}
        for i, s in zip(legit["source_row_id"].to_numpy(), legit_scored)
    ]).to_csv(run_dir / "month6_d1_pass_legit_d2_scores.csv", index=False)
    pd.DataFrame([
        {"source_row_id": int(i), "fraud_bool": 1, "d2_score": float(s)}
        for i, s in zip(fraud["source_row_id"].to_numpy(), fraud_scored)
    ]).to_csv(run_dir / "month6_d1_pass_fraud_d2_scores.csv", index=False)

    plot_roc(curve, selected, checkpoints, run_dir / "figure1_native_month6_roc.png")
    plot_balance(curve, selected, checkpoints, run_dir / "figure2_native_month6_operational_balance.png")

    compare_budgets = checkpoints.loc[checkpoints["budget"].isin([0.05, 0.10, 0.15])].copy()
    summary = {
        "score_contract_id": SCORE_CONTRACT_ID,
        "scorer_fingerprint": scorer.fingerprint,
        "reference_path": str(args.reference),
        "month7_opened": False,
        "population": "Month == 6 native applications; D1 PASS only",
        "threshold_selection": "native Month-6 D1-PASS labels only; attacker outcomes not used",
        "d1_confusion_month6": confusion,
        "d1_pass_counts": {
            "legitimate_fraud_bool_0": int(len(legit)),
            "missed_fraud_fraud_bool_1": int(len(fraud)),
        },
        "n_unique_thresholds_evaluated": int(len(curve)),
        "tie_break_rule": TIE_BREAK_RULE,
        "n_tied_max_youden_thresholds": int(len(tied)),
        "max_youden_operating_point": selected.to_dict(),
        "tied_max_youden_thresholds": tied.to_dict(orient="records"),
        "review_budget_checkpoints": checkpoints.to_dict(orient="records"),
        "comparison_vs_5_10_15": compare_budgets.to_dict(orient="records"),
        "interpretation": (
            "Max Youden J is the neutral equal-weight statistical operating "
            "point (sensitivity and specificity weighted equally). It is not "
            "claimed to be a real-world bank-optimal threshold. Real "
            "deployment costs are unknown."
        ),
    }
    (run_dir / "NATIVE_OPERATING_POINT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "max_youden": selected.to_dict()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
