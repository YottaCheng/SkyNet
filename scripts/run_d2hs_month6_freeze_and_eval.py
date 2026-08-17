#!/usr/bin/env python3
"""Build, freeze, and evaluate D2-HS on Month-6 native labels then attacks.

Does not modify D1 or D2-S, does not open Month 7, does not call an LLM,
and does not use A0–A3 outcomes to select alpha or thresholds.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.benchmark_pins import PINNED_D1_ARTEFACT_ID  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, OUTPUTS_ROOT  # noqa: E402
from d2.calibrate import load_month6_d1_scores, load_month6_d1_threshold  # noqa: E402
from d2.contract import REQUIRED_APPLICATION_FIELDS, SCORE_CONTRACT_ID  # noqa: E402
from d2.scoring import D2SScorer  # noqa: E402

D1_THRESHOLD = 0.04724566638469696
EXPECTED_LEGIT = 101422
EXPECTED_FRAUD = 705
EXPECTED_ATTACKS = {"A0": 26, "A1-Pro": 27, "A2": 37, "A3-Pro": 31}
# Frozen D2-S Pro-ThinkOff REVIEW counts among the same 121 D1-PASS cases.
# Used only to confirm D2-S scoring identity; never to select alpha.
EXPECTED_D2S_REVIEW = {
    0.05: {"A0": 3, "A1-Pro": 0, "A2": 3, "A3-Pro": 2},
    0.10: {"A0": 4, "A1-Pro": 0, "A2": 9, "A3-Pro": 2},
    0.15: {"A0": 5, "A1-Pro": 3, "A2": 12, "A3-Pro": 3},
}
EXPECTED_D2S_THRESHOLDS = {
    0.05: 0.6298681955497826,
    0.10: 0.5918014572249943,
    0.15: 0.5720686786860827,
}
N_ANCHORS = 50
ALPHAS = tuple(i / 10 for i in range(11))
BUDGETS = (0.05, 0.10, 0.15)
PRIMARY_BUDGET = 0.10
FROZEN_D2S_FINGERPRINT = (
    "cfd5330f096dabb1749be447ee4da4d5f498d2599f4f22c24a0b706e570bfd94"
)

D2S_REFERENCE = (
    OUTPUTS_ROOT
    / "development"
    / "d2s"
    / "reference_fit"
    / "d2s_reference_d2s-v1.0.0-pairwise8-20260816_20260816T005649Z"
    / "d2s_reference.json"
)
NATIVE_D2S_DIR = (
    OUTPUTS_ROOT
    / "development"
    / "d2s"
    / "native_operating_point"
    / "d2s_month6_native_operating_point_d2s-v1.0.0-pairwise8-20260816_20260816T013859Z"
)
ATTACK_DATASET = (
    OUTPUTS_ROOT
    / "development"
    / "d2l"
    / "input_dataset"
    / "month6_successful_d1_bypasses_pro_thinkoff_20260816T194310Z"
    / "successful_d1_bypasses.jsonl"
)
OUT_PARENT = OUTPUTS_ROOT / "development" / "d2hs"


class D2HSError(RuntimeError):
    """Fail-closed D2-HS builder error."""


def _assert_not_month7(path: Path) -> None:
    text = str(path).lower()
    if "month7" in text or "month_7" in text:
        raise D2HSError(f"Refusing a Month-7 path: {path}")


def residual_risk(scores: np.ndarray, reference_sorted: np.ndarray) -> np.ndarray:
    """P(legit D1-PASS S <= s). Higher D1 score → higher residual risk."""
    n = float(reference_sorted.size)
    return np.searchsorted(reference_sorted, scores, side="right") / n


def budget_threshold(scores: np.ndarray, budget: float) -> tuple[float, int, float]:
    threshold = float(np.quantile(scores, 1.0 - float(budget)))
    n_review = int((scores >= threshold).sum())
    rate = n_review / float(scores.size)
    return threshold, n_review, rate


def hybrid(alpha: float, d1_rr: np.ndarray, d2s: np.ndarray) -> np.ndarray:
    return alpha * d1_rr + (1.0 - alpha) * d2s


def load_native_populations() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    d1_scores = load_month6_d1_scores()
    threshold = load_month6_d1_threshold()
    if abs(threshold - D1_THRESHOLD) > 1e-15:
        raise D2HSError(f"D1 threshold {threshold} != {D1_THRESHOLD}")
    passed = d1_scores.loc[d1_scores["y_score"] < threshold].copy()
    legit = passed.loc[passed["y_true"] == 0].copy()
    fraud = passed.loc[passed["y_true"] == 1].copy()
    if len(legit) != EXPECTED_LEGIT:
        raise D2HSError(f"Legit D1-PASS {len(legit)} != {EXPECTED_LEGIT}")
    if len(fraud) != EXPECTED_FRAUD:
        raise D2HSError(f"Fraud D1-PASS {len(fraud)} != {EXPECTED_FRAUD}")

    d2_legit = pd.read_csv(NATIVE_D2S_DIR / "month6_d1_pass_legit_d2_scores.csv")
    d2_fraud = pd.read_csv(NATIVE_D2S_DIR / "month6_d1_pass_fraud_d2_scores.csv")
    if len(d2_legit) != EXPECTED_LEGIT or int((d2_legit["fraud_bool"] != 0).sum()):
        raise D2HSError("Native legit D2-S table does not match the D1-PASS population.")
    if len(d2_fraud) != EXPECTED_FRAUD or int((d2_fraud["fraud_bool"] != 1).sum()):
        raise D2HSError("Native fraud D2-S table does not match the D1-PASS population.")

    legit = legit.merge(
        d2_legit,
        left_on="row_id",
        right_on="source_row_id",
        how="inner",
        validate="one_to_one",
    )
    fraud = fraud.merge(
        d2_fraud,
        left_on="row_id",
        right_on="source_row_id",
        how="inner",
        validate="one_to_one",
    )
    if len(legit) != EXPECTED_LEGIT or len(fraud) != EXPECTED_FRAUD:
        raise D2HSError("D1/D2-S join dropped rows.")
    return legit, fraud, float(threshold)


def load_attacks(scorer: D2SScorer, reference_sorted: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    with ATTACK_DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            meta = record["attack_metadata"]
            defence = record["defender_metadata"]
            app = record["application"]
            if defence["d1_decision"] != "PASS":
                raise D2HSError(f"{record['record_id']} is not D1 PASS.")
            d1_score = float(defence["d1_score"])
            if d1_score >= D1_THRESHOLD:
                raise D2HSError(f"{record['record_id']} D1 score is not a PASS.")
            rows.append(
                {
                    "record_id": record["record_id"],
                    "condition_id": meta["condition_id"],
                    "anchor_id": str(meta["anchor_id"]),
                    "d1_score": d1_score,
                }
            )
            applications.append(
                {
                    name: (np.nan if app.get(name) is None else app.get(name))
                    for name in REQUIRED_APPLICATION_FIELDS
                }
            )
    frame = pd.DataFrame(rows)
    counts = frame["condition_id"].value_counts().to_dict()
    if {k: int(counts.get(k, 0)) for k in EXPECTED_ATTACKS} != EXPECTED_ATTACKS:
        raise D2HSError(f"Attack counts {counts} != {EXPECTED_ATTACKS}")
    if len(frame) != 121:
        raise D2HSError(f"Attack n={len(frame)} != 121")
    app_frame = pd.DataFrame(applications)
    d2 = scorer.score_many(app_frame)["d2_score"].to_numpy(dtype="float64")
    frame["d2s_score"] = d2
    frame["d1_residual_risk"] = residual_risk(
        frame["d1_score"].to_numpy(dtype="float64"), reference_sorted
    )
    return frame


def plot_alpha_curve(sweep: pd.DataFrame, selected_alpha: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(
        sweep["alpha"],
        sweep["native_fraud_recovery_rate"],
        color="#1f4e79",
        marker="o",
        linewidth=1.8,
        label="Native missed-fraud recovery @ 10% budget",
    )
    sel = sweep.loc[np.isclose(sweep["alpha"], selected_alpha)].iloc[0]
    ax.scatter(
        [selected_alpha],
        [sel["native_fraud_recovery_rate"]],
        color="#c0392b",
        s=48,
        zorder=5,
        label=f"Selected α={selected_alpha:.1f}",
    )
    ax.axhline(
        float(sweep.loc[np.isclose(sweep["alpha"], 0.0), "native_fraud_recovery_rate"].iloc[0]),
        color="#5b8aa9",
        linestyle="--",
        linewidth=1.0,
        label="α=0 pure D2-S",
    )
    ax.axhline(
        float(sweep.loc[np.isclose(sweep["alpha"], 1.0), "native_fraud_recovery_rate"].iloc[0]),
        color="#8c6d31",
        linestyle=":",
        linewidth=1.0,
        label="α=1 pure D1 residual risk",
    )
    ax.set_xlabel("α (weight on D1 residual risk)")
    ax.set_ylabel("Native missed-fraud recovery rate")
    ax.set_title("D2-HS alpha search at the 10% experimental review budget")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, max(0.22, float(sweep["native_fraud_recovery_rate"].max()) + 0.04))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_attack_interception(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    labels = list(table["attacker"])
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(
        x - w / 2,
        table["d2s_interception_rate"],
        width=w,
        color="#5b8aa9",
        label="D2-S",
    )
    ax.bar(
        x + w / 2,
        table["d2hs_interception_rate"],
        width=w,
        color="#c0392b",
        label="D2-HS",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Interception rate (REVIEW / D1-PASS attacks)")
    ax.set_title("Attack interception at the 10% experimental review budget")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    _assert_not_month7(DEFAULT_C1_ARTEFACT_DIR)
    _assert_not_month7(D2S_REFERENCE)
    _assert_not_month7(ATTACK_DATASET)
    if "month7" in str(NATIVE_D2S_DIR).lower():
        raise D2HSError("Native D2-S path looks like Month 7.")

    scorer = D2SScorer.load(D2S_REFERENCE)
    if scorer.month7_opened:
        raise D2HSError("Refusing a D2-S scorer that opened Month 7.")
    if scorer.fingerprint != FROZEN_D2S_FINGERPRINT:
        raise D2HSError(f"Unexpected D2-S fingerprint {scorer.fingerprint}")
    if PINNED_D1_ARTEFACT_ID != "c1_pipeline_sha256_16=243c851b0c665c9c":
        raise D2HSError(f"Unexpected D1 artefact id {PINNED_D1_ARTEFACT_ID}")

    legit, fraud, d1_threshold = load_native_populations()
    legit_d1 = legit["y_score"].to_numpy(dtype="float64")
    legit_d2 = legit["d2_score"].to_numpy(dtype="float64")
    fraud_d1 = fraud["y_score"].to_numpy(dtype="float64")
    fraud_d2 = fraud["d2_score"].to_numpy(dtype="float64")
    reference_sorted = np.sort(legit_d1)
    legit_rr = residual_risk(legit_d1, reference_sorted)
    fraud_rr = residual_risk(fraud_d1, reference_sorted)

    sweep_rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        legit_h = hybrid(alpha, legit_rr, legit_d2)
        fraud_h = hybrid(alpha, fraud_rr, fraud_d2)
        row: dict[str, Any] = {"alpha": float(alpha)}
        for budget in BUDGETS:
            thr, n_rev, rate = budget_threshold(legit_h, budget)
            fraud_review = int((fraud_h >= thr).sum())
            recovery = fraud_review / float(EXPECTED_FRAUD)
            prefix = f"b{int(budget * 100):02d}"
            row[f"{prefix}_threshold"] = thr
            row[f"{prefix}_legit_review_count"] = n_rev
            row[f"{prefix}_legit_review_rate"] = rate
            row[f"{prefix}_fraud_review_count"] = fraud_review
            row[f"{prefix}_native_fraud_recovery_rate"] = recovery
        primary_thr, _, _ = budget_threshold(legit_h, PRIMARY_BUDGET)
        primary_recovery_count = int((fraud_h >= primary_thr).sum())
        row["native_fraud_recovery_count"] = primary_recovery_count
        row["native_fraud_recovery_rate"] = primary_recovery_count / float(EXPECTED_FRAUD)
        sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows)

    max_count = int(sweep["native_fraud_recovery_count"].max())
    tied = sweep.loc[sweep["native_fraud_recovery_count"] == max_count].sort_values("alpha")
    selected = tied.iloc[0]
    selected_alpha = float(selected["alpha"])

    rec0 = float(sweep.loc[np.isclose(sweep["alpha"], 0.0), "native_fraud_recovery_rate"].iloc[0])
    rec1 = float(sweep.loc[np.isclose(sweep["alpha"], 1.0), "native_fraud_recovery_rate"].iloc[0])
    rec_sel = float(selected["native_fraud_recovery_rate"])
    synergy = (
        selected_alpha not in (0.0, 1.0)
        and rec_sel > rec0
        and rec_sel > rec1
    )

    # Freeze selected hybrid thresholds on legit only.
    legit_h_sel = hybrid(selected_alpha, legit_rr, legit_d2)
    frozen_thresholds: dict[str, dict[str, float | int]] = {}
    for budget in BUDGETS:
        thr, n_rev, rate = budget_threshold(legit_h_sel, budget)
        frozen_thresholds[f"{int(budget * 100)}pct"] = {
            "budget": budget,
            "threshold": thr,
            "n_legitimate_review": n_rev,
            "empirical_legit_review_rate": rate,
        }

    # D2-S-only thresholds (alpha=0) for comparison, still legit-only.
    d2s_thresholds: dict[str, dict[str, float | int]] = {}
    for budget in BUDGETS:
        thr, n_rev, rate = budget_threshold(legit_d2, budget)
        expected_thr = EXPECTED_D2S_THRESHOLDS[budget]
        if abs(thr - expected_thr) > 1e-12:
            raise D2HSError(
                f"D2-S {int(budget*100)}% threshold {thr} != frozen {expected_thr}"
            )
        d2s_thresholds[f"{int(budget * 100)}pct"] = {
            "budget": budget,
            "threshold": thr,
            "n_legitimate_review": n_rev,
            "empirical_legit_review_rate": rate,
        }

    native_rows: list[dict[str, Any]] = []
    fraud_h_sel = hybrid(selected_alpha, fraud_rr, fraud_d2)
    for budget in BUDGETS:
        key = f"{int(budget * 100)}pct"
        d2s_thr = float(d2s_thresholds[key]["threshold"])
        hs_thr = float(frozen_thresholds[key]["threshold"])
        d2s_fraud_review = int((fraud_d2 >= d2s_thr).sum())
        if budget == 0.05 and d2s_fraud_review != 29:
            raise D2HSError(f"D2-S native fraud REVIEW @5% is {d2s_fraud_review}, expected 29")
        if budget == 0.10 and d2s_fraud_review != 58:
            raise D2HSError(f"D2-S native fraud REVIEW @10% is {d2s_fraud_review}, expected 58")
        if budget == 0.15 and d2s_fraud_review != 85:
            raise D2HSError(f"D2-S native fraud REVIEW @15% is {d2s_fraud_review}, expected 85")
        native_rows.append(
            {
                "budget": budget,
                "label": (
                    "PRIMARY EXPERIMENTAL REVIEW BUDGET"
                    if budget == PRIMARY_BUDGET
                    else ("lower-friction sensitivity" if budget == 0.05 else "higher-review sensitivity")
                ),
                "d2s_threshold": d2s_thr,
                "d2hs_alpha": selected_alpha,
                "d2hs_threshold": hs_thr,
                "legit_n": EXPECTED_LEGIT,
                "fraud_n": EXPECTED_FRAUD,
                "d2s_legit_review_rate": float(d2s_thresholds[key]["empirical_legit_review_rate"]),
                "d2hs_legit_review_rate": float(frozen_thresholds[key]["empirical_legit_review_rate"]),
                "d2s_fraud_review_count": d2s_fraud_review,
                "d2s_native_fraud_recovery_rate": d2s_fraud_review / EXPECTED_FRAUD,
                "d2hs_fraud_review_count": int((fraud_h_sel >= hs_thr).sum()),
                "d2hs_native_fraud_recovery_rate": int((fraud_h_sel >= hs_thr).sum()) / EXPECTED_FRAUD,
            }
        )
    native_table = pd.DataFrame(native_rows)

    # Attacks only after freeze.
    attacks = load_attacks(scorer, reference_sorted)
    attacks["d2hs_score"] = hybrid(
        selected_alpha,
        attacks["d1_residual_risk"].to_numpy(dtype="float64"),
        attacks["d2s_score"].to_numpy(dtype="float64"),
    )

    attack_rows: list[dict[str, Any]] = []
    primary10_rows: list[dict[str, Any]] = []
    conditions = list(EXPECTED_ATTACKS)
    for budget in BUDGETS:
        key = f"{int(budget * 100)}pct"
        d2s_thr = float(d2s_thresholds[key]["threshold"])
        hs_thr = float(frozen_thresholds[key]["threshold"])
        for condition in conditions + ["POOLED"]:
            sub = attacks if condition == "POOLED" else attacks.loc[attacks["condition_id"] == condition]
            n_d1 = int(len(sub) if condition != "POOLED" else 121)
            denom_e2e = N_ANCHORS if condition != "POOLED" else N_ANCHORS * 4
            d2s_review = int((sub["d2s_score"] >= d2s_thr).sum())
            hs_review = int((sub["d2hs_score"] >= hs_thr).sum())
            d2s_clear = n_d1 - d2s_review
            hs_clear = n_d1 - hs_review
            d1_asr = n_d1 / float(denom_e2e)
            row = {
                "budget": budget,
                "attacker": condition,
                "n_d1_success": n_d1,
                "n_anchors": denom_e2e,
                "d1_asr": d1_asr,
                "d2s_threshold": d2s_thr,
                "d2hs_threshold": hs_thr,
                "d2s_review_count": d2s_review,
                "d2s_interception_rate": d2s_review / n_d1,
                "d2s_clear_count": d2s_clear,
                "d2s_e2e_bypass_rate": d2s_clear / float(denom_e2e),
                "d2hs_review_count": hs_review,
                "d2hs_interception_rate": hs_review / n_d1,
                "d2hs_clear_count": hs_clear,
                "d2hs_e2e_bypass_rate": hs_clear / float(denom_e2e),
            }
            if condition != "POOLED":
                expected_review = EXPECTED_D2S_REVIEW[budget][condition]
                if d2s_review != expected_review:
                    raise D2HSError(
                        f"D2-S REVIEW {condition} @{int(budget*100)}% "
                        f"is {d2s_review}, expected {expected_review} "
                        "(scoring drifted from the frozen Pro-ThinkOff eval)."
                    )
            attack_rows.append(row)
            if budget == PRIMARY_BUDGET:
                primary10_rows.append(
                    {
                        "attacker": condition,
                        "d1_asr": d1_asr,
                        "d2s_interception": d2s_review / n_d1,
                        "d2hs_interception": hs_review / n_d1,
                        "d1_d2s_e2e_bypass": d2s_clear / float(denom_e2e),
                        "d1_d2hs_e2e_bypass": hs_clear / float(denom_e2e),
                        "n_d1_success": n_d1,
                        "d2s_review_count": d2s_review,
                        "d2hs_review_count": hs_review,
                        "d2s_clear_count": d2s_clear,
                        "d2hs_clear_count": hs_clear,
                        "d2s_interception_rate": d2s_review / n_d1,
                        "d2hs_interception_rate": hs_review / n_d1,
                    }
                )
    attack_table = pd.DataFrame(attack_rows)
    primary10 = pd.DataFrame(primary10_rows)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_PARENT / f"month6_hybrid_d1_d2s_{stamp}"
    if out_dir.exists():
        raise D2HSError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    scores_path = out_dir / "d1_residual_risk_legit_d1_pass_scores.npy"
    np.save(scores_path, reference_sorted)
    transform_sha = hashlib.sha256(reference_sorted.tobytes()).hexdigest()

    config = {
        "d2hs_id": f"d2hs-v1.0.0-alpha{selected_alpha:.1f}-{stamp}",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "month7_opened": False,
        "attack_outcomes_used_for_selection": False,
        "alpha": selected_alpha,
        "hybrid_formula": "hybrid_score = alpha * d1_residual_risk + (1-alpha) * d2s_score",
        "decision_rule": "REVIEW if hybrid_score >= threshold else CLEAR",
        "primary_experimental_review_budget": PRIMARY_BUDGET,
        "d1_artefact_id": PINNED_D1_ARTEFACT_ID,
        "d1_threshold": d1_threshold,
        "d2s_score_contract_id": SCORE_CONTRACT_ID,
        "d2s_fingerprint": scorer.fingerprint,
        "d1_residual_risk": {
            "definition": "P_{Month-6 legitimate D1-PASS}(S <= s_obs)",
            "fraud_labels_used": False,
            "n_reference": EXPECTED_LEGIT,
            "reference_scores_file": scores_path.name,
            "reference_scores_sha256": transform_sha,
        },
        "tie_break": "If native fraud recovery is tied at 10%, choose the smaller alpha.",
        "selected_native_fraud_recovery_at_10pct": rec_sel,
        "selected_native_fraud_review_count_at_10pct": int(selected["native_fraud_recovery_count"]),
        "synergy_vs_both_endpoints": synergy,
        "d2hs_thresholds": frozen_thresholds,
        "d2s_comparison_thresholds": d2s_thresholds,
        "n_legit_d1_pass": EXPECTED_LEGIT,
        "n_fraud_d1_pass": EXPECTED_FRAUD,
    }
    (out_dir / "selected_d2hs_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    sweep.to_csv(out_dir / "alpha_search.csv", index=False)
    native_table.to_csv(out_dir / "native_5_10_15_table.csv", index=False)
    attack_table.to_csv(out_dir / "attack_5_10_15_table.csv", index=False)
    primary10.to_csv(out_dir / "d2s_vs_d2hs_primary10.csv", index=False)

    plot_alpha_curve(sweep, selected_alpha, out_dir / "alpha_vs_native_fraud_recovery.png")
    plot_attack_interception(primary10, out_dir / "d2s_vs_d2hs_attack_interception.png")

    def fmt_row(r: pd.Series) -> str:
        return (
            f"| {r['attacker']} | {r['d1_asr']:.3f} | "
            f"{r['d2s_interception']:.3f} ({int(r['d2s_review_count'])}/{int(r['n_d1_success'])}) | "
            f"{r['d2hs_interception']:.3f} ({int(r['d2hs_review_count'])}/{int(r['n_d1_success'])}) | "
            f"{r['d1_d2s_e2e_bypass']:.3f} | {r['d1_d2hs_e2e_bypass']:.3f} |"
        )

    alpha_lines = [
        f"| {row.alpha:.1f} | {int(row.native_fraud_recovery_count)}/{EXPECTED_FRAUD} | "
        f"{row.native_fraud_recovery_rate:.4f} | {row.b10_legit_review_rate:.4f} | {row.b10_threshold:.6f} |"
        for row in sweep.itertuples(index=False)
    ]

    def fmt_attack(r: pd.Series) -> str:
        return (
            f"| {r['attacker']} | {int(r['n_d1_success'])} | "
            f"{int(r['d2s_review_count'])}/{int(r['n_d1_success'])} "
            f"({r['d2s_interception_rate']:.3f}) | "
            f"{int(r['d2hs_review_count'])}/{int(r['n_d1_success'])} "
            f"({r['d2hs_interception_rate']:.3f}) | "
            f"{int(r['d2hs_clear_count'])} | "
            f"{r['d2s_e2e_bypass_rate']:.3f} | {r['d2hs_e2e_bypass_rate']:.3f} |"
        )

    def attack_block(budget: float) -> str:
        sub = attack_table.loc[np.isclose(attack_table["budget"], budget)]
        lines = [fmt_attack(r) for _, r in sub.iterrows()]
        return "\n".join(lines)

    native_lines = [
        (
            f"| {int(r.budget * 100)}% | {r.label} | "
            f"{int(r.d2s_fraud_review_count)}/{EXPECTED_FRAUD} "
            f"({r.d2s_native_fraud_recovery_rate:.4f}) | "
            f"{int(r.d2hs_fraud_review_count)}/{EXPECTED_FRAUD} "
            f"({r.d2hs_native_fraud_recovery_rate:.4f}) | "
            f"{r.d2hs_threshold:.6f} |"
        )
        for r in native_table.itertuples(index=False)
    ]
    report = f"""# D2-HS Month-6 freeze and attack evaluation

Created: {config['frozen_utc']}
Output: `{out_dir}`

D1 and D2-S were not modified. Attack outcomes were not used to select α or thresholds.
Month 7 was not opened. No LLM/API call was made.

## Freeze

- Selected α = **{selected_alpha:.1f}**
- Selection: maximum native missed-fraud recovery at the **PRIMARY EXPERIMENTAL REVIEW BUDGET (10%)**
- Tie-break: smaller α
- Native recovery at 10%: **{int(selected['native_fraud_recovery_count'])}/{EXPECTED_FRAUD} = {rec_sel:.4f}**
- α=0 (pure D2-S) recovery @10%: {rec0:.4f}
- α=1 (pure D1 residual risk) recovery @10%: {rec1:.4f}
- Intermediate-α synergy vs both endpoints: **{synergy}**
- D1 fingerprint: `{PINNED_D1_ARTEFACT_ID}`
- D2-S fingerprint: `{scorer.fingerprint}`
- D2-S contract: `{SCORE_CONTRACT_ID}`

10% is the **PRIMARY EXPERIMENTAL REVIEW BUDGET** among legitimate D1-PASS applications, not a bank-optimal operating point.
5% = lower-friction sensitivity. 15% = higher-review sensitivity.

A synergy claim is allowed only if an intermediate α beats both α=0 and α=1 at the same 10% budget. That condition is **{synergy}**.

## Alpha sweep @ 10% legitimate-review budget

| α | Native fraud REVIEW | Recovery | Empirical legit review | Threshold |
|---:|---:|---:|---:|---:|
{chr(10).join(alpha_lines)}

## Native Month-6 missed-fraud recovery at 5/10/15%

Population: 101,422 legitimate D1-PASS; 705 native missed fraud. Thresholds from legitimate scores only.

| Budget | Label | D2-S recovery | D2-HS recovery | D2-HS threshold |
|---|---|---:|---:|---:|
{chr(10).join(native_lines)}

## Primary 10% attack comparison

Interception = REVIEW / successful D1 bypasses.
End-to-end bypass = (D1 PASS and layer CLEAR) / original anchors (50 per attacker; 200 pooled).

| Attacker | D1 ASR | D2-S interception | D2-HS interception | D1+D2-S E2E bypass | D1+D2-HS E2E bypass |
|---|---:|---:|---:|---:|---:|
{chr(10).join(fmt_row(r) for _, r in primary10.iterrows())}

## Attack evaluation at 5% (lower-friction sensitivity)

| Attacker | D1 success | D2-S interception | D2-HS interception | D2-HS CLEAR | D1+D2-S E2E | D1+D2-HS E2E |
|---|---:|---:|---:|---:|---:|---:|
{attack_block(0.05)}

## Attack evaluation at 10% (PRIMARY EXPERIMENTAL REVIEW BUDGET)

| Attacker | D1 success | D2-S interception | D2-HS interception | D2-HS CLEAR | D1+D2-S E2E | D1+D2-HS E2E |
|---|---:|---:|---:|---:|---:|---:|
{attack_block(0.10)}

## Attack evaluation at 15% (higher-review sensitivity)

| Attacker | D1 success | D2-S interception | D2-HS interception | D2-HS CLEAR | D1+D2-S E2E | D1+D2-HS E2E |
|---|---:|---:|---:|---:|---:|---:|
{attack_block(0.15)}
"""
    (out_dir / "BUILD_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "alpha": selected_alpha, "synergy": synergy}, indent=2))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
