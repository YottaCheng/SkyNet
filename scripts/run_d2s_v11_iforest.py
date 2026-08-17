#!/usr/bin/env python3
"""Fit and evaluate D2-S v1.1 Isolation Forest on frozen v1.0 scores.

Does not modify D2-S v1.0 or D1, does not rerun attackers, does not open
Month 7, and does not use fraud labels, D1 scores, or attack outcomes to
fit the Isolation Forest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

IMPL = Path(__file__).resolve().parents[1]
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.benchmark_pins import PINNED_D1_ARTEFACT_ID  # noqa: E402
from attack_lab.paths import OUTPUTS_ROOT  # noqa: E402
from d2.calibrate import load_month6_d1_scores, load_month6_d1_threshold  # noqa: E402
from d2.contract import (  # noqa: E402
    CALIBRATION_MONTHS,
    RELATIONSHIP_IDS,
    REQUIRED_APPLICATION_FIELDS,
    SCORE_CONTRACT_ID,
    SEALED_MONTHS,
)
from d2.data import DEFAULT_RAW_PATH, load_month6_applications, load_reference_legitimate  # noqa: E402
from d2.errors import D2DataError  # noqa: E402
from d2.iforest_v11 import (  # noqa: E402
    FIXED_IFOREST_PARAMS,
    FROZEN_D2S_V10_FINGERPRINT,
    IFOREST_FEATURE_IDS,
    SCORE_CONTRACT_ID_V11,
    collapse_to_iforest_features,
    fit_iforest_aggregator,
)
from d2.scoring import D2SScorer  # noqa: E402

EXPECTED_LEGIT = 101422
EXPECTED_FRAUD = 705
EXPECTED_REFERENCE_N = 786838
EXPECTED_ATTACKS = {"A0": 26, "A1-Pro": 27, "A2": 37, "A3-Pro": 31}
N_ANCHORS = 50
BUDGETS = (0.05, 0.10, 0.15)
PRIMARY_BUDGET = 0.10
D1_THRESHOLD = 0.04724566638469696
EXPECTED_V10_THRESHOLDS = {
    0.05: 0.6298681955497826,
    0.10: 0.5918014572249943,
    0.15: 0.5720686786860827,
}
EXPECTED_V10_REVIEW = {
    0.05: {"A0": 3, "A1-Pro": 0, "A2": 3, "A3-Pro": 2},
    0.10: {"A0": 4, "A1-Pro": 0, "A2": 9, "A3-Pro": 2},
    0.15: {"A0": 5, "A1-Pro": 3, "A2": 12, "A3-Pro": 3},
}
EXPECTED_V10_NATIVE_FRAUD_REVIEW = {0.05: 29, 0.10: 58, 0.15: 85}

D2S_V10_REFERENCE = (
    OUTPUTS_ROOT
    / "development"
    / "d2s"
    / "reference_fit"
    / "d2s_reference_d2s-v1.0.0-pairwise8-20260816_20260816T005649Z"
    / "d2s_reference.json"
)
ATTACK_DATASET = (
    OUTPUTS_ROOT
    / "development"
    / "d2l"
    / "input_dataset"
    / "month6_successful_d1_bypasses_pro_thinkoff_20260816T194310Z"
    / "successful_d1_bypasses.jsonl"
)
OUT_PARENT = OUTPUTS_ROOT / "development" / "d2s" / "v11_isolation_forest"


class D2V11Error(RuntimeError):
    """Fail-closed D2-S v1.1 builder error."""


def _assert_not_month7(path: Path) -> None:
    text = str(path).lower()
    if "month7" in text or "month_7" in text:
        raise D2V11Error(f"Refusing a Month-7 path: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def budget_threshold(scores: np.ndarray, budget: float) -> dict[str, float | int]:
    threshold = float(np.quantile(scores, 1.0 - float(budget)))
    n_review = int((scores >= threshold).sum())
    n_tie = int((scores == threshold).sum())
    return {
        "budget": float(budget),
        "threshold": threshold,
        "n_legitimate_review": n_review,
        "empirical_legit_review_rate": n_review / float(scores.size),
        "n_scores_equal_to_threshold": n_tie,
    }


def summarise(scores: np.ndarray) -> dict[str, float]:
    return {
        "n": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if scores.size > 1 else float("nan"),
        "min": float(np.min(scores)),
        "p50": float(np.quantile(scores, 0.50)),
        "p90": float(np.quantile(scores, 0.90)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
    }


def load_attacks(scorer: D2SScorer) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    with ATTACK_DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            meta = record["attack_metadata"]
            defence = record["defender_metadata"]
            app = record["application"]
            if defence["d1_decision"] != "PASS":
                raise D2V11Error(f"{record['record_id']} is not D1 PASS.")
            rows.append(
                {
                    "record_id": record["record_id"],
                    "condition_id": meta["condition_id"],
                    "anchor_id": str(meta["anchor_id"]),
                }
            )
            applications.append(
                {
                    name: (np.nan if app.get(name) is None else app.get(name))
                    for name in REQUIRED_APPLICATION_FIELDS
                }
            )
    frame = pd.DataFrame(rows)
    counts = {k: int((frame["condition_id"] == k).sum()) for k in EXPECTED_ATTACKS}
    if counts != EXPECTED_ATTACKS or len(frame) != 121:
        raise D2V11Error(f"Attack counts {counts} != {EXPECTED_ATTACKS}")
    rel = scorer.score_many(pd.DataFrame(applications))
    frame["d2s_v10_score"] = rel["d2_score"].to_numpy(dtype="float64")
    features = collapse_to_iforest_features(rel)
    for name in IFOREST_FEATURE_IDS:
        frame[name] = features[name].to_numpy(dtype="float64")
    return frame


def metrics_block(
    *,
    scores: np.ndarray,
    threshold: float,
    n_anchors: int,
) -> dict[str, Any]:
    review = scores >= threshold
    n_d1 = int(scores.size)
    n_review = int(review.sum())
    n_clear = n_d1 - n_review
    return {
        "n_d1_success": n_d1,
        "review_count": n_review,
        "interception_rate": n_review / n_d1 if n_d1 else float("nan"),
        "clear_count": n_clear,
        "e2e_bypass_rate": n_clear / float(n_anchors),
        "n_anchors": n_anchors,
    }


def recommend(primary_rows: pd.DataFrame) -> tuple[str, str]:
    v10 = primary_rows.loc[primary_rows["version"] == "d2s_v1.0"].set_index("attacker")
    v11 = primary_rows.loc[primary_rows["version"] == "d2s_v1.1"].set_index("attacker")
    attackers = ["A0", "A1-Pro", "A2", "A3-Pro"]
    improved = [
        name
        for name in attackers
        if float(v11.loc[name, "interception_rate"]) > float(v10.loc[name, "interception_rate"])
    ]
    pooled_gain = int(v11.loc["POOLED", "review_count"]) - int(v10.loc["POOLED", "review_count"])
    pooled_better = pooled_gain > 0
    meaningful = pooled_gain >= 5
    broad = len(improved) >= 2
    if pooled_better and meaningful and broad:
        choice = "FREEZE_CANDIDATE_D2S_V1_1_IFOREST"
        reason = (
            f"Pooled 10% interception rose by {pooled_gain} REVIEW cases and "
            f"{len(improved)}/4 attackers improved ({', '.join(improved)})."
        )
    else:
        choice = "KEEP_D2S_V1_0"
        reason = (
            "v1.1 did not show a meaningful and broad gain at the 10% "
            f"experimental budget (pooled REVIEW delta={pooled_gain}, "
            f"attackers improved={improved})."
        )
    return choice, reason


def main() -> int:
    _assert_not_month7(D2S_V10_REFERENCE)
    _assert_not_month7(ATTACK_DATASET)
    _assert_not_month7(DEFAULT_RAW_PATH)
    if SCORE_CONTRACT_ID != "d2s-v1.0.0-pairwise8-20260816":
        raise D2V11Error("D2-S v1.0 SCORE_CONTRACT_ID drifted.")
    if PINNED_D1_ARTEFACT_ID != "c1_pipeline_sha256_16=243c851b0c665c9c":
        raise D2V11Error(f"D1 artefact drifted: {PINNED_D1_ARTEFACT_ID}")

    v10_sha_before = file_sha256(D2S_V10_REFERENCE)
    scorer = D2SScorer.load(D2S_V10_REFERENCE)
    if scorer.month7_opened:
        raise D2V11Error("Frozen D2-S v1.0 reports Month 7 opened.")
    if scorer.fingerprint != FROZEN_D2S_V10_FINGERPRINT:
        raise D2V11Error(f"Unexpected D2-S v1.0 fingerprint {scorer.fingerprint}")
    if scorer.reference_n != EXPECTED_REFERENCE_N:
        raise D2V11Error(f"v1.0 reference_n {scorer.reference_n} != {EXPECTED_REFERENCE_N}")

    print("Loading Months 0–5 legitimate reference...", flush=True)
    loaded_ref = load_reference_legitimate(DEFAULT_RAW_PATH, verify_hash=True)
    if loaded_ref.month7_opened:
        raise D2V11Error("Reference load opened Month 7.")
    if set(loaded_ref.months) - {0, 1, 2, 3, 4, 5}:
        raise D2V11Error(f"Reference months drifted: {loaded_ref.months}")
    if int((loaded_ref.frame["fraud_bool"] != 0).sum()):
        raise D2V11Error("Reference frame contains fraud rows.")
    if len(loaded_ref.frame) != EXPECTED_REFERENCE_N:
        raise D2V11Error(
            f"Reference n={len(loaded_ref.frame)} != {EXPECTED_REFERENCE_N}"
        )
    if loaded_ref.raw_sha256 != scorer.reference_sha256:
        raise D2V11Error("Raw SHA256 does not match the frozen D2-S v1.0 artefact.")

    print(f"Scoring {len(loaded_ref.frame)} reference rows with frozen D2-S v1.0...", flush=True)
    train_rel = scorer.score_many(loaded_ref.frame)
    print("Fitting Isolation Forest on seven consistency channels...", flush=True)
    aggregator = fit_iforest_aggregator(
        train_rel.loc[:, list(RELATIONSHIP_IDS)],
        v10_fingerprint=scorer.fingerprint,
        month7_opened=False,
    )
    if aggregator.n_train != EXPECTED_REFERENCE_N:
        raise D2V11Error(f"IF train n={aggregator.n_train} != {EXPECTED_REFERENCE_N}")

    d1_scores = load_month6_d1_scores()
    d1_threshold = load_month6_d1_threshold()
    if abs(d1_threshold - D1_THRESHOLD) > 1e-15:
        raise D2V11Error(f"D1 threshold drifted: {d1_threshold}")
    month6 = load_month6_applications(DEFAULT_RAW_PATH, fraud_bool=None, verify_hash=False)
    if month6.month7_opened:
        raise D2DataError("Month-6 load reported month7_opened=True.")
    months = set(int(m) for m in month6.frame["month"].unique())
    if months - set(CALIBRATION_MONTHS) or month6.frame["month"].isin(list(SEALED_MONTHS)).any():
        raise D2DataError("Month-6 frame violated the sealed-month boundary.")

    merged = month6.frame.merge(
        d1_scores,
        left_on="source_row_id",
        right_on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if int((merged["y_true"] != merged["fraud_bool"]).sum()):
        raise D2DataError("D1 score join disagreed with fraud_bool.")
    passed = merged.loc[merged["y_score"] < d1_threshold].copy()
    legit = passed.loc[passed["fraud_bool"] == 0].copy()
    fraud = passed.loc[passed["fraud_bool"] == 1].copy()
    if len(legit) != EXPECTED_LEGIT or len(fraud) != EXPECTED_FRAUD:
        raise D2V11Error(
            f"D1-PASS counts legit={len(legit)} fraud={len(fraud)} "
            f"!= {EXPECTED_LEGIT}/{EXPECTED_FRAUD}"
        )

    print("Scoring Month-6 D1-PASS legitimate and native missed-fraud rows...", flush=True)
    legit_rel = scorer.score_many(legit)
    fraud_rel = scorer.score_many(fraud)
    legit_features = collapse_to_iforest_features(legit_rel)
    fraud_features = collapse_to_iforest_features(fraud_rel)
    legit_v10 = legit_rel["d2_score"].to_numpy(dtype="float64")
    fraud_v10 = fraud_rel["d2_score"].to_numpy(dtype="float64")
    legit_v11 = aggregator.score_features(legit_features)
    fraud_v11 = aggregator.score_features(fraud_features)

    v10_thresholds: dict[str, dict[str, float | int]] = {}
    v11_thresholds: dict[str, dict[str, float | int]] = {}
    for budget in BUDGETS:
        v10_row = budget_threshold(legit_v10, budget)
        expected = EXPECTED_V10_THRESHOLDS[budget]
        if abs(float(v10_row["threshold"]) - expected) > 1e-12:
            raise D2V11Error(
                f"v1.0 {int(budget*100)}% threshold {v10_row['threshold']} != {expected}"
            )
        native_v10 = int((fraud_v10 >= float(v10_row["threshold"])).sum())
        if native_v10 != EXPECTED_V10_NATIVE_FRAUD_REVIEW[budget]:
            raise D2V11Error(
                f"v1.0 native fraud REVIEW @{int(budget*100)}% is {native_v10}, "
                f"expected {EXPECTED_V10_NATIVE_FRAUD_REVIEW[budget]}"
            )
        v10_thresholds[f"{int(budget * 100)}pct"] = v10_row
        v11_thresholds[f"{int(budget * 100)}pct"] = budget_threshold(legit_v11, budget)

    print("Scoring 121 frozen D1-bypass applications...", flush=True)
    attacks = load_attacks(scorer)
    attacks["d2s_v11_anomaly_score"] = aggregator.score_features(
        attacks.loc[:, list(IFOREST_FEATURE_IDS)]
    )

    attack_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    conditions = list(EXPECTED_ATTACKS)
    for budget in BUDGETS:
        key = f"{int(budget * 100)}pct"
        v10_thr = float(v10_thresholds[key]["threshold"])
        v11_thr = float(v11_thresholds[key]["threshold"])
        native_v10 = int((fraud_v10 >= v10_thr).sum())
        native_v11 = int((fraud_v11 >= v11_thr).sum())
        for condition in conditions + ["POOLED"]:
            mask = np.ones(len(attacks), dtype=bool) if condition == "POOLED" else (
                attacks["condition_id"].to_numpy() == condition
            )
            sub_v10 = attacks.loc[mask, "d2s_v10_score"].to_numpy(dtype="float64")
            sub_v11 = attacks.loc[mask, "d2s_v11_anomaly_score"].to_numpy(dtype="float64")
            n_anchors = N_ANCHORS if condition != "POOLED" else N_ANCHORS * 4
            v10_m = metrics_block(scores=sub_v10, threshold=v10_thr, n_anchors=n_anchors)
            v11_m = metrics_block(scores=sub_v11, threshold=v11_thr, n_anchors=n_anchors)
            if condition != "POOLED" and v10_m["review_count"] != EXPECTED_V10_REVIEW[budget][condition]:
                raise D2V11Error(
                    f"v1.0 REVIEW {condition} @{int(budget*100)}% is "
                    f"{v10_m['review_count']}, expected {EXPECTED_V10_REVIEW[budget][condition]}"
                )
            for version, metrics, thr in (
                ("d2s_v1.0", v10_m, v10_thr),
                ("d2s_v1.1", v11_m, v11_thr),
            ):
                row = {
                    "budget": budget,
                    "version": version,
                    "attacker": condition,
                    "threshold": thr,
                    **metrics,
                }
                attack_rows.append(row)
                if budget == PRIMARY_BUDGET:
                    primary_rows.append(row)
        comparison_rows.append(
            {
                "budget": budget,
                "label": (
                    "PRIMARY EXPERIMENTAL REVIEW BUDGET"
                    if budget == PRIMARY_BUDGET
                    else (
                        "lower-friction sensitivity"
                        if budget == 0.05
                        else "higher-review sensitivity"
                    )
                ),
                "v10_legit_review_rate": float(v10_thresholds[key]["empirical_legit_review_rate"]),
                "v11_legit_review_rate": float(v11_thresholds[key]["empirical_legit_review_rate"]),
                "v10_threshold": v10_thr,
                "v11_threshold": v11_thr,
                "v10_native_fraud_review": native_v10,
                "v11_native_fraud_review": native_v11,
                "v10_native_fraud_recovery": native_v10 / EXPECTED_FRAUD,
                "v11_native_fraud_recovery": native_v11 / EXPECTED_FRAUD,
            }
        )
        for condition in conditions + ["POOLED"]:
            v10_row = next(
                r
                for r in attack_rows
                if r["budget"] == budget and r["version"] == "d2s_v1.0" and r["attacker"] == condition
            )
            v11_row = next(
                r
                for r in attack_rows
                if r["budget"] == budget and r["version"] == "d2s_v1.1" and r["attacker"] == condition
            )
            comparison_rows[-1][f"v10_{condition}_interception"] = v10_row["interception_rate"]
            comparison_rows[-1][f"v11_{condition}_interception"] = v11_row["interception_rate"]
            comparison_rows[-1][f"v10_{condition}_review"] = v10_row["review_count"]
            comparison_rows[-1][f"v11_{condition}_review"] = v11_row["review_count"]
            comparison_rows[-1][f"v10_{condition}_e2e"] = v10_row["e2e_bypass_rate"]
            comparison_rows[-1][f"v11_{condition}_e2e"] = v11_row["e2e_bypass_rate"]

    primary_table = pd.DataFrame(primary_rows)
    recommendation, rec_reason = recommend(primary_table)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_PARENT / f"d2s_v11_iforest_{SCORE_CONTRACT_ID_V11}_{stamp}"
    if out_dir.exists():
        raise D2V11Error(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    model_path = out_dir / "D2S_V11_IFOREST_MODEL.joblib"
    config_path = out_dir / "D2S_V11_IFOREST_CONFIG.json"
    aggregator.save(model_path, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "d1_artefact_id": PINNED_D1_ARTEFACT_ID,
            "d1_unchanged": True,
            "d2s_v10_unchanged": True,
            "d2s_v10_reference_path": str(D2S_V10_REFERENCE),
            "d2s_v10_reference_sha256": v10_sha_before,
            "d2s_v10_fingerprint": scorer.fingerprint,
            "training_population": {
                "months": [0, 1, 2, 3, 4, 5],
                "fraud_bool": 0,
                "n": EXPECTED_REFERENCE_N,
                "raw_sha256": loaded_ref.raw_sha256,
            },
            "sklearn_version_runtime": sklearn.__version__,
            "month7_opened": False,
            "attack_outcomes_used_for_fitting": False,
            "hyperparameter_search": False,
            "algorithms_tested": ["IsolationForest"],
            "recommendation": recommendation,
            "recommendation_reason": rec_reason,
        }
    )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    (out_dir / "D2S_V11_THRESHOLDS.json").write_text(
        json.dumps(
            {
                "score_contract_id": SCORE_CONTRACT_ID_V11,
                "decision_rule": "REVIEW if d2s_v11_anomaly_score >= threshold else CLEAR",
                "predict_used": False,
                "contamination_is_not_operating_threshold": True,
                "primary_experimental_review_budget": PRIMARY_BUDGET,
                "thresholds": v11_thresholds,
                "v10_comparison_thresholds": v10_thresholds,
                "n_legitimate_d1_pass": EXPECTED_LEGIT,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    legit_out = legit_features.copy()
    legit_out.insert(0, "source_row_id", legit["source_row_id"].to_numpy())
    legit_out["d2s_v10_score"] = legit_v10
    legit_out["d2s_v11_anomaly_score"] = legit_v11
    legit_out.to_csv(out_dir / "D2S_V11_LEGITIMATE_SCORES.csv", index=False)

    fraud_out = fraud_features.copy()
    fraud_out.insert(0, "source_row_id", fraud["source_row_id"].to_numpy())
    fraud_out["d2s_v10_score"] = fraud_v10
    fraud_out["d2s_v11_anomaly_score"] = fraud_v11
    for budget in BUDGETS:
        key = f"{int(budget * 100)}pct"
        fraud_out[f"v11_review_{int(budget * 100)}pct"] = (
            fraud_v11 >= float(v11_thresholds[key]["threshold"])
        ).astype("int64")
        fraud_out[f"v10_review_{int(budget * 100)}pct"] = (
            fraud_v10 >= float(v10_thresholds[key]["threshold"])
        ).astype("int64")
    fraud_out.to_csv(out_dir / "D2S_V11_NATIVE_FRAUD_RESULTS.csv", index=False)

    attack_out = attacks.copy()
    for budget in BUDGETS:
        key = f"{int(budget * 100)}pct"
        attack_out[f"v10_review_{int(budget * 100)}pct"] = (
            attack_out["d2s_v10_score"] >= float(v10_thresholds[key]["threshold"])
        ).astype("int64")
        attack_out[f"v11_review_{int(budget * 100)}pct"] = (
            attack_out["d2s_v11_anomaly_score"] >= float(v11_thresholds[key]["threshold"])
        ).astype("int64")
    attack_out.to_csv(out_dir / "D2S_V11_ATTACK_RESULTS.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(out_dir / "D2S_V10_V11_COMPARISON.csv", index=False)

    v10_sha_after = file_sha256(D2S_V10_REFERENCE)
    if v10_sha_after != v10_sha_before:
        raise D2V11Error("D2-S v1.0 reference artefact changed during the run.")

    def fmt_attackers(budget: float, version: str) -> str:
        lines = []
        for condition in conditions + ["POOLED"]:
            row = next(
                r
                for r in attack_rows
                if r["budget"] == budget and r["version"] == version and r["attacker"] == condition
            )
            lines.append(
                f"| {row['attacker']} | {row['n_d1_success']} | {row['review_count']} | "
                f"{row['interception_rate']:.3f} | {row['clear_count']} | "
                f"{row['e2e_bypass_rate']:.3f} |"
            )
        return "\n".join(lines)

    primary = next(r for r in comparison_rows if r["budget"] == PRIMARY_BUDGET)
    dist_legit_v10 = summarise(legit_v10)
    dist_legit_v11 = summarise(legit_v11)
    dist_fraud_v10 = summarise(fraud_v10)
    dist_fraud_v11 = summarise(fraud_v11)
    dist_atk_v10 = summarise(attacks["d2s_v10_score"].to_numpy(dtype="float64"))
    dist_atk_v11 = summarise(attacks["d2s_v11_anomaly_score"].to_numpy(dtype="float64"))

    report = f"""# D2-S v1.1 Isolation Forest development report

Created: {aggregator.fitted_utc}
Output: `{out_dir}`

This is a **development candidate**. D2-S v1.0 was not replaced.

## Recommendation

**{recommendation}**

{rec_reason}

Rule used: freeze v1.1 only if, at the PRIMARY EXPERIMENTAL REVIEW BUDGET (10%),
pooled REVIEW rises by at least 5 cases and at least two of four attackers
improve. Month 6 is development data. Month 7 remains sealed.

## Isolation

- D1 unchanged: `{PINNED_D1_ARTEFACT_ID}`
- D2-S v1.0 unchanged: `{scorer.fingerprint}`
- v1.0 artefact SHA256 before/after: `{v10_sha_before}`
- Month 7 opened: false
- Fraud labels used in Isolation Forest fit: false
- D1 continuous score used: false
- Attacker outcomes used in fit: false
- Raw application features entering Isolation Forest: false
- Algorithms tested: IsolationForest only
- Hyperparameter search: none

## Model

- Contract: `{SCORE_CONTRACT_ID_V11}`
- Training: Months 0–5, fraud_bool == 0, n = {EXPECTED_REFERENCE_N}
- Features (7): payment_channel = max(C01, C14); C13; C09; C03; C10; C11; C15
- sklearn: {sklearn.__version__}
- Parameters: {json.dumps(FIXED_IFOREST_PARAMS, sort_keys=True)}
- Score: `d2s_v11_anomaly_score = -IsolationForest.score_samples(X)`
- `predict()` is not the operating rule
- Internal contamination threshold is not the bank REVIEW threshold

## Thresholds (Month-6 legitimate D1-PASS only, n = {EXPECTED_LEGIT})

| Budget | Label | v1.1 threshold | Realised legit REVIEW | Ties at threshold |
|---|---|---:|---:|---:|
| 5% | lower-friction sensitivity | {v11_thresholds['5pct']['threshold']:.6f} | {v11_thresholds['5pct']['n_legitimate_review']} ({float(v11_thresholds['5pct']['empirical_legit_review_rate']):.4f}) | {v11_thresholds['5pct']['n_scores_equal_to_threshold']} |
| 10% | PRIMARY EXPERIMENTAL REVIEW BUDGET | {v11_thresholds['10pct']['threshold']:.6f} | {v11_thresholds['10pct']['n_legitimate_review']} ({float(v11_thresholds['10pct']['empirical_legit_review_rate']):.4f}) | {v11_thresholds['10pct']['n_scores_equal_to_threshold']} |
| 15% | higher-review sensitivity | {v11_thresholds['15pct']['threshold']:.6f} | {v11_thresholds['15pct']['n_legitimate_review']} ({float(v11_thresholds['15pct']['empirical_legit_review_rate']):.4f}) | {v11_thresholds['15pct']['n_scores_equal_to_threshold']} |

10% is an experimental legitimate-review budget among D1-PASS applications, not a bank-optimal operating point.

## Direct v1.0 vs v1.1 comparison at 10%

| Metric | D2-S v1.0 | D2-S v1.1 |
|---|---:|---:|
| Realised legit REVIEW | {primary['v10_legit_review_rate']:.4f} | {primary['v11_legit_review_rate']:.4f} |
| A0 interception | {primary['v10_A0_interception']:.3f} ({primary['v10_A0_review']}/26) | {primary['v11_A0_interception']:.3f} ({primary['v11_A0_review']}/26) |
| A1-Pro interception | {primary['v10_A1-Pro_interception']:.3f} ({primary['v10_A1-Pro_review']}/27) | {primary['v11_A1-Pro_interception']:.3f} ({primary['v11_A1-Pro_review']}/27) |
| A2 interception | {primary['v10_A2_interception']:.3f} ({primary['v10_A2_review']}/37) | {primary['v11_A2_interception']:.3f} ({primary['v11_A2_review']}/37) |
| A3-Pro interception | {primary['v10_A3-Pro_interception']:.3f} ({primary['v10_A3-Pro_review']}/31) | {primary['v11_A3-Pro_interception']:.3f} ({primary['v11_A3-Pro_review']}/31) |
| Pooled interception | {primary['v10_POOLED_interception']:.3f} ({primary['v10_POOLED_review']}/121) | {primary['v11_POOLED_interception']:.3f} ({primary['v11_POOLED_review']}/121) |
| Pooled E2E bypass | {primary['v10_POOLED_e2e']:.3f} | {primary['v11_POOLED_e2e']:.3f} |
| Native missed-fraud recovery | {primary['v10_native_fraud_review']}/705 ({primary['v10_native_fraud_recovery']:.4f}) | {primary['v11_native_fraud_review']}/705 ({primary['v11_native_fraud_recovery']:.4f}) |

## v1.1 attack evaluation

Interception = REVIEW / D1-PASS attacks. E2E bypass = CLEAR / original anchors (50; pooled 200).

### 5% lower-friction sensitivity

| Attacker | D1 success | REVIEW | Interception | CLEAR | E2E bypass |
|---|---:|---:|---:|---:|---:|
{fmt_attackers(0.05, "d2s_v1.1")}

### 10% PRIMARY EXPERIMENTAL REVIEW BUDGET

| Attacker | D1 success | REVIEW | Interception | CLEAR | E2E bypass |
|---|---:|---:|---:|---:|---:|
{fmt_attackers(0.10, "d2s_v1.1")}

### 15% higher-review sensitivity

| Attacker | D1 success | REVIEW | Interception | CLEAR | E2E bypass |
|---|---:|---:|---:|---:|---:|
{fmt_attackers(0.15, "d2s_v1.1")}

## Score distributions

| Population | Version | n | mean | p50 | p90 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| Month-6 legit D1-PASS | v1.0 | {dist_legit_v10['n']} | {dist_legit_v10['mean']:.4f} | {dist_legit_v10['p50']:.4f} | {dist_legit_v10['p90']:.4f} | {dist_legit_v10['p99']:.4f} | {dist_legit_v10['max']:.4f} |
| Month-6 legit D1-PASS | v1.1 | {dist_legit_v11['n']} | {dist_legit_v11['mean']:.4f} | {dist_legit_v11['p50']:.4f} | {dist_legit_v11['p90']:.4f} | {dist_legit_v11['p99']:.4f} | {dist_legit_v11['max']:.4f} |
| Native missed fraud | v1.0 | {dist_fraud_v10['n']} | {dist_fraud_v10['mean']:.4f} | {dist_fraud_v10['p50']:.4f} | {dist_fraud_v10['p90']:.4f} | {dist_fraud_v10['p99']:.4f} | {dist_fraud_v10['max']:.4f} |
| Native missed fraud | v1.1 | {dist_fraud_v11['n']} | {dist_fraud_v11['mean']:.4f} | {dist_fraud_v11['p50']:.4f} | {dist_fraud_v11['p90']:.4f} | {dist_fraud_v11['p99']:.4f} | {dist_fraud_v11['max']:.4f} |
| Successful D1 bypasses | v1.0 | {dist_atk_v10['n']} | {dist_atk_v10['mean']:.4f} | {dist_atk_v10['p50']:.4f} | {dist_atk_v10['p90']:.4f} | {dist_atk_v10['p99']:.4f} | {dist_atk_v10['max']:.4f} |
| Successful D1 bypasses | v1.1 | {dist_atk_v11['n']} | {dist_atk_v11['mean']:.4f} | {dist_atk_v11['p50']:.4f} | {dist_atk_v11['p90']:.4f} | {dist_atk_v11['p99']:.4f} | {dist_atk_v11['max']:.4f} |

v1.0 and v1.1 scores are not on a shared numeric scale. Compare them only at matched legitimate-review budgets.
"""
    (out_dir / "D2S_V11_IFOREST_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "recommendation": recommendation}, indent=2))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
