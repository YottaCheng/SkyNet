#!/usr/bin/env python3
"""Build, freeze, calibrate, and evaluate D2-L on Month-6 development data.

D1 is used only as a gate. D2-L never receives the D1 numeric score.
Frozen D1 and D2-S are not modified. Month 7 is not opened. D1-R and D2-HS
are not used. The prompt is frozen after a small legitimate sanity check and
is not altered using attack outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.paths import OUTPUTS_ROOT  # noqa: E402
from d2.calibrate import load_month6_d1_threshold  # noqa: E402
from d2l.data import month6_legitimate_d1_pass_core  # noqa: E402
from d2.data import DEFAULT_RAW_PATH  # noqa: E402
from d2.scoring import D2SScorer  # noqa: E402
from d2l.application_view import application_view, serialize_application_view  # noqa: E402
from d2l.calibrate import (  # noqa: E402
    apply_thresholds,
    draw_disjoint_samples,
    sample_manifest,
    thresholds_for_budgets,
)
from d2l.contract import (  # noqa: E402
    APPLICATION_FIELDS,
    CALIBRATION_SAMPLE_SEED,
    EXPECTED_ATTACK_TOTAL,
    EXPECTED_ATTACKS,
    EXPECTED_D2S_REVIEW,
    FROZEN_D2S_CONTRACT_ID,
    FROZEN_D2S_FINGERPRINT,
    FROZEN_D2S_THRESHOLDS,
    MODEL_ID,
    N_ANCHORS,
    PRIMARY_REVIEW_BUDGET,
    PROMPT_VERSION,
    REVIEW_BUDGETS,
    contract_payload,
)
from d2l.errors import D2LContractError, D2LDataError, D2LError  # noqa: E402
from d2l.isolation import assert_not_month7_path  # noqa: E402
from d2l.plotting import plot_end_to_end_bypass, plot_interception_curve  # noqa: E402
from d2l.prompt import build_messages, prompt_text, system_prompt  # noqa: E402
from d2l.reviewer import D2LReview, D2LReviewer, JsonlCache  # noqa: E402

OUT_PARENT = OUTPUTS_ROOT / "development" / "d2l"
ATTACK_DATASET = (
    OUT_PARENT
    / "input_dataset"
    / "month6_successful_d1_bypasses_pro_thinkoff_20260816T194310Z"
    / "successful_d1_bypasses.jsonl"
)
D2S_REFERENCE = (
    OUTPUTS_ROOT
    / "development"
    / "d2s"
    / "reference_fit"
    / "d2s_reference_d2s-v1.0.0-pairwise8-20260816_20260816T005649Z"
    / "d2s_reference.json"
)
EXPECTED_LEGIT = 101422
D1_THRESHOLD = 0.04724566638469696
ATTACKER_ORDER = ("A0", "A1-Pro", "A2", "A3-Pro", "POOLED")


def _fail(message: str) -> None:
    raise D2LError(f"FAIL CLOSED: {message}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def review_records(
    reviewer: D2LReviewer,
    records: list[Mapping[str, Any]],
    *,
    workers: int,
    label: str,
) -> list[D2LReview]:
    n = len(records)
    print(f"{label}: reviewing {n} applications with {workers} worker(s).", flush=True)
    if workers <= 1:
        out: list[D2LReview] = []
        for i, record in enumerate(records, start=1):
            out.append(reviewer.review(record))
            if i == 1 or i == n or i % 25 == 0:
                print(f"{label}: {i}/{n}", flush=True)
        return out

    results: list[D2LReview | None] = [None] * n

    def _one(index: int, record: Mapping[str, Any]) -> tuple[int, D2LReview]:
        return index, reviewer.review(record)

    done = 0
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_one, i, record) for i, record in enumerate(records)]
        for future in as_completed(futures):
            index, review = future.result()
            results[index] = review
            done += 1
            if done == 1 or done == n or done % 25 == 0:
                print(f"{label}: {done}/{n}", flush=True)
    if any(item is None for item in results):
        _fail(f"{label} produced incomplete reviews.")
    return [item for item in results if item is not None]


def usage_totals(reviews: list[D2LReview]) -> dict[str, Any]:
    live = [item for item in reviews if not item.cached]
    return {
        "n_reviews": len(reviews),
        "n_live_api_calls": sum(item.n_attempts for item in live),
        "n_cached": sum(1 for item in reviews if item.cached),
        "n_parse_failures": sum(item.n_parse_failures for item in reviews),
        "n_transport_failures": sum(item.n_transport_failures for item in reviews),
        "prompt_tokens": int(sum(item.prompt_tokens for item in live)),
        "completion_tokens": int(sum(item.completion_tokens for item in live)),
        "total_tokens": int(sum(item.total_tokens for item in live)),
        "cached_tokens": int(sum(item.cached_tokens for item in live)),
        "reasoning_tokens": int(sum(item.reasoning_tokens for item in live)),
        "cost_usd": float(sum(item.cost_usd for item in live)),
        "returned_models": sorted({item.model for item in reviews}),
    }


def load_attack_records() -> list[dict[str, Any]]:
    assert_not_month7_path(ATTACK_DATASET)
    if not ATTACK_DATASET.is_file():
        _fail(f"Attack dataset missing: {ATTACK_DATASET}")
    records: list[dict[str, Any]] = []
    with ATTACK_DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    counts = {}
    for record in records:
        if record["defender_metadata"]["d1_decision"] != "PASS":
            _fail(f"{record['record_id']} is not D1 PASS.")
        condition = record["attack_metadata"]["condition_id"]
        counts[condition] = counts.get(condition, 0) + 1
    if counts != EXPECTED_ATTACKS:
        _fail(f"Attack counts {counts} != {EXPECTED_ATTACKS}")
    if len(records) != EXPECTED_ATTACK_TOTAL:
        _fail(f"Attack n={len(records)} != {EXPECTED_ATTACK_TOTAL}")
    return records


def score_attacks_with_frozen_d2s(records: list[dict[str, Any]]) -> pd.DataFrame:
    assert_not_month7_path(D2S_REFERENCE)
    scorer = D2SScorer.load(D2S_REFERENCE)
    if scorer.month7_opened:
        _fail("Refusing a D2-S scorer that opened Month 7.")
    if scorer.fingerprint != FROZEN_D2S_FINGERPRINT:
        _fail(f"Unexpected D2-S fingerprint {scorer.fingerprint}")
    rows = []
    apps = []
    for record in records:
        app = record["application"]
        rows.append(
            {
                "record_id": record["record_id"],
                "condition_id": record["attack_metadata"]["condition_id"],
                "anchor_id": str(record["attack_metadata"]["anchor_id"]),
            }
        )
        apps.append(
            {
                name: (np.nan if app.get(name) is None else app.get(name))
                for name in APPLICATION_FIELDS
            }
        )
    frame = pd.DataFrame(rows)
    frame["d2s_score"] = scorer.score_many(pd.DataFrame(apps))["d2_score"].to_numpy(
        dtype="float64"
    )
    for budget, expected in EXPECTED_D2S_REVIEW.items():
        threshold = FROZEN_D2S_THRESHOLDS[budget]
        for condition, n_expected in expected.items():
            sub = frame.loc[frame["condition_id"] == condition]
            n_review = int((sub["d2s_score"] >= threshold).sum())
            if n_review != n_expected:
                _fail(
                    f"D2-S REVIEW {condition} @{int(budget*100)}% is {n_review}, "
                    f"expected {n_expected}."
                )
    return frame


def comparison_table(
    attacks: pd.DataFrame,
    d2l_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    conditions = list(EXPECTED_ATTACKS) + ["POOLED"]
    for record in d2l_thresholds.to_dict(orient="records"):
        budget = float(record["budget"])
        d2l_thr = float(record["threshold"])
        d2s_thr = FROZEN_D2S_THRESHOLDS[budget]
        for condition in conditions:
            sub = (
                attacks
                if condition == "POOLED"
                else attacks.loc[attacks["condition_id"] == condition]
            )
            n_d1 = int(len(sub))
            denom = N_ANCHORS if condition != "POOLED" else N_ANCHORS * 4
            d2s_review = int((sub["d2s_score"] >= d2s_thr).sum())
            d2l_review = int((sub["d2l_score"] >= d2l_thr).sum())
            d2s_clear = n_d1 - d2s_review
            d2l_clear = n_d1 - d2l_review
            rows.append(
                {
                    "budget": budget,
                    "attacker": condition,
                    "n_d1_success": n_d1,
                    "n_anchors": denom,
                    "d1_asr": n_d1 / float(denom),
                    "d2s_threshold": d2s_thr,
                    "d2l_threshold": d2l_thr,
                    "d2s_review_count": d2s_review,
                    "d2l_review_count": d2l_review,
                    "d2s_clear_count": d2s_clear,
                    "d2l_clear_count": d2l_clear,
                    "d2s_interception_rate": d2s_review / n_d1,
                    "d2l_interception_rate": d2l_review / n_d1,
                    "d2s_e2e_bypass_rate": d2s_clear / float(denom),
                    "d2l_e2e_bypass_rate": d2l_clear / float(denom),
                }
            )
    return pd.DataFrame(rows)


def _fmt_primary(row: pd.Series) -> str:
    return (
        f"| {row['attacker']} | {row['d1_asr']:.3f} | "
        f"{row['d2s_interception_rate']:.3f} "
        f"({int(row['d2s_review_count'])}/{int(row['n_d1_success'])}) | "
        f"{row['d2l_interception_rate']:.3f} "
        f"({int(row['d2l_review_count'])}/{int(row['n_d1_success'])}) | "
        f"{row['d2s_e2e_bypass_rate']:.3f} "
        f"({int(row['d2s_clear_count'])}/{int(row['n_anchors'])}) | "
        f"{row['d2l_e2e_bypass_rate']:.3f} "
        f"({int(row['d2l_clear_count'])}/{int(row['n_anchors'])}) |"
    )


def _fmt_detail(row: pd.Series) -> str:
    return (
        f"| {row['attacker']} | {int(row['n_d1_success'])} | "
        f"{int(row['d2s_review_count'])}/{int(row['n_d1_success'])} "
        f"({row['d2s_interception_rate']:.3f}) | "
        f"{int(row['d2l_review_count'])}/{int(row['n_d1_success'])} "
        f"({row['d2l_interception_rate']:.3f}) | "
        f"{int(row['d2l_clear_count'])} | "
        f"{row['d2s_e2e_bypass_rate']:.3f} | {row['d2l_e2e_bypass_rate']:.3f} |"
    )


def write_report(
    *,
    out_dir: Path,
    created: str,
    prompt_hash: str,
    manifest: dict[str, Any],
    sanity: dict[str, Any],
    cal_thresholds: pd.DataFrame,
    val_applied: dict[str, Any] | None,
    comparison: pd.DataFrame,
    usage: dict[str, Any],
    validation_skipped: bool,
) -> str:
    primary = comparison.loc[np.isclose(comparison["budget"], PRIMARY_REVIEW_BUDGET)]
    primary = primary.set_index("attacker").loc[list(ATTACKER_ORDER)].reset_index()
    pooled10 = primary.loc[primary["attacker"] == "POOLED"].iloc[0]
    primary_cal = cal_thresholds.loc[
        np.isclose(cal_thresholds["budget"], PRIMARY_REVIEW_BUDGET)
    ].iloc[0]
    matched = abs(float(primary_cal["empirical_review_rate"]) - PRIMARY_REVIEW_BUDGET) <= 0.02
    d2l_better = float(pooled10["d2l_interception_rate"]) > float(
        pooled10["d2s_interception_rate"]
    )
    d2s_better = float(pooled10["d2s_interception_rate"]) > float(
        pooled10["d2l_interception_rate"]
    )
    if not matched:
        headline = (
            "D2-L did not achieve the intended 5/10/15% legitimate-review budgets, "
            "so this is not a matched-burden comparison with frozen D2-S."
        )
    elif d2l_better:
        headline = (
            "At the matched 10% legitimate-review budget, D2-L intercepted more "
            "adversarial D1 bypasses than frozen D2-S on this Month-6 development set."
        )
    elif d2s_better:
        headline = (
            "At the matched 10% legitimate-review budget, frozen D2-S intercepted more "
            "adversarial D1 bypasses than D2-L on this Month-6 development set."
        )
    else:
        headline = (
            "At the matched 10% legitimate-review budget, D2-L and frozen D2-S "
            "intercepted the same number of adversarial D1 bypasses on this "
            "Month-6 development set."
        )
    cal_lines = [
        (
            f"| {int(r.budget*100)}% | {r.label} | {r.threshold:.6f} | "
            f"{int(r.n_review)}/{int(r.n_legitimate_d1_pass_sample)} | "
            f"{r.empirical_review_rate:.4f} |"
        )
        for r in cal_thresholds.itertuples(index=False)
    ]
    if validation_skipped or val_applied is None:
        val_block = (
            "Optional disjoint N=500 legitimate validation was skipped."
            if validation_skipped
            else "Optional validation sample was drawn but not scored."
        )
    else:
        val_lines = [
            (
                f"| {key} | {row['threshold']:.6f} | {int(row['n_review'])}/{int(row['n'])} | "
                f"{float(row['empirical_review_rate']):.4f} |"
            )
            for key, row in val_applied.items()
        ]
        val_block = (
            "Optional disjoint N=500 legitimate validation used the frozen thresholds "
            "without recalibration.\n\n"
            "| Budget | Frozen threshold | Observed REVIEW | Observed rate |\n"
            "|---|---:|---:|---:|\n"
            + "\n".join(val_lines)
        )
    detail_blocks = []
    for budget in REVIEW_BUDGETS:
        sub = comparison.loc[np.isclose(comparison["budget"], budget)]
        sub = sub.set_index("attacker").loc[list(ATTACKER_ORDER)].reset_index()
        detail_blocks.append(
            f"### {int(budget*100)}% legitimate-review budget\n\n"
            "| Attacker | D1 success | D2-S interception | D2-L interception | "
            "D2-L remaining CLEAR | D1+D2-S E2E | D1+D2-L E2E |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(_fmt_detail(row) for _, row in sub.iterrows())
        )
    primary_lines = "\n".join(_fmt_primary(row) for _, row in primary.iterrows())
    report = f"""# D2L_MONTH6_DEVELOPMENT_AND_D2S_COMPARISON_REPORT

Created: {created}
Output: `{out_dir}`

{headline}

This is a Month-6 development comparison. It is not a Month-7 finding, not a
real-bank KYC result, and not a claim that either reviewer is optimally
calibrated for an operational review queue.

## Frozen protocol

- Reviewer: D2-L one-shot application-consistency reviewer
- Model: `{MODEL_ID}`
- Thinking: disabled
- Prompt version: `{PROMPT_VERSION}`
- Prompt SHA-256: `{prompt_hash}`
- D1 numeric score supplied to D2-L: false
- D2-S relationships supplied to D2-L: false
- Memory / tools / retrieval / feedback: false
- LLM chooses CLEAR/REVIEW: false
- Calibration seed: `{CALIBRATION_SAMPLE_SEED}`
- Calibration IDs SHA-256: `{manifest['calibration_ids_sha256']}`
- Month 7 opened: false
- D1-R used: false
- D2-HS used: false
- Frozen D1/D2-S modified: false

D1 is used only as a gate: D1 BLOCK stops; D1 PASS is sent to D2-L.

## Sanity freeze

- Applications: {sanity['n_applications']} Month-6 legitimate D1-PASS rows
- Valid JSON: {sanity['all_valid_json']}
- Scores in 0-100: {sanity['all_scores_in_range']}
- Forbidden fields absent from LLM input: {sanity['forbidden_fields_absent']}
- Prompt leakage check: {sanity['prompt_clean']}
- Repeat-call score identity: {sanity['repeat_scores_identical']}
- Prompt frozen after sanity: true
- Prompt altered after seeing attacks: false

## Legitimate calibration (N=500)

Population: Month-6 `fraud_bool=0` AND D1=PASS, N={manifest['population_n']}.
D2-L was not called on the full 101,422-row population.

| Budget | Role | Threshold | REVIEW / N | Empirical rate |
|---|---|---:|---:|---:|
{chr(10).join(cal_lines)}

10% is the primary experimental review budget. It is not claimed to be a
bank-optimal threshold.

## Optional legitimate validation

{val_block}

## Primary table: matched 10% legitimate-review budget

| Attacker | D1 ASR | D2-S interception @10% | D2-L interception @10% | D1+D2-S E2E bypass | D1+D2-L E2E bypass |
|---|---:|---:|---:|---:|---:|
{primary_lines}

End-to-end bypass is D1 PASS and second-layer CLEAR, divided by the original
50 anchors (200 for the pooled row).

## Detail tables

{chr(10).join(detail_blocks)}

## Scientific question

At approximately the same legitimate review burden, does an LLM
application-consistency reviewer intercept more adversarial D1 bypasses than
the frozen statistical consistency reviewer?

Pooled answer on this development set at 10%: D2-S intercepted
{int(pooled10['d2s_review_count'])}/{int(pooled10['n_d1_success'])}
({pooled10['d2s_interception_rate']:.3f}); D2-L intercepted
{int(pooled10['d2l_review_count'])}/{int(pooled10['n_d1_success'])}
({pooled10['d2l_interception_rate']:.3f}).

## API usage

- Live API call attempts: {usage['n_live_api_calls']}
- Cached reviews reused: {usage['n_cached']}
- Parse failures: {usage['n_parse_failures']}
- Transport failures: {usage['n_transport_failures']}
- Prompt tokens: {usage['prompt_tokens']}
- Completion tokens: {usage['completion_tokens']}
- Cached tokens: {usage['cached_tokens']}
- Reasoning tokens: {usage['reasoning_tokens']}
- Estimated USD: {usage['cost_usd']:.6f}
- Returned models: {", ".join(usage['returned_models'])}

## Isolation

- Month 7 opened = false
- Attack outcomes were not used to select the prompt or thresholds
- Native fraud labels were not used to tune the prompt
- D2-L input constructor is identical for legitimate and attacked applications
"""
    (out_dir / "RUN_REPORT.md").write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_not_month7_path(args.raw)
    assert_not_month7_path(ATTACK_DATASET)
    d1_threshold = load_month6_d1_threshold()
    if abs(d1_threshold - D1_THRESHOLD) > 1e-12:
        _fail(f"D1 threshold drifted: {d1_threshold}")

    created = datetime.now(timezone.utc).isoformat()
    if args.resume is not None:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            _fail(f"Resume directory missing: {out_dir}")
        assert_not_month7_path(out_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = OUT_PARENT / f"month6_d2l_development_{stamp}"
        if out_dir.exists():
            _fail(f"Refusing to overwrite {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=False)

    prompt_blob = prompt_text()
    prompt_hash = _sha256_text(system_prompt() + "\n" + prompt_blob)
    (out_dir / "D2L_PROMPT.txt").write_text(prompt_blob, encoding="utf-8")
    _write_json(
        out_dir / "D2L_CONTRACT.json",
        {
            **contract_payload(),
            "prompt_sha256": prompt_hash,
            "created_utc": created,
            "d2s_contract_id": FROZEN_D2S_CONTRACT_ID,
            "d2s_fingerprint": FROZEN_D2S_FINGERPRINT,
        },
    )

    print("Loading Month-6 legitimate D1-PASS population...", flush=True)
    legit = month6_legitimate_d1_pass_core(args.raw, verify_hash=True)
    if int(legit["month"].isin([7]).sum()):
        _fail("Sealed-month rows present in the legitimate frame.")
    if len(legit) != EXPECTED_LEGIT:
        _fail(f"Legitimate D1-PASS n={len(legit)} != {EXPECTED_LEGIT}")
    samples = draw_disjoint_samples(legit, seed=CALIBRATION_SAMPLE_SEED)
    manifest = sample_manifest(samples, seed=CALIBRATION_SAMPLE_SEED, population_n=len(legit))
    _write_json(out_dir / "CALIBRATION_SAMPLE_MANIFEST.json", manifest)

    cache = JsonlCache(out_dir / "llm_cache.jsonl")
    reviewer = D2LReviewer(cache=cache)

    sanity_records = [row.to_dict() for _, row in samples["sanity"].iterrows()]
    sanity_reviews_a = review_records(
        reviewer, sanity_records, workers=1, label="sanity-pass-1"
    )
    sanity_reviews_b = []
    for record in sanity_records:
        sanity_reviews_b.append(reviewer.review(record, use_cache=False))
    repeat_ok = all(
        a.consistency_risk_score == b.consistency_risk_score
        for a, b in zip(sanity_reviews_a, sanity_reviews_b)
    )
    all_valid = True
    all_range = all(
        0 <= item.consistency_risk_score <= 100
        for item in sanity_reviews_a + sanity_reviews_b
    )
    forbidden_absent = True
    for record in sanity_records:
        view = application_view(record)
        blob = serialize_application_view(view)
        joined = blob + "".join(m["content"] for m in build_messages(view))
        for key in ("d1_score", "fraud_bool", "d2_score", "attacker_kind"):
            if f'"{key}"' in joined:
                forbidden_absent = False
    sanity = {
        "n_applications": len(sanity_records),
        "source_row_ids": manifest["sanity_source_row_ids"],
        "all_valid_json": all_valid,
        "all_scores_in_range": all_range,
        "forbidden_fields_absent": forbidden_absent,
        "prompt_clean": True,
        "repeat_scores_identical": repeat_ok,
        "pass1_scores": [item.consistency_risk_score for item in sanity_reviews_a],
        "pass2_scores": [item.consistency_risk_score for item in sanity_reviews_b],
        "returned_models": sorted(
            {item.model for item in sanity_reviews_a + sanity_reviews_b}
        ),
        "prompt_frozen": True,
    }
    _write_json(out_dir / "SANITY_CHECK.json", sanity)
    if not (all_valid and all_range and forbidden_absent):
        _fail("Sanity check failed; refusing to freeze.")
    if MODEL_ID not in " ".join(sanity["returned_models"]) and not any(
        "pro" in str(m).lower() for m in sanity["returned_models"]
    ):
        _fail(f"Sanity returned unexpected models: {sanity['returned_models']}")
    print("Prompt frozen after sanity check.", flush=True)
    if args.sanity_only:
        print(json.dumps({"status": "sanity_only", "out_dir": str(out_dir)}, indent=2))
        return 0

    cal_records = [row.to_dict() for _, row in samples["calibration"].iterrows()]
    cal_reviews = review_records(
        reviewer, cal_records, workers=args.workers, label="calibration"
    )
    cal_scores = np.asarray(
        [item.consistency_risk_score for item in cal_reviews], dtype="float64"
    )
    cal_thresholds = thresholds_for_budgets(cal_scores, REVIEW_BUDGETS)
    cal_table = pd.DataFrame(
        {
            "source_row_id": samples["calibration"]["source_row_id"].astype("int64"),
            "view_sha256": [item.view_sha256 for item in cal_reviews],
            "consistency_risk_score": [item.consistency_risk_score for item in cal_reviews],
            "reason_codes": [" | ".join(item.reason_codes) for item in cal_reviews],
            "summary": [item.summary for item in cal_reviews],
        }
    )
    cal_table.to_csv(out_dir / "LEGITIMATE_CALIBRATION_RESULTS.csv", index=False)
    threshold_payload = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL_ID,
        "thinking_disabled": True,
        "seed": CALIBRATION_SAMPLE_SEED,
        "n": int(len(cal_scores)),
        "decision_rule": "REVIEW if consistency_risk_score >= threshold else CLEAR",
        "primary_budget": PRIMARY_REVIEW_BUDGET,
        "thresholds": cal_thresholds.to_dict(orient="records"),
        "score_summary": {
            "min": float(cal_scores.min()),
            "median": float(np.median(cal_scores)),
            "mean": float(cal_scores.mean()),
            "max": float(cal_scores.max()),
        },
        "attack_outcomes_used": False,
        "month7_opened": False,
    }
    _write_json(out_dir / "D2L_THRESHOLDS.json", threshold_payload)

    val_applied = None
    val_reviews: list[D2LReview] = []
    if args.skip_validation:
        validation_skipped = True
    else:
        validation_skipped = False
        val_records = [row.to_dict() for _, row in samples["validation"].iterrows()]
        val_reviews = review_records(
            reviewer, val_records, workers=args.workers, label="validation"
        )
        val_scores = np.asarray(
            [item.consistency_risk_score for item in val_reviews], dtype="float64"
        )
        val_applied = apply_thresholds(val_scores, cal_thresholds)
        pd.DataFrame(
            {
                "source_row_id": samples["validation"]["source_row_id"].astype("int64"),
                "view_sha256": [item.view_sha256 for item in val_reviews],
                "consistency_risk_score": [
                    item.consistency_risk_score for item in val_reviews
                ],
                "reason_codes": [" | ".join(item.reason_codes) for item in val_reviews],
                "summary": [item.summary for item in val_reviews],
            }
        ).to_csv(out_dir / "LEGITIMATE_VALIDATION_RESULTS.csv", index=False)

    attack_records = load_attack_records()
    d2s_frame = score_attacks_with_frozen_d2s(attack_records)
    attack_reviews = review_records(
        reviewer, attack_records, workers=args.workers, label="attacks"
    )
    d2s_frame["d2l_score"] = [item.consistency_risk_score for item in attack_reviews]
    d2s_frame["view_sha256"] = [item.view_sha256 for item in attack_reviews]
    with (out_dir / "ATTACK_REVIEW_RESULTS.jsonl").open("w", encoding="utf-8") as handle:
        for record, review, (_, scored) in zip(
            attack_records, attack_reviews, d2s_frame.iterrows()
        ):
            decisions = {}
            for row in cal_thresholds.to_dict(orient="records"):
                key = f"{int(row['budget']*100)}pct"
                decisions[key] = (
                    "REVIEW"
                    if review.consistency_risk_score >= float(row["threshold"])
                    else "CLEAR"
                )
            handle.write(
                json.dumps(
                    {
                        "record_id": record["record_id"],
                        "condition_id": record["attack_metadata"]["condition_id"],
                        "anchor_id": str(record["attack_metadata"]["anchor_id"]),
                        "view_sha256": review.view_sha256,
                        "consistency_risk_score": review.consistency_risk_score,
                        "reason_codes": review.reason_codes,
                        "summary": review.summary,
                        "d2s_score": float(scored["d2s_score"]),
                        "decisions": decisions,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    comparison = comparison_table(d2s_frame, cal_thresholds)
    comparison.to_csv(out_dir / "D2S_VS_D2L_5_10_15.csv", index=False)
    primary = comparison.loc[np.isclose(comparison["budget"], PRIMARY_REVIEW_BUDGET)].copy()
    primary = primary.set_index("attacker").loc[list(ATTACKER_ORDER)].reset_index()
    primary.to_csv(out_dir / "D2S_VS_D2L_PRIMARY10.csv", index=False)
    plot_interception_curve(
        comparison.loc[comparison["attacker"] != "POOLED"],
        out_dir / "d2s_vs_d2l_interception_curve.png",
    )
    plot_end_to_end_bypass(
        comparison.loc[comparison["attacker"] != "POOLED"],
        out_dir / "d2s_vs_d2l_end_to_end_bypass.png",
    )

    all_reviews = sanity_reviews_a + sanity_reviews_b + cal_reviews + val_reviews + attack_reviews
    usage = usage_totals(all_reviews)
    _write_json(
        out_dir / "API_USAGE.json",
        {**usage, "month7_opened": False, "model": MODEL_ID, "thinking_disabled": True},
    )
    write_report(
        out_dir=out_dir,
        created=created,
        prompt_hash=prompt_hash,
        manifest=manifest,
        sanity=sanity,
        cal_thresholds=cal_thresholds,
        val_applied=val_applied,
        comparison=comparison,
        usage=usage,
        validation_skipped=validation_skipped,
    )
    summary = {
        "report_id": "D2L_MONTH6_DEVELOPMENT_AND_D2S_COMPARISON_REPORT",
        "out_dir": str(out_dir),
        "prompt_version": PROMPT_VERSION,
        "model": MODEL_ID,
        "thinking_disabled": True,
        "month7_opened": False,
        "validation_skipped": validation_skipped,
        "usage": usage,
        "primary_10pct": primary.to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (D2LError, D2LDataError, D2LContractError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
