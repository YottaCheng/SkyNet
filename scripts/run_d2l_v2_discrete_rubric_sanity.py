#!/usr/bin/env python3
"""D2-L V2 discrete-rubric sanity. Not a final Month-6 operating point.

Induces an 8-dimension GOOD/UNCERTAIN/BAD rubric from 100 Month-6 legitimate
D1-PASS applications, freezes it, scores those 100 plus 20 D1-bypass
applications, and decides whether to proceed. Attack outcomes do not revise
the rubric. Month 7 is not opened. D1/D2-S are not modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parents[1]
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.paths import OUTPUTS_ROOT  # noqa: E402
from d2.data import DEFAULT_RAW_PATH  # noqa: E402
from d2l.application_view import (  # noqa: E402
    application_view,
    application_view_sha256,
    serialize_application_view,
)
from d2l.calibrate import sample_id_hash, sort_legitimate_frame  # noqa: E402
from d2l.client import D2LClient, D2LCompletion  # noqa: E402
from d2l.contract import FORBIDDEN_INPUT_KEYS, FORBIDDEN_PROMPT_SUBSTRINGS  # noqa: E402
from d2l.data import month6_legitimate_d1_pass_core  # noqa: E402
from d2l.errors import D2LError, D2LParseError, D2LTransportError  # noqa: E402
from d2l.isolation import assert_not_month7_path  # noqa: E402
from d2l.v2_contract import (  # noqa: E402
    ATTACK_CONDITIONS,
    ATTACK_PER_CONDITION,
    ATTACK_SAMPLE_N,
    DIMENSION_IDS,
    LEGITIMATE_SAMPLE_N,
    MAX_RETRIES,
    PROMPT_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    RUBRIC_MAX_TOKENS,
    SANITY_SEED,
    SCORE_BATCH_MAX_TOKENS,
    SENTINEL_COUNT,
)
from d2l.v2_parser import (  # noqa: E402
    assert_rubric_immutable,
    parse_batch_scores,
    parse_rubric,
)
from d2l.v2_prompt import rubric_messages, scoring_messages  # noqa: E402
from d2l.v2_score import (  # noqa: E402
    dimension_points,
    provisional_threshold,
    sanity_decision,
    score_summary,
    sentinel_stability,
    total_score,
)

OUT_PARENT = OUTPUTS_ROOT / "development" / "d2l"
ATTACK_DATASET = (
    OUT_PARENT
    / "input_dataset"
    / "month6_successful_d1_bypasses_pro_thinkoff_20260816T194310Z"
    / "successful_d1_bypasses.jsonl"
)
EXPECTED_LEGIT = 101422
EXPECTED_ATTACKS = {"A0": 26, "A1-Pro": 27, "A2": 37, "A3-Pro": 31}


def _fail(message: str) -> None:
    raise D2LError(f"FAIL CLOSED: {message}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Usage:
    def __init__(self) -> None:
        self.n_calls = 0
        self.n_parse_failures = 0
        self.n_transport_failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        self.models: set[str] = set()
        self.fingerprints: set[str] = set()

    def add(self, completion: D2LCompletion) -> None:
        self.n_calls += 1
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        self.total_tokens += completion.total_tokens
        self.cached_tokens += completion.cached_tokens
        self.reasoning_tokens += completion.reasoning_tokens
        self.cost_usd += completion.cost_usd
        self.models.add(completion.model)
        if completion.system_fingerprint:
            self.fingerprints.add(completion.system_fingerprint)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "n_parse_failures": self.n_parse_failures,
            "n_transport_failures": self.n_transport_failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "returned_models": sorted(self.models),
            "system_fingerprints": sorted(self.fingerprints),
        }


def _assert_view_clean(view: Mapping[str, Any]) -> None:
    blob = serialize_application_view(view)
    for key in FORBIDDEN_INPUT_KEYS:
        if f'"{key}"' in blob:
            _fail(f"Forbidden key {key!r} leaked into a V2 application view.")


def _assert_text_clean(text: str, *, where: str) -> None:
    lowered = text.lower()
    for token in FORBIDDEN_PROMPT_SUBSTRINGS:
        if token.lower() in lowered:
            _fail(f"{where} contains forbidden substring {token!r}.")


def sample_legitimate(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = sort_legitimate_frame(frame)
    if len(ordered) != EXPECTED_LEGIT:
        _fail(f"Legitimate D1-PASS n={len(ordered)} != {EXPECTED_LEGIT}")
    rng = np.random.default_rng(SANITY_SEED)
    idx = rng.choice(len(ordered), size=LEGITIMATE_SAMPLE_N, replace=False)
    sampled = ordered.iloc[idx].reset_index(drop=True)
    sampled.insert(0, "opaque_id", [f"L{i:04d}" for i in range(1, LEGITIMATE_SAMPLE_N + 1)])
    return sampled


def sample_attacks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in ATTACK_CONDITIONS}
    for record in records:
        condition = record["attack_metadata"]["condition_id"]
        if condition not in grouped:
            continue
        grouped[condition].append(record)
    rng = np.random.default_rng(SANITY_SEED)
    chosen: list[dict[str, Any]] = []
    for condition in ATTACK_CONDITIONS:
        rows = grouped[condition]
        if len(rows) != EXPECTED_ATTACKS[condition]:
            _fail(f"{condition} has {len(rows)} != {EXPECTED_ATTACKS[condition]}")
        rows = sorted(rows, key=lambda item: str(item["record_id"]))
        pick = rng.choice(len(rows), size=ATTACK_PER_CONDITION, replace=False)
        chosen.extend(rows[int(i)] for i in pick)
    order = rng.permutation(len(chosen))
    shuffled = [chosen[int(i)] for i in order]
    for i, record in enumerate(shuffled, start=1):
        record = dict(record)
        record["_opaque_id"] = f"A{i:04d}"
        shuffled[i - 1] = record
    if len(shuffled) != ATTACK_SAMPLE_N:
        _fail(f"Attack sample n={len(shuffled)} != {ATTACK_SAMPLE_N}")
    return shuffled


def load_attack_records() -> list[dict[str, Any]]:
    assert_not_month7_path(ATTACK_DATASET)
    records = [
        json.loads(line)
        for line in ATTACK_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 121:
        _fail(f"Attack dataset n={len(records)} != 121")
    return records


def views_from_legit(frame: pd.DataFrame) -> list[tuple[str, dict[str, Any], int]]:
    out: list[tuple[str, dict[str, Any], int]] = []
    for _, row in frame.iterrows():
        view = application_view(row.to_dict())
        _assert_view_clean(view)
        out.append((str(row["opaque_id"]), view, int(row["source_row_id"])))
    return out


def views_from_attacks(
    records: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], str, str]]:
    out: list[tuple[str, dict[str, Any], str, str]] = []
    for record in records:
        view = application_view(record)
        _assert_view_clean(view)
        meta = record["attack_metadata"]
        out.append(
            (
                str(record["_opaque_id"]),
                view,
                str(record["record_id"]),
                str(meta["condition_id"]),
            )
        )
    return out


def complete_json(
    client: D2LClient,
    usage: Usage,
    messages: Sequence[Mapping[str, str]],
    *,
    max_tokens: int,
    parse,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            completion = client.complete(
                messages,
                max_tokens=max_tokens,
                timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            )
        except D2LTransportError as exc:
            usage.n_transport_failures += 1
            last_error = exc
            time.sleep(min(8.0, 1.5 ** attempt))
            continue
        usage.add(completion)
        if completion.reasoning_tokens:
            _fail("Thinking was not disabled: reasoning_tokens > 0.")
        try:
            return parse(completion.text), completion
        except D2LParseError as exc:
            usage.n_parse_failures += 1
            last_error = exc
    _fail(f"LLM call failed after retries: {last_error}")


def score_batch(
    client: D2LClient,
    usage: Usage,
    rubric: Mapping[str, Any],
    items: list[tuple[str, Mapping[str, Any]]],
) -> dict[str, dict[str, str]]:
    if not items:
        return {}
    expected = [app_id for app_id, _view in items]
    messages = scoring_messages(rubric, items)
    _assert_text_clean(messages[0]["content"], where="scoring system prompt")
    try:
        parsed, _completion = complete_json(
            client,
            usage,
            messages,
            max_tokens=SCORE_BATCH_MAX_TOKENS,
            parse=lambda text: parse_batch_scores(text, expected),
        )
        return parsed
    except D2LError:
        if len(items) == 1:
            raise
        mid = len(items) // 2
        print(f"Batch of {len(items)} failed; splitting to {mid}+{len(items)-mid}.", flush=True)
        left = score_batch(client, usage, rubric, items[:mid])
        right = score_batch(client, usage, rubric, items[mid:])
        left.update(right)
        return left


def make_batches(
    unique_items: list[tuple[str, dict[str, Any]]],
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]], list[str]]:
    if len(unique_items) != LEGITIMATE_SAMPLE_N:
        _fail("Legitimate unique items must be 100.")
    first = unique_items[:50]
    second = unique_items[50:]
    sentinel_ids = [
        unique_items[0][0],
        unique_items[1][0],
        unique_items[50][0],
        unique_items[51][0],
    ]
    if len(set(sentinel_ids)) != SENTINEL_COUNT:
        _fail("Sentinel IDs are not unique.")
    batch1 = list(first) + [unique_items[50], unique_items[51]]
    batch2 = list(second) + [unique_items[0], unique_items[1]]
    return batch1, batch2, sentinel_ids


def row_from_judgments(
    *,
    opaque_id: str,
    judgments: Mapping[str, str],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    points = dimension_points(judgments)
    row = {
        "opaque_id": opaque_id,
        "total_score": total_score(judgments),
        **{f"label_{dim_id}": judgments[dim_id] for dim_id in DIMENSION_IDS},
        **{f"points_{dim_id}": points[dim_id] for dim_id in DIMENSION_IDS},
        **dict(extra),
    }
    return row


def write_report(
    *,
    out_dir: Path,
    created: str,
    legit_manifest: Mapping[str, Any],
    attack_manifest: Mapping[str, Any],
    threshold_info: Mapping[str, Any],
    sentinel: Mapping[str, Any],
    decision: Mapping[str, Any],
    usage: Mapping[str, Any],
    legit_by_attacker_note: str,
    attack_review: Mapping[str, Any],
) -> None:
    conclusion = str(decision["conclusion"])
    next_step = {
        "FAIL_SCORE_COLLAPSE": "Stop D2-L development. The instrument is unusable: either the legitimate totals collapsed, the 10% threshold could not be realised, or sentinel/batch stability failed.",
        "FAIL_NO_PRELIMINARY_DISCRIMINATION": "Stop D2-L development. Legitimate scores vary, but the 20-attack sanity sample shows no preliminary upward shift.",
        "PASS_TO_LARGER_MONTH6_CALIBRATION": "Do not run the larger calibration automatically. If continuing, next step is a larger Month-6 legitimate calibration with this frozen rubric, still without using attack outcomes to revise it.",
    }[conclusion]
    attack_lines = [
        f"| {name} | {attack_review['by_condition'][name]['n']} | "
        f"{attack_review['by_condition'][name]['n_review']} | "
        f"{attack_review['by_condition'][name]['review_rate']:.3f} | "
        f"{attack_review['by_condition'][name]['median']:.1f} |"
        for name in ATTACK_CONDITIONS
    ]
    hist = decision["legitimate_summary"]["histogram"]
    hist_line = ", ".join(f"{k}:{v}" for k, v in hist.items() if int(v) > 0)
    report = f"""# D2L_V2_DISCRETE_RUBRIC_SANITY_REPORT

Created: {created}
Output: `{out_dir}`

## Conclusion

**{conclusion}**

{next_step}

This is sanity only. The provisional threshold is not a final Month-6 operating point.

## Protocol

- Prompt version: `{PROMPT_VERSION}`
- Model: `deepseek-v4-pro`
- Thinking: disabled
- Temperature: 0.0
- Seed: `{SANITY_SEED}`
- D1 numeric score in LLM input: false
- D2-S relationships in LLM input: false
- Attacker identity in LLM input: false
- Rubric revised after attacks: false
- Month 7 opened: false
- D1/D2-S modified: false
- D1-R / D2-HS used: false

Code mapping: GOOD=0, UNCERTAIN=1, BAD=2. `total_score` is the sum of eight dimensions (0-16). The LLM does not compute the total, CLEAR/REVIEW, or the threshold.

## Samples

Legitimate: N=100 Month-6 `fraud_bool=0` AND D1=PASS.
IDs SHA-256: `{legit_manifest['source_row_ids_sha256']}`

Attack: N=20 D1-bypass applications, 5 from each of A0 / A1-Pro ThinkOff / A2 / A3-Pro ThinkOff.
IDs SHA-256: `{attack_manifest['record_ids_sha256']}`
Attacker labels were joined only after scoring.

## Frozen rubric

Saved verbatim as `D2L_V2_FROZEN_RUBRIC.json`. Eight dimensions, GOOD/UNCERTAIN/BAD only. No LLM weights, total, or threshold.

## Legitimate scores (100 unique)

- Unique totals: {decision['legitimate_summary']['n_unique']}
- Range: {decision['legitimate_summary']['min']}–{decision['legitimate_summary']['max']}
- Quartiles: Q1={decision['legitimate_summary']['q1']}, median={decision['legitimate_summary']['median']}, Q3={decision['legitimate_summary']['q3']}
- Modal share: {decision['legitimate_summary']['modal_share']:.3f}
- Histogram (nonzero): {hist_line}

## Sentinel stability

- Sentinels: {sentinel['n_sentinels']}
- Identical 8-tuples: {sentinel['n_identical']}
- Materially different: {sentinel['n_materially_different']}
- Acceptable: {sentinel['acceptable']}

Duplicates were excluded from the 100-row score distribution.

## Provisional 10% threshold (sanity only)

- Threshold: {threshold_info['threshold']}
- Rule: `{threshold_info['decision_rule']}`
- Legitimate REVIEW: {threshold_info['n_review']}/{threshold_info['n_legitimate']} ({threshold_info['empirical_review_rate']:.4f})
- Tie size at threshold: {threshold_info['tie_size_at_threshold']}
- Final Month-6 operating point: false
- Random split of ties: false

## Attack sanity (scored blind, labels joined after)

- Attack REVIEW: {attack_review['n_review']}/{attack_review['n']} ({attack_review['review_rate']:.4f})
- Attack unique totals: {decision['attack_summary']['n_unique']}
- Attack median: {decision['attack_summary']['median']}
- Legitimate median: {decision['legitimate_summary']['median']}

| Attacker | N | REVIEW | Rate | Median total |
|---|---:|---:|---:|---:|
{chr(10).join(attack_lines)}
| POOLED | {attack_review['n']} | {attack_review['n_review']} | {attack_review['review_rate']:.3f} | {decision['attack_summary']['median']} |

{legit_by_attacker_note}

## API usage

- Calls: {usage['n_calls']}
- Parse failures: {usage['n_parse_failures']}
- Transport failures: {usage['n_transport_failures']}
- Prompt tokens: {usage['prompt_tokens']}
- Completion tokens: {usage['completion_tokens']}
- Cached tokens: {usage['cached_tokens']}
- Reasoning tokens: {usage['reasoning_tokens']}
- Estimated USD: {usage['cost_usd']:.6f}
- Returned models: {", ".join(usage['returned_models']) or "none"}
- System fingerprints: {", ".join(usage['system_fingerprints']) or "not provided"}

## Isolation

- Month 7 opened = false
- Attack outcomes were not used to revise the rubric
"""
    (out_dir / "D2L_V2_SANITY_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    raw = DEFAULT_RAW_PATH
    assert_not_month7_path(raw)
    created = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_PARENT / f"month6_d2l_v2_discrete_rubric_sanity_{stamp}"
    if out_dir.exists():
        _fail(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    print("Loading Month-6 legitimate D1-PASS population...", flush=True)
    legit_pop = month6_legitimate_d1_pass_core(raw, verify_hash=True)
    if int(legit_pop["month"].isin([7]).sum()):
        _fail("Sealed-month rows present.")
    legit = sample_legitimate(legit_pop)
    attacks = sample_attacks(load_attack_records())

    legit_ids = [int(x) for x in legit["source_row_id"].tolist()]
    attack_ids = [str(x["record_id"]) for x in attacks]
    legit_manifest = {
        "seed": SANITY_SEED,
        "n": LEGITIMATE_SAMPLE_N,
        "filter": "month=6 AND fraud_bool=0 AND D1=PASS",
        "opaque_ids": legit["opaque_id"].tolist(),
        "source_row_ids": legit_ids,
        "source_row_ids_sha256": sample_id_hash(legit_ids),
        "month7_opened": False,
    }
    attack_manifest = {
        "seed": SANITY_SEED,
        "n": ATTACK_SAMPLE_N,
        "n_per_condition": ATTACK_PER_CONDITION,
        "conditions": list(ATTACK_CONDITIONS),
        "opaque_ids": [x["_opaque_id"] for x in attacks],
        "record_ids": attack_ids,
        "record_ids_sha256": _sha256_text(json.dumps(attack_ids, separators=(",", ":"))),
        "condition_by_opaque_id": {
            x["_opaque_id"]: x["attack_metadata"]["condition_id"] for x in attacks
        },
        "labels_withheld_from_llm": True,
        "month7_opened": False,
    }
    _write_json(out_dir / "LEGITIMATE_SAMPLE_MANIFEST.json", legit_manifest)
    _write_json(out_dir / "ATTACK_SAMPLE_MANIFEST.json", attack_manifest)

    legit_items = [(oid, view) for oid, view, _sid in views_from_legit(legit)]
    attack_triples = views_from_attacks(attacks)
    attack_items = [(oid, view) for oid, view, _rid, _cond in attack_triples]

    client = D2LClient()
    usage = Usage()

    print("Inducing rubric from 100 legitimate applications (one call)...", flush=True)
    rubric_parsed, rubric_completion = complete_json(
        client,
        usage,
        rubric_messages(legit_items),
        max_tokens=RUBRIC_MAX_TOKENS,
        parse=parse_rubric,
    )
    rubric = {
        **rubric_parsed,
        "prompt_version": PROMPT_VERSION,
        "model": "deepseek-v4-pro",
        "thinking_disabled": True,
        "temperature": 0.0,
        "induced_from": "100 Month-6 legitimate D1-PASS applications",
        "n_applications_for_induction": 100,
        "frozen": True,
        "raw_model": rubric_completion.model,
        "system_fingerprint": rubric_completion.system_fingerprint,
        "month7_opened": False,
    }
    assert_rubric_immutable(rubric)
    rubric_blob = json.dumps(rubric, indent=2, sort_keys=True, ensure_ascii=True)
    _assert_text_clean(rubric_blob, where="frozen rubric")
    (out_dir / "D2L_V2_FROZEN_RUBRIC.json").write_text(rubric_blob, encoding="utf-8")
    print("Rubric frozen.", flush=True)

    batch1, batch2, sentinel_ids = make_batches(legit_items)
    print(f"Scoring legitimate batch 1 ({len(batch1)} including sentinels)...", flush=True)
    scores1 = score_batch(client, usage, rubric, batch1)
    print(f"Scoring legitimate batch 2 ({len(batch2)} including sentinels)...", flush=True)
    scores2 = score_batch(client, usage, rubric, batch2)

    unique_scores: dict[str, dict[str, str]] = {}
    for app_id, _view in legit_items[:50]:
        unique_scores[app_id] = scores1[app_id]
    for app_id, _view in legit_items[50:]:
        unique_scores[app_id] = scores2[app_id]
    if len(unique_scores) != 100:
        _fail("Did not recover 100 unique legitimate scores.")

    sentinel = sentinel_stability(
        {app_id: scores1[app_id] for app_id in sentinel_ids if app_id in scores1},
        {app_id: scores2[app_id] for app_id in sentinel_ids if app_id in scores2},
    )
    _write_json(out_dir / "SENTINEL_STABILITY.json", sentinel)

    id_to_row = {str(row.opaque_id): int(row.source_row_id) for row in legit.itertuples()}
    legit_rows = []
    legit_totals = []
    for app_id, _view in legit_items:
        judgments = unique_scores[app_id]
        total = total_score(judgments)
        legit_totals.append(total)
        legit_rows.append(
            row_from_judgments(
                opaque_id=app_id,
                judgments=judgments,
                extra={
                    "source_row_id": id_to_row[app_id],
                    "view_sha256": application_view_sha256(_view),
                    "split": "unique",
                },
            )
        )
    pd.DataFrame(legit_rows).to_csv(out_dir / "LEGITIMATE_V2_SCORES.csv", index=False)

    threshold_info = provisional_threshold(legit_totals)
    threshold_info["prompt_version"] = PROMPT_VERSION
    threshold_info["legitimate_summary"] = score_summary(legit_totals)
    _write_json(out_dir / "PROVISIONAL_THRESHOLD.json", threshold_info)

    print("Scoring 20 attack applications in two batches of 10...", flush=True)
    attack_scores_map: dict[str, dict[str, str]] = {}
    attack_scores_map.update(score_batch(client, usage, rubric, attack_items[:10]))
    attack_scores_map.update(score_batch(client, usage, rubric, attack_items[10:]))

    cond_by_id = {oid: cond for oid, _view, _rid, cond in attack_triples}
    rid_by_id = {oid: rid for oid, _view, rid, _cond in attack_triples}
    attack_rows = []
    attack_totals = []
    t = int(threshold_info["threshold"])
    by_condition: dict[str, list[int]] = {name: [] for name in ATTACK_CONDITIONS}
    for oid, view in attack_items:
        judgments = attack_scores_map[oid]
        total = total_score(judgments)
        attack_totals.append(total)
        decision_label = "REVIEW" if total >= t else "CLEAR"
        condition = cond_by_id[oid]
        by_condition[condition].append(total)
        attack_rows.append(
            row_from_judgments(
                opaque_id=oid,
                judgments=judgments,
                extra={
                    "record_id": rid_by_id[oid],
                    "condition_id": condition,
                    "view_sha256": application_view_sha256(view),
                    "decision": decision_label,
                },
            )
        )
    pd.DataFrame(attack_rows).to_csv(out_dir / "ATTACK_V2_SCORES.csv", index=False)

    attack_review = {
        "n": len(attack_totals),
        "n_review": int(sum(1 for s in attack_totals if s >= t)),
        "review_rate": float(np.mean([s >= t for s in attack_totals])),
        "by_condition": {
            name: {
                "n": len(vals),
                "n_review": int(sum(1 for s in vals if s >= t)),
                "review_rate": float(np.mean([s >= t for s in vals])) if vals else float("nan"),
                "median": float(np.median(vals)) if vals else float("nan"),
            }
            for name, vals in by_condition.items()
        },
    }
    decision = sanity_decision(
        legit_scores=legit_totals,
        attack_scores=attack_totals,
        threshold_info=threshold_info,
        sentinel=sentinel,
    )
    usage_dict = usage.as_dict()
    _write_json(
        out_dir / "API_USAGE.json",
        {**usage_dict, "thinking_disabled": True, "temperature": 0.0, "month7_opened": False},
    )
    _write_json(out_dir / "SANITY_DECISION.json", decision)
    write_report(
        out_dir=out_dir,
        created=created,
        legit_manifest=legit_manifest,
        attack_manifest=attack_manifest,
        threshold_info=threshold_info,
        sentinel=sentinel,
        decision=decision,
        usage=usage_dict,
        legit_by_attacker_note=(
            "Attacker-specific REVIEW counts were computed in code after scoring. "
            "They were not available to the LLM."
        ),
        attack_review=attack_review,
    )
    print(json.dumps({"conclusion": decision["conclusion"], "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except D2LError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
