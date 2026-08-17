#!/usr/bin/env python3
"""Build the primary Pro/ThinkOff view and the D2-L input dataset.

Reads the complete development N=50 benchmark.  Does not modify that
benchmark, the archive copy, or Month 7.  Does not call any LLM/API.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.benchmark_pins import (  # noqa: E402
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
)
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, OUTPUTS_ROOT  # noqa: E402
from baf_data.config import FROZEN_CONFIG  # noqa: E402

SOURCE_BENCHMARK = (
    OUTPUTS_ROOT
    / "development"
    / "attacker_benchmark"
    / "dev_model_selection_benchmark_m2_Q5_seed1_20260812T232252Z"
)
ARCHIVE_BENCHMARK = (
    OUTPUTS_ROOT
    / "archive"
    / "smoke"
    / "dev_model_selection_benchmark_m2_Q5_seed1_20260812T232252Z"
)
PRIMARY_VIEW_DIR = (
    OUTPUTS_ROOT / "development" / "attacker_benchmark" / "primary_pro_thinkoff"
)
EXPERIMENTS_DIR = OUTPUTS_ROOT / "experiments"
D2L_PARENT = OUTPUTS_ROOT / "development" / "d2l" / "input_dataset"

PRIMARY_CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("A0", "A0", "a0"),
    ("A1-Pro", "A1-Pro", "a1"),
    ("A2", "A2", "a2"),
    ("A3-Pro", "A3-Pro", "a3"),
)
EXPECTED_SUCCESS = {"A0": 26, "A1-Pro": 27, "A2": 37, "A3-Pro": 31}
EXPECTED_TOTAL = 121
FEATURE_COLUMNS = FROZEN_CONFIG.feature_columns


class BuildError(RuntimeError):
    """Fail-closed builder error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_not_month7(path: Path) -> None:
    text = str(path).lower()
    if "month7" in text or "month_7" in text:
        raise BuildError(f"Refusing a Month-7 path: {path}")


def _assert_untouched(path: Path, expected_sha256: str) -> None:
    got = _sha256_file(path)
    if got != expected_sha256:
        raise BuildError(f"Protected file changed: {path}")


def first_successful_step(episode: Mapping[str, Any]) -> dict[str, Any]:
    for step in episode.get("steps") or []:
        defence = step.get("internal_defence") or {}
        validity = step.get("validity") or {}
        if defence.get("decision") != "PASS":
            continue
        if validity.get("is_valid") is not True:
            continue
        features = validity.get("candidate_features")
        if not isinstance(features, Mapping):
            continue
        return dict(step)
    raise BuildError("Successful episode has no valid D1-PASS step.")


def application_from_step(step: Mapping[str, Any]) -> dict[str, Any]:
    features = (step.get("validity") or {}).get("candidate_features") or {}
    missing = [name for name in FEATURE_COLUMNS if name not in features]
    if missing:
        raise BuildError(f"Submitted application missing fields: {missing}")
    return {name: features[name] for name in FEATURE_COLUMNS}


def features_for_d1_score(application: Mapping[str, Any]) -> dict[str, Any]:
    """JSON null must become NaN so the frozen imputer matches in-memory scoring."""
    return {
        name: (np.nan if application[name] is None else application[name])
        for name in FEATURE_COLUMNS
    }


def thinking_and_model(episode_dir: Path, condition_id: str, episode: Mapping[str, Any]) -> dict[str, Any]:
    if condition_id == "A0":
        return {
            "attacker_kind": "a0",
            "model": None,
            "prompt_version": None,
            "gower_policy": None,
            "thinking_disabled": None,
            "thinking_label": "n/a",
        }
    if condition_id == "A2":
        return {
            "attacker_kind": "a2",
            "model": None,
            "prompt_version": None,
            "gower_policy": PINNED_A2_GOWER_POLICY,
            "thinking_disabled": None,
            "thinking_label": "n/a",
        }
    cfg_path = episode_dir / "model_config.json"
    payload: dict[str, Any] = {}
    if cfg_path.is_file():
        payload = _json_load(cfg_path)
    else:
        query_cfgs = sorted(episode_dir.glob("query_*/model_config.json"))
        if query_cfgs:
            payload = _json_load(query_cfgs[0])
        else:
            meta = ((episode.get("steps") or [{}])[0].get("research_meta") or {})
            payload = {
                "model": meta.get("llm_model"),
                "prompt_version": meta.get("prompt_version"),
                "thinking_disabled": meta.get("thinking_disabled"),
            }
    thinking_disabled = payload.get("thinking_disabled")
    model = payload.get("model") or payload.get("llm_model")
    prompt_version = payload.get("prompt_version")
    if condition_id == "A1-Pro":
        if model != MODEL_PRO:
            raise BuildError(f"{episode_dir}: A1-Pro model {model!r} != {MODEL_PRO!r}")
        if prompt_version != PINNED_A1_PROMPT_VERSION:
            raise BuildError(f"{episode_dir}: A1-Pro prompt {prompt_version!r}")
        if thinking_disabled is not True:
            raise BuildError(f"{episode_dir}: A1-Pro thinking_disabled={thinking_disabled!r}")
        return {
            "attacker_kind": "a1",
            "model": MODEL_PRO,
            "prompt_version": PINNED_A1_PROMPT_VERSION,
            "gower_policy": None,
            "thinking_disabled": True,
            "thinking_label": "OFF",
        }
    if condition_id == "A3-Pro":
        if model != MODEL_PRO:
            raise BuildError(f"{episode_dir}: A3-Pro model {model!r} != {MODEL_PRO!r}")
        if prompt_version != PINNED_A3_PROMPT_VERSION:
            raise BuildError(f"{episode_dir}: A3-Pro prompt {prompt_version!r}")
        if thinking_disabled is not True:
            raise BuildError(f"{episode_dir}: A3-Pro thinking_disabled={thinking_disabled!r}")
        return {
            "attacker_kind": "a3",
            "model": MODEL_PRO,
            "prompt_version": PINNED_A3_PROMPT_VERSION,
            "gower_policy": None,
            "thinking_disabled": True,
            "thinking_label": "OFF",
        }
    raise BuildError(f"Unexpected condition {condition_id}")


def write_primary_view(source_sha256: str) -> dict[str, Any]:
    if PRIMARY_VIEW_DIR.exists():
        existing = PRIMARY_VIEW_DIR / "PRIMARY_VIEW.json"
        if not existing.is_file():
            raise BuildError(f"Primary view dir exists but is incomplete: {PRIMARY_VIEW_DIR}")
        return _json_load(existing)
    PRIMARY_VIEW_DIR.mkdir(parents=True, exist_ok=False)
    run_config = _json_load(SOURCE_BENCHMARK / "run_config.json")
    if run_config.get("month7_opened") is not False:
        raise BuildError("Source benchmark reports month7_opened != false.")
    if str(run_config.get("d1_artefact_id")) != PINNED_D1_ARTEFACT_ID:
        raise BuildError("Source D1 artefact id mismatch.")

    index_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for condition_id, directory, _kind in PRIMARY_CONDITIONS:
        cond_dir = SOURCE_BENCHMARK / "benchmark" / directory
        summary = _json_load(cond_dir / "condition_summary.json")
        if summary.get("month7_opened") is not False:
            raise BuildError(f"{condition_id} month7_opened != false.")
        n_success = 0
        n_ep = 0
        for episode_path in sorted(cond_dir.glob("anchor_*/seed_*/episode_result.json")):
            n_ep += 1
            episode = _json_load(episode_path)
            success = bool(episode.get("success") is True)
            if success:
                n_success += 1
            thinking_and_model(episode_path.parent, condition_id, episode)
            rel = episode_path.relative_to(SOURCE_BENCHMARK).as_posix()
            index_rows.append(
                {
                    "condition_id": condition_id,
                    "anchor_id": str(episode.get("case_id") or episode_path.parent.parent.name.replace("anchor_", "")),
                    "seed": int(episode_path.parent.name.split("_")[-1]),
                    "success": success,
                    "source_episode_relpath": rel,
                    "source_episode_uri": f"source_benchmark://{rel}",
                }
            )
        if n_ep != 50:
            raise BuildError(f"{condition_id} episode count {n_ep} != 50")
        if n_success != EXPECTED_SUCCESS[condition_id]:
            raise BuildError(
                f"{condition_id} success count {n_success} != {EXPECTED_SUCCESS[condition_id]}"
            )
        condition_rows.append(
            {
                "condition_id": condition_id,
                "source_condition_dir": f"source_benchmark://benchmark/{directory}",
                "source_condition_summary": f"source_benchmark://benchmark/{directory}/condition_summary.json",
                "n_episodes": n_ep,
                "n_success": n_success,
                "asr_at_5": summary.get("asr_curve", {}).get("ASR@5"),
                "attacker_kind": summary.get("attacker_kind"),
                "llm_model": summary.get("llm_model"),
                "prompt_version": summary.get("prompt_version"),
                "gower_policy": summary.get("gower_policy"),
            }
        )

    excluded = ["A1-Flash", "A3-Flash"]
    for name in excluded:
        if not (SOURCE_BENCHMARK / "benchmark" / name).is_dir():
            raise BuildError(f"Source benchmark missing {name}; Flash conditions must remain in the full copy.")

    manifest = {
        "view_id": "primary_pro_thinkoff",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(SOURCE_BENCHMARK),
        "source_run_config_sha256": source_sha256,
        "lightweight": True,
        "duplicates_source_episodes": False,
        "mutates_source_benchmark": False,
        "selected_conditions": [
            {
                "condition_id": "A0",
                "requirement": "frozen random",
            },
            {
                "condition_id": "A1-Pro",
                "requirement": f"V4.3, {MODEL_PRO}, Thinking OFF",
                "prompt_version": PINNED_A1_PROMPT_VERSION,
            },
            {
                "condition_id": "A2",
                "requirement": "Gower v2",
                "gower_policy": PINNED_A2_GOWER_POLICY,
            },
            {
                "condition_id": "A3-Pro",
                "requirement": f"V2.3, {MODEL_PRO}, Thinking OFF",
                "prompt_version": PINNED_A3_PROMPT_VERSION,
            },
        ],
        "retained_in_source_only": excluded,
        "n_success_expected": EXPECTED_SUCCESS,
        "n_success_observed": {row["condition_id"]: row["n_success"] for row in condition_rows},
        "month7_opened": False,
        "conditions": condition_rows,
    }
    (PRIMARY_VIEW_DIR / "PRIMARY_VIEW.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (PRIMARY_VIEW_DIR / "episode_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    return manifest


def clean_empty_experiments() -> dict[str, Any]:
    """Remove leftover empty experiments/ husks. Keep the write-root directory."""
    report = {"removed": [], "kept": [], "moved": [], "hardcoded_live_dependencies": []}
    live_refs = [
        "04_implementation/src/attack_lab/paths.py:EXPERIMENTS_ROOT (future stage='experiments' writes)",
        "04_implementation/scripts/run_month6_frozen_a0_a3.py:OUTPUT_PARENT (would recreate on a new run)",
    ]
    report["hardcoded_live_dependencies"] = live_refs
    if not EXPERIMENTS_DIR.exists():
        EXPERIMENTS_DIR.mkdir(parents=True)
        (EXPERIMENTS_DIR / ".gitkeep").write_text("", encoding="utf-8")
        report["kept"].append(str(EXPERIMENTS_DIR))
        return report

    junk_names = {".DS_Store", ".gitkeep"}
    # Collect dirs deepest-first.
    dirs = sorted(
        [p for p in EXPERIMENTS_DIR.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for directory in dirs:
        if directory == EXPERIMENTS_DIR:
            continue
        children = list(directory.iterdir())
        if not children:
            directory.rmdir()
            report["removed"].append(str(directory))
            continue
        if all(child.is_file() and child.name in junk_names for child in children):
            for child in children:
                child.unlink()
            directory.rmdir()
            report["removed"].append(str(directory))
            continue
        # Non-empty research leftover would be moved, not deleted.
        research = [c for c in children if not (c.is_file() and c.name in junk_names)]
        if research:
            raise BuildError(
                "Non-empty experiments leftover requires archive move, not delete: "
                f"{directory} -> {research}"
            )
    # Recreate a clean write root marker.
    keep = EXPERIMENTS_DIR / ".gitkeep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")
    report["kept"].append(str(EXPERIMENTS_DIR))
    return report


def build_dataset(defender: FrozenXGBoostDefender, stamp: str) -> dict[str, Any]:
    out_dir = D2L_PARENT / f"month6_successful_d1_bypasses_pro_thinkoff_{stamp}"
    if out_dir.exists():
        raise BuildError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    if defender.artefact_id != PINNED_D1_ARTEFACT_ID:
        raise BuildError(f"Loaded D1 {defender.artefact_id!r} != {PINNED_D1_ARTEFACT_ID!r}")

    records: list[dict[str, Any]] = []
    score_stats: dict[str, list[float]] = {cid: [] for cid, _, _ in PRIMARY_CONDITIONS}
    logged_present = 0
    logged_absent = 0
    max_abs_delta = 0.0

    run_config = _json_load(SOURCE_BENCHMARK / "run_config.json")
    m = int(run_config["m"])
    q_max = int(run_config["Q"])
    k = int(run_config["K"])

    for condition_id, directory, _kind in PRIMARY_CONDITIONS:
        cond_dir = SOURCE_BENCHMARK / "benchmark" / directory
        for episode_path in sorted(cond_dir.glob("anchor_*/seed_*/episode_result.json")):
            _assert_not_month7(episode_path)
            episode = _json_load(episode_path)
            if episode.get("success") is not True:
                continue
            step = first_successful_step(episode)
            application = application_from_step(step)
            pins = thinking_and_model(episode_path.parent, condition_id, episode)
            defence_logged = step.get("internal_defence") or {}
            logged_score = defence_logged.get("risk_score")
            logged_decision = defence_logged.get("decision")
            logged_threshold = defence_logged.get("threshold")
            logged_artefact = defence_logged.get("artefact_id")
            if logged_artefact not in (None, PINNED_D1_ARTEFACT_ID):
                raise BuildError(f"{episode_path}: logged artefact {logged_artefact!r}")

            recomputed = defender.score_application(features_for_d1_score(application))
            if recomputed.decision != "PASS":
                raise BuildError(
                    f"{episode_path}: recomputed decision {recomputed.decision} "
                    f"score={recomputed.risk_score}"
                )
            if recomputed.artefact_id != PINNED_D1_ARTEFACT_ID:
                raise BuildError("Recomputed artefact id mismatch.")

            if logged_score is None:
                logged_absent += 1
                delta = None
            else:
                logged_present += 1
                delta = abs(float(logged_score) - float(recomputed.risk_score))
                if delta > 1e-12:
                    raise BuildError(
                        f"{episode_path}: recomputed D1 score {recomputed.risk_score} "
                        f"!= logged {logged_score} (abs delta {delta})"
                    )
                max_abs_delta = max(max_abs_delta, delta)

            successful_query = int(step.get("attempt") or (step.get("budget_event") or {}).get("q_used") or 0)
            anchor_id = str(episode.get("case_id") or episode_path.parent.parent.name.replace("anchor_", ""))
            seed = int(episode_path.parent.name.split("_")[-1])
            rel = episode_path.relative_to(SOURCE_BENCHMARK).as_posix()
            record_id = f"{condition_id}:{anchor_id}:seed_{seed}:q{successful_query}"
            record = {
                "record_id": record_id,
                "application": application,
                "attack_metadata": {
                    "anchor_id": anchor_id,
                    "condition_id": condition_id,
                    "successful_query": successful_query,
                    "attacker_kind": pins["attacker_kind"],
                    "model": pins["model"],
                    "prompt_version": pins["prompt_version"],
                    "gower_policy": pins["gower_policy"],
                    "thinking_disabled": pins["thinking_disabled"],
                    "thinking_label": pins["thinking_label"],
                    "m": m,
                    "Q": q_max,
                    "K": k,
                    "seed": seed,
                    "source_episode_id": f"{directory}/anchor_{anchor_id}/seed_{seed}",
                    "source_episode_relpath": rel,
                },
                "defender_metadata": {
                    "d1_artefact_id": PINNED_D1_ARTEFACT_ID,
                    "d1_score": float(recomputed.risk_score),
                    "d1_threshold": float(recomputed.threshold),
                    "d1_decision": recomputed.decision,
                    "d1_score_logged": None if logged_score is None else float(logged_score),
                    "d1_threshold_logged": None if logged_threshold is None else float(logged_threshold),
                    "d1_decision_logged": logged_decision,
                    "d1_score_recomputed": True,
                    "d1_score_logged_existed": logged_score is not None,
                    "abs_score_delta_vs_log": delta,
                },
            }
            records.append(record)
            score_stats[condition_id].append(float(recomputed.risk_score))

    counts = {cid: len(score_stats[cid]) for cid, _, _ in PRIMARY_CONDITIONS}
    if counts != EXPECTED_SUCCESS:
        raise BuildError(f"Success counts {counts} != expected {EXPECTED_SUCCESS}")
    if len(records) != EXPECTED_TOTAL:
        raise BuildError(f"Total {len(records)} != {EXPECTED_TOTAL}")
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise BuildError("Duplicate record_id values.")
    if any(r["defender_metadata"]["d1_decision"] != "PASS" for r in records):
        raise BuildError("Non-PASS recomputed decision present.")

    def summarise(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype="float64")
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }

    by_attacker = {cid: summarise(vals) for cid, vals in score_stats.items()}

    jsonl_path = out_dir / "successful_d1_bypasses.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    csv_path = out_dir / "successful_d1_bypasses.csv"
    fieldnames = (
        [
            "record_id",
            "condition_id",
            "anchor_id",
            "successful_query",
            "attacker_kind",
            "model",
            "prompt_version",
            "gower_policy",
            "thinking_label",
            "m",
            "Q",
            "K",
            "source_episode_id",
            "d1_score",
            "d1_threshold",
            "d1_decision",
            "d1_score_logged",
            "d1_score_logged_existed",
        ]
        + [f"application__{name}" for name in FEATURE_COLUMNS]
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "record_id": record["record_id"],
                **{k: record["attack_metadata"][k] for k in (
                    "condition_id",
                    "anchor_id",
                    "successful_query",
                    "attacker_kind",
                    "model",
                    "prompt_version",
                    "gower_policy",
                    "thinking_label",
                    "m",
                    "Q",
                    "K",
                    "source_episode_id",
                )},
                "d1_score": record["defender_metadata"]["d1_score"],
                "d1_threshold": record["defender_metadata"]["d1_threshold"],
                "d1_decision": record["defender_metadata"]["d1_decision"],
                "d1_score_logged": record["defender_metadata"]["d1_score_logged"],
                "d1_score_logged_existed": record["defender_metadata"]["d1_score_logged_existed"],
            }
            for name in FEATURE_COLUMNS:
                row[f"application__{name}"] = record["application"][name]
            writer.writerow(row)

    schema = {
        "record_id": "Stable id: condition:anchor:seed:qN",
        "application": {
            "description": "Exact submitted application fields from the first valid D1-PASS attempt.",
            "fields": list(FEATURE_COLUMNS),
            "source": "episode_result.steps[].validity.candidate_features",
        },
        "attack_metadata": {
            "anchor_id": "Month-6 starting-case id",
            "condition_id": "A0 | A1-Pro | A2 | A3-Pro",
            "successful_query": "Query index of the first D1-PASS (1-based attempt)",
            "attacker_kind": "a0 | a1 | a2 | a3",
            "model": "LLM model id, or null for A0/A2",
            "prompt_version": "Pinned prompt id, or null for A0/A2",
            "gower_policy": "Pinned A2 policy, else null",
            "thinking_disabled": "true for A1-Pro/A3-Pro; null otherwise",
            "m": "Edit budget",
            "Q": "Query budget",
            "K": "Reference-pool size",
            "source_episode_id": "Pointer into the development benchmark copy",
        },
        "defender_metadata": {
            "d1_score": "Offline recomputed frozen-D1 risk score",
            "d1_threshold": "Frozen Month-6 threshold",
            "d1_decision": "PASS iff d1_score < d1_threshold",
            "d1_score_logged": "risk_score from the source episode log, if present",
            "note": "Stored for later LLM-input views. Not yet exposed or hidden from an LLM.",
        },
        "not_included": [
            "D2-S score",
            "LLM-D2 judgement",
            "attacker prompts / raw LLM completions",
            "A1-Flash / A3-Flash episodes",
            "Month 7",
        ],
    }
    (out_dir / "FIELD_SCHEMA.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "dataset_id": out_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(SOURCE_BENCHMARK),
        "archive_benchmark_untouched": str(ARCHIVE_BENCHMARK),
        "d1_artefact_id": PINNED_D1_ARTEFACT_ID,
        "d1_artefact_dir": str(DEFAULT_C1_ARTEFACT_DIR),
        "d1_threshold": float(defender.threshold),
        "month7_opened": False,
        "llm_api_called": False,
        "d2s_scored": False,
        "llm_d2_called": False,
        "conditions": list(EXPECTED_SUCCESS),
        "n_by_condition": counts,
        "n_total": len(records),
        "all_recomputed_pass": True,
        "numeric_d1_score_existed_in_source_logs": logged_present == len(records),
        "n_logged_scores": logged_present,
        "n_missing_logged_scores": logged_absent,
        "d1_score_recomputed": True,
        "max_abs_delta_recomputed_vs_logged": max_abs_delta,
        "d1_score_by_attacker": by_attacker,
        "files": {
            "jsonl": jsonl_path.name,
            "csv": csv_path.name,
            "schema": "FIELD_SCHEMA.json",
            "report": "BUILD_REPORT.md",
        },
    }
    (out_dir / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    def fmt_stats(block: dict[str, float]) -> str:
        return (
            f"n={block['n']} min={block['min']:.8f} "
            f"median={block['median']:.8f} max={block['max']:.8f}"
        )

    report = f"""# D2-L input dataset build report

Created: {manifest['created_utc']}
Dataset: `{out_dir}`

## Counts

| Condition | Expected | Observed |
|---|---:|---:|
| A0 | 26 | {counts['A0']} |
| A1-Pro ThinkOff | 27 | {counts['A1-Pro']} |
| A2 | 37 | {counts['A2']} |
| A3-Pro ThinkOff | 31 | {counts['A3-Pro']} |
| **Total** | **121** | **{len(records)}** |

Fail-closed count check: passed.

## D1 scores

- Numeric D1 `risk_score` **already existed** in source episode logs for **{logged_present}/121** records.
- Missing logged scores: {logged_absent}.
- All 121 scores were **offline recomputed** with frozen D1 `{PINNED_D1_ARTEFACT_ID}`.
- JSON `null` application fields were converted to `NaN` **only for recompute**, matching the in-memory attack-lab scoring path. Stored application fields keep the exact logged values.
- All 121 recomputed decisions: **PASS**.
- Frozen threshold: `{defender.threshold}`.
- Max |recomputed − logged| = {max_abs_delta:.8e} (fail-closed if > 1e-12).

### Recomputed D1 score by attacker

| Condition | Summary |
|---|---|
| A0 | {fmt_stats(by_attacker['A0'])} |
| A1-Pro | {fmt_stats(by_attacker['A1-Pro'])} |
| A2 | {fmt_stats(by_attacker['A2'])} |
| A3-Pro | {fmt_stats(by_attacker['A3-Pro'])} |

## Isolation

- Source development benchmark was not modified.
- Archive benchmark copy was not modified.
- A1-Flash and A3-Flash were not included.
- D2-S was not scored.
- LLM-D2 was not called.
- No LLM/API call was made.
- Month 7 remained sealed (`month7_opened=false`).
"""
    (out_dir / "BUILD_REPORT.md").write_text(report, encoding="utf-8")
    return {"out_dir": str(out_dir), "manifest": manifest, "report": report}


def main() -> int:
    _assert_not_month7(SOURCE_BENCHMARK)
    _assert_not_month7(DEFAULT_C1_ARTEFACT_DIR)
    if not SOURCE_BENCHMARK.is_dir():
        raise BuildError(f"Source benchmark missing: {SOURCE_BENCHMARK}")
    if not ARCHIVE_BENCHMARK.is_dir():
        raise BuildError(f"Archive benchmark missing: {ARCHIVE_BENCHMARK}")

    source_run = SOURCE_BENCHMARK / "run_config.json"
    archive_run = ARCHIVE_BENCHMARK / "run_config.json"
    source_sha = _sha256_file(source_run)
    archive_sha = _sha256_file(archive_run)
    if source_sha != archive_sha:
        raise BuildError("Development and archive run_config.json hashes differ.")

    experiments_report = clean_empty_experiments()
    view = write_primary_view(source_sha)

    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset = build_dataset(defender, stamp)

    # Prove protected trees were not mutated.
    _assert_untouched(source_run, source_sha)
    _assert_untouched(archive_run, archive_sha)

    summary = {
        "primary_view": str(PRIMARY_VIEW_DIR),
        "dataset_dir": dataset["out_dir"],
        "n_total": dataset["manifest"]["n_total"],
        "n_by_condition": dataset["manifest"]["n_by_condition"],
        "all_recomputed_pass": True,
        "month7_opened": False,
        "experiments_cleanup": experiments_report,
        "source_benchmark_untouched": True,
        "archive_benchmark_untouched": True,
    }
    print(json.dumps(summary, indent=2))
    print(dataset["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
