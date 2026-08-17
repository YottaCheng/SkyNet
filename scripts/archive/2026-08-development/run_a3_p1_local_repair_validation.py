#!/usr/bin/env python3
"""Validate restored A3 P1 with up to three local proposals per real query.

This is fixed Month-6 development/correction evidence only.  It compares the
existing P1 single-candidate planner at max_local=3 against the rejected old
runner deviation at max_local=1 on the exact same 25 anchors.  It never opens
Month 7 and never freezes a configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "04_implementation" / "src"
ABLATION_SCRIPTS = (
    ROOT
    / "04_implementation"
    / "archive"
    / "2026-08-active-stack-cleanup"
    / "scripts"
)
for import_path in (SRC, ABLATION_SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from attack_lab.attackers.a3_agent import (  # noqa: E402
    FORMAL_A3_MODEL_CONFIG,
    PROMPT_VERSION_P1_COMPACT,
)
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory  # noqa: E402
from attack_lab.types import to_jsonable  # noqa: E402
from run_a3_prompt_ablation import (  # noqa: E402
    DEFAULT_EXPERIMENT_SEED,
    DEFAULT_RAW_PATH,
    EXPERIMENTS_ROOT,
    PREDEFINED_ENGINEERING_CRITERIA,
    _implementation_provenance,
    preflight_outbound_payloads,
    run_variant,
)

OLD_T0_PARENT = (
    EXPERIMENTS_ROOT
    / "a3"
    / "prompt_development"
    / "a3_prompt_temperature_ablation_n25_m2_q5_seed20260804_20260809T193343Z"
)
OLD_T0_RUN = (
    OLD_T0_PARENT
    / "condition_P1_t0_a3_episodic_p1_compact_v2_stable_prefix"
)
OLD_ANCHORS = OLD_T0_PARENT / "prompt_dev_anchors.json"
MAX_LOCAL = 3


def _engineering_gate(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    p = summary["primary_criteria"]
    novelty = p["E_post_block_projected_novelty_rate"]
    checks = {
        "projected_duplicates_zero": p[
            "A_projected_duplicate_submission_count"
        ]
        == 0,
        "exhaustion_zero": p["B_regeneration_exhaustion_rate"]
        <= PREDEFINED_ENGINEERING_CRITERIA[
            "local_generation_exhaustion_rate_max"
        ],
        "valid_submission_rate": p[
            "C_valid_submission_rate_over_attempted_query_records"
        ]
        >= PREDEFINED_ENGINEERING_CRITERIA["valid_submission_rate_min"],
        "no_leakage": p["D_forbidden_leakage_hits"] == 0,
        "env_step_equals_q": p["D_env_step_equals_q_used_all_anchors"],
        "no_parse_failures": p["F_parse_failure_events"] == 0,
        "no_invalid_submissions": p["C_invalid_submissions"] == 0,
        "post_block_novelty": novelty is None
        or novelty
        >= PREDEFINED_ENGINEERING_CRITERIA[
            "post_block_projected_novelty_rate_min"
        ],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, failures


def _normalise_max_local_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result["max_local_generation_attempts_per_query"] = "NORMALISED"
    return result


def _normalise_max_local_prompt(text: str) -> str:
    return text.replace(
        '"max_local_generation_attempts_per_query": 1',
        '"max_local_generation_attempts_per_query": "NORMALISED"',
    ).replace(
        '"max_local_generation_attempts_per_query": 3',
        '"max_local_generation_attempts_per_query": "NORMALISED"',
    )


def _first_candidate_audit(
    *, old_run: Path, new_run: Path, anchors: Sequence[str]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for anchor_id in anchors:
        old_anchor = old_run / f"anchor_{anchor_id}"
        new_anchor = new_run / f"anchor_{anchor_id}"
        old_queries = {
            path.name: path for path in sorted(old_anchor.glob("query_*"))
        }
        new_queries = {
            path.name: path for path in sorted(new_anchor.glob("query_*"))
        }
        for query_name in sorted(set(old_queries) & set(new_queries)):
            old_gen = old_queries[query_name] / "local_gen_01"
            new_gen = new_queries[query_name] / "local_gen_01"
            old_candidate_path = old_gen / "a3_parsed_candidate.json"
            new_candidate_path = new_gen / "a3_parsed_candidate.json"
            old_payload_path = old_gen / "a3_prompt_payload.json"
            new_payload_path = new_gen / "a3_prompt_payload.json"
            old_prompt_path = old_gen / "a3_prompt_full.txt"
            new_prompt_path = new_gen / "a3_prompt_full.txt"
            old_hash_path = old_gen / "a3_prompt_hash.txt"
            new_hash_path = new_gen / "a3_prompt_hash.txt"
            if not all(
                path.exists()
                for path in (
                    old_payload_path,
                    new_payload_path,
                    old_prompt_path,
                    new_prompt_path,
                    old_hash_path,
                    new_hash_path,
                )
            ):
                continue
            old_payload = json.loads(old_payload_path.read_text(encoding="utf-8"))
            new_payload = json.loads(new_payload_path.read_text(encoding="utf-8"))
            old_candidate = (
                json.loads(old_candidate_path.read_text(encoding="utf-8"))
                if old_candidate_path.exists()
                else None
            )
            new_candidate = (
                json.loads(new_candidate_path.read_text(encoding="utf-8"))
                if new_candidate_path.exists()
                else None
            )
            old_prompt = old_prompt_path.read_text(encoding="utf-8")
            new_prompt = new_prompt_path.read_text(encoding="utf-8")
            normalised_payload_equal = (
                _normalise_max_local_payload(old_payload)
                == _normalise_max_local_payload(new_payload)
            )
            normalised_prompt_equal = (
                _normalise_max_local_prompt(old_prompt)
                == _normalise_max_local_prompt(new_prompt)
            )
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "query": query_name,
                    "raw_prompt_hash_equal": (
                        old_hash_path.read_text(encoding="utf-8").strip()
                        == new_hash_path.read_text(encoding="utf-8").strip()
                    ),
                    "prompt_equal_after_max_local_normalisation": (
                        normalised_prompt_equal
                    ),
                    "payload_equal_after_max_local_normalisation": (
                        normalised_payload_equal
                    ),
                    "first_candidate_available_both": (
                        old_candidate is not None and new_candidate is not None
                    ),
                    "first_candidate_exact_equal": (
                        old_candidate == new_candidate
                        if old_candidate is not None and new_candidate is not None
                        else None
                    ),
                    "first_candidate_changes_equal": (
                        old_candidate.get("changes") == new_candidate.get("changes")
                        if old_candidate is not None and new_candidate is not None
                        else None
                    ),
                    "old_first_strategy": (
                        old_candidate.get("strategy_label") if old_candidate else None
                    ),
                    "new_first_strategy": (
                        new_candidate.get("strategy_label") if new_candidate else None
                    ),
                }
            )
    available = [row for row in rows if row["first_candidate_available_both"]]
    same_scientific_inputs = [
        row
        for row in available
        if row["payload_equal_after_max_local_normalisation"]
        and row["prompt_equal_after_max_local_normalisation"]
    ]
    return {
        "note": (
            "Raw prompt hashes are expected to differ because the visible frozen "
            "max_local field changes from 1 to 3. Candidate differences under "
            "otherwise normalised-equal inputs may arise from that explicit prompt "
            "field and/or external API nondeterminism; they are not automatically "
            "classified as code semantic changes. Later-query inputs can also differ "
            "because earlier generated candidates changed the public trajectory."
        ),
        "n_common_queries": len(rows),
        "raw_prompt_hash_match_count": sum(
            bool(row["raw_prompt_hash_equal"]) for row in rows
        ),
        "normalised_prompt_match_count": sum(
            bool(row["prompt_equal_after_max_local_normalisation"]) for row in rows
        ),
        "normalised_payload_match_count": sum(
            bool(row["payload_equal_after_max_local_normalisation"]) for row in rows
        ),
        "n_first_candidates_available_both": len(available),
        "first_candidate_exact_match_count": sum(
            bool(row["first_candidate_exact_equal"]) for row in available
        ),
        "first_candidate_changes_match_count": sum(
            bool(row["first_candidate_changes_equal"]) for row in available
        ),
        "n_normalised_equal_scientific_inputs": len(same_scientific_inputs),
        "first_candidate_exact_matches_under_normalised_equal_inputs": sum(
            bool(row["first_candidate_exact_equal"])
            for row in same_scientific_inputs
        ),
        "rows": rows,
    }


def _repair_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    queries = [
        step
        for anchor in summary["per_anchor"]
        for step in anchor.get("trajectory", [])
    ]
    reason_counts: Counter[str] = Counter()
    for query in queries:
        for local in query.get("local_generation_records", []):
            reason = local.get("local_rejection_reason")
            if reason:
                reason_counts[str(reason)] += 1
    first_legal = [q for q in queries if q.get("local_generation_attempts") == 1]
    second_or_later = [q for q in queries if q.get("local_generation_attempts", 0) >= 2]
    third = [q for q in queries if q.get("local_generation_attempts", 0) >= 3]
    repaired_submissions = [q for q in second_or_later if q.get("submitted")]
    exhausted = [q for q in queries if q.get("regeneration_exhausted")]
    return {
        "n_real_query_records": len(queries),
        "n_first_candidate_legal_and_submitted": len(first_legal),
        "n_triggered_second_generation": len(second_or_later),
        "n_triggered_third_generation": len(third),
        "local_rejection_reason_counts": dict(reason_counts),
        "n_repaired_to_legal_submission": len(repaired_submissions),
        "repaired_submission_public_outcomes": dict(
            Counter(str(q.get("public_label")) for q in repaired_submissions)
        ),
        "n_three_attempt_exhaustions": len(exhausted),
        "exhaustion_locations": [
            {
                "anchor_id": anchor["anchor_id"],
                "query_index": query["query_index"],
            }
            for anchor in summary["per_anchor"]
            for query in anchor.get("trajectory", [])
            if query.get("regeneration_exhausted")
        ],
    }


def _paired_comparison(
    *, old: Mapping[str, Any], new: Mapping[str, Any], anchors: Sequence[str]
) -> dict[str, Any]:
    old_by = {str(row["anchor_id"]): row for row in old["per_anchor"]}
    new_by = {str(row["anchor_id"]): row for row in new["per_anchor"]}
    if list(old_by) != list(anchors) or list(new_by) != list(anchors):
        raise RuntimeError("Old/new anchor order differs from the frozen 25 set.")
    rows = []
    for anchor_id in anchors:
        before, after = old_by[anchor_id], new_by[anchor_id]
        rows.append(
            {
                "anchor_id": anchor_id,
                "old_success": bool(before["success"]),
                "new_success": bool(after["success"]),
                "transition": f"{bool(before['success'])}->{bool(after['success'])}",
                "old_stop_reason": before["stop_reason"],
                "new_stop_reason": after["stop_reason"],
                "old_queries_used": before["queries_used"],
                "new_queries_used": after["queries_used"],
                "old_attempts_to_success": before["attempts_to_success"],
                "new_attempts_to_success": after["attempts_to_success"],
            }
        )
    gate_ok, gate_failures = _engineering_gate(new)
    old_curve, new_curve = old["asr_curve"], new["asr_curve"]
    return {
        "status": "month6_a3_p1_local_repair_development_only",
        "decision_status": "candidate for supervisor audit",
        "exact_anchor_order_match": True,
        "asr_curve_old_maxlocal1": old_curve,
        "asr_curve_new_maxlocal3": new_curve,
        "asr_curve_delta": {
            key: float(new_curve[key]) - float(old_curve[key]) for key in old_curve
        },
        "asr5_delta": float(new_curve["ASR@5"]) - float(old_curve["ASR@5"]),
        "auc_old": old["asr_curve_auc_mean"],
        "auc_new": new["asr_curve_auc_mean"],
        "n_success_old": old["n_success"],
        "n_success_new": new["n_success"],
        "success_transition_counts": dict(Counter(row["transition"] for row in rows)),
        "exhaustions_old": old["regeneration_exhaustions"],
        "exhaustions_new": new["regeneration_exhaustions"],
        "valid_submission_rate_old": old["primary_criteria"][
            "C_valid_submission_rate_over_attempted_query_records"
        ],
        "valid_submission_rate_new": new["primary_criteria"][
            "C_valid_submission_rate_over_attempted_query_records"
        ],
        "mean_queries_to_success_old": old["mean_queries_to_success"],
        "mean_queries_to_success_new": new["mean_queries_to_success"],
        "stop_reason_counts_old": old["stop_reason_counts"],
        "stop_reason_counts_new": new["stop_reason_counts"],
        "llm_calls_old": old["total_llm_calls"],
        "llm_calls_new": new["total_llm_calls"],
        "token_usage_old": old["token_usage"],
        "token_usage_new": new["token_usage"],
        "cost_old_usd": old["total_estimated_cost_usd"],
        "cost_new_usd": new["total_estimated_cost_usd"],
        "meets_existing_engineering_thresholds": gate_ok,
        "failed_engineering_thresholds": gate_failures,
        "known_old_exhaustion_cases": [
            row for row in rows if row["anchor_id"] in {"842574", "867193", "892990"}
        ],
        "per_anchor": rows,
    }


def _write_report(parent: Path, result: Mapping[str, Any]) -> None:
    paired = result["paired_comparison"]
    repair = result["repair_audit"]
    first = result["first_candidate_audit"]
    lines = [
        "A3 P1 local-repair validation (max_local=3)",
        "STATUS: MONTH-6 DEVELOPMENT/CORRECTION EVIDENCE ONLY",
        "DECISION: candidate for supervisor audit; NOT FROZEN",
        "",
        f"ASR old: {paired['asr_curve_old_maxlocal1']}",
        f"ASR new: {paired['asr_curve_new_maxlocal3']}",
        f"ASR@5 delta: {paired['asr5_delta']}",
        f"AUC old/new: {paired['auc_old']}/{paired['auc_new']}",
        f"Success old/new: {paired['n_success_old']}/{paired['n_success_new']}",
        f"Exhaustions old/new: {paired['exhaustions_old']}/{paired['exhaustions_new']}",
        f"Valid submission rate old/new: {paired['valid_submission_rate_old']}/"
        f"{paired['valid_submission_rate_new']}",
        f"Transitions: {paired['success_transition_counts']}",
        f"Stop reasons old/new: {paired['stop_reason_counts_old']}/"
        f"{paired['stop_reason_counts_new']}",
        "",
        "Local repair:",
        f"- real queries: {repair['n_real_query_records']}",
        f"- first candidate legal: {repair['n_first_candidate_legal_and_submitted']}",
        f"- triggered second/third: {repair['n_triggered_second_generation']}/"
        f"{repair['n_triggered_third_generation']}",
        f"- rejection reasons: {repair['local_rejection_reason_counts']}",
        f"- repaired submissions: {repair['n_repaired_to_legal_submission']} "
        f"{repair['repaired_submission_public_outcomes']}",
        f"- three-attempt exhaustions: {repair['n_three_attempt_exhaustions']}",
        "",
        "First-candidate consistency:",
        f"- common queries: {first['n_common_queries']}",
        f"- raw prompt hash matches: {first['raw_prompt_hash_match_count']}",
        f"- normalised prompt/payload matches: "
        f"{first['normalised_prompt_match_count']}/"
        f"{first['normalised_payload_match_count']}",
        f"- exact candidate/changes matches: "
        f"{first['first_candidate_exact_match_count']}/"
        f"{first['first_candidate_changes_match_count']}",
        f"- interpretation: {first['note']}",
        "",
        "DEVELOPMENT AND CORRECTION TRACE",
        "old runner deviation: P1 temperature=0 was run with max_local=1 despite "
        "the existing formal A3 configuration specifying max_local=3.",
        "RP1 correction rejected: B=3 removed false exhaustion but reduced attack "
        "performance and provided no observed rank2/3 online fallback contribution.",
        "restored P1+3: unchanged legacy P1 single-candidate prompt/schema; local "
        "deterministic rejection alone can trigger up to two additional planner calls.",
        "evidence: fixed paired 25-anchor result, local-repair audit, prompt/candidate "
        "consistency audit, token ledger and invariant checks.",
        "remaining risks: external API output at temperature=0 is not guaranteed "
        "bitwise deterministic; supervisor audit is required before any freeze.",
    ]
    (parent / "validation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument("--old-run", type=Path, default=OLD_T0_RUN)
    parser.add_argument("--anchors-file", type=Path, default=OLD_ANCHORS)
    parser.add_argument("--parent-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    old_summary_path = args.old_run / "summary.json"
    if not args.raw.exists():
        raise FileNotFoundError(f"BAF raw data not found: {args.raw}")
    if not old_summary_path.exists() or not args.anchors_file.exists():
        raise FileNotFoundError("Frozen old T0 summary or anchor file is missing.")
    old = json.loads(old_summary_path.read_text(encoding="utf-8"))
    anchors_meta = json.loads(args.anchors_file.read_text(encoding="utf-8"))
    anchors = [str(item) for item in anchors_meta["anchor_ids"]]
    if len(anchors) != 25 or old["n_anchors"] != 25:
        raise RuntimeError("Validation is frozen to the exact old 25 anchors.")
    if [str(row["anchor_id"]) for row in old["per_anchor"]] != anchors:
        raise RuntimeError("Old summary anchor order differs from frozen metadata.")
    if FORMAL_A3_MODEL_CONFIG.max_local_generation_attempts_per_query != MAX_LOCAL:
        raise RuntimeError("FORMAL_A3_MODEL_CONFIG no longer specifies max_local=3.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.parent_dir is not None:
        parent = args.parent_dir
        if parent.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {parent}")
        parent.mkdir(parents=True, exist_ok=False)
    else:
        parent = new_run_directory(
            f"a3_p1_local_repair_validation_n25_m2_q5_t0_maxlocal3_seed"
            f"{DEFAULT_EXPERIMENT_SEED}_{stamp}",
            parent=EXPERIMENTS_ROOT / "a3" / "prompt_development",
            stage="experiments",
        )

    budget = AttackBudget(q_max=5, m_max=2)
    preflight = preflight_outbound_payloads(
        anchor_ids=anchors,
        budget=budget,
        experiment_seed=DEFAULT_EXPERIMENT_SEED,
        raw_path=args.raw,
        artefact_dir=args.artefact_dir,
        temperatures=(0.0,),
        prompt_version=PROMPT_VERSION_P1_COMPACT,
        max_local_generation_attempts_per_query=MAX_LOCAL,
    )
    (parent / "outbound_preflight_manifest.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol = {
        "status": "month6_a3_p1_local_repair_development_only",
        "decision_status": "candidate for supervisor audit",
        "prompt_version": PROMPT_VERSION_P1_COMPACT,
        "model": FORMAL_A3_MODEL_CONFIG.model,
        "temperature": 0.0,
        "top_p": FORMAL_A3_MODEL_CONFIG.top_p,
        "thinking_disabled": FORMAL_A3_MODEL_CONFIG.thinking_disabled,
        "max_parse_retries": FORMAL_A3_MODEL_CONFIG.max_parse_retries,
        "max_local_generation_attempts_per_query": MAX_LOCAL,
        "Q": 5,
        "m": 2,
        "K": 10,
        "feedback_mode": "label_only",
        "data_split": "dev_month6",
        "month7_opened": False,
        "anchors": anchors,
        "old_maxlocal1_run": str(args.old_run),
        "p1_prompt_schema_changed": False,
    }
    (parent / "validation_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (parent / "run_manifest.json").write_text(
        json.dumps(
            {
                **protocol,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command_argv": sys.argv,
                "outbound_preflight_status": preflight["status"],
                "provenance": _implementation_provenance(args.artefact_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        print(f"Outbound preflight PASS; no API calls made. parent_dir: {parent}")
        return 0

    condition = parent / (
        "condition_P1_t0_maxlocal3_" + PROMPT_VERSION_P1_COMPACT
    )
    condition.mkdir(parents=True, exist_ok=False)
    new = run_variant(
        variant="P1",
        anchor_ids=anchors,
        budget=budget,
        experiment_seed=DEFAULT_EXPERIMENT_SEED,
        run_dir=condition,
        raw_path=args.raw,
        artefact_dir=args.artefact_dir,
        temperature=0.0,
        max_local_generation_attempts_per_query=MAX_LOCAL,
    )
    new["run_dir"] = str(condition)
    (condition / "summary.json").write_text(
        json.dumps(to_jsonable(new), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paired = _paired_comparison(old=old, new=new, anchors=anchors)
    repair = _repair_audit(new)
    first = _first_candidate_audit(
        old_run=args.old_run, new_run=condition, anchors=anchors
    )
    result = {
        "status": "month6_a3_p1_local_repair_development_only",
        "decision_status": "candidate for supervisor audit",
        "paired_comparison": paired,
        "repair_audit": repair,
        "first_candidate_audit": first,
    }
    (parent / "paired_comparison_and_repair_audit.json").write_text(
        json.dumps(to_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(parent, result)
    print((parent / "validation_report.txt").read_text(encoding="utf-8"))
    print(f"parent_dir: {parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
