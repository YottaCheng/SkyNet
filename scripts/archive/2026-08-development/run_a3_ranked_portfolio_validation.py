#!/usr/bin/env python3
"""Validate the A3 ranked-portfolio correction on the frozen 25 Month-6 anchors.

This is development/correction evidence only, not a dissertation finding.  It
never selects or opens Month 7 and refuses any condition other than the frozen
P1 semantics at temperature=0, Q=5, m=2, K=10 and portfolio cap B=3.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    PROMPT_VERSION_P1_RANKED_PORTFOLIO,
    RANKED_PORTFOLIO_CAP,
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


def _engineering_gate(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    primary = summary["primary_criteria"]
    novelty = primary["E_post_block_projected_novelty_rate"]
    checks = {
        "projected_duplicates_zero": (
            primary["A_projected_duplicate_submission_count"] == 0
        ),
        "local_generation_exhaustion_zero": (
            primary["B_regeneration_exhaustion_rate"]
            <= PREDEFINED_ENGINEERING_CRITERIA[
                "local_generation_exhaustion_rate_max"
            ]
        ),
        "valid_submission_rate": (
            primary["C_valid_submission_rate_over_attempted_query_records"]
            >= PREDEFINED_ENGINEERING_CRITERIA["valid_submission_rate_min"]
        ),
        "no_forbidden_leakage": primary["D_forbidden_leakage_hits"] == 0,
        "env_step_equals_q": primary["D_env_step_equals_q_used_all_anchors"],
        "no_parse_failures": primary["F_parse_failure_events"] == 0,
        "no_invalid_submissions": primary["C_invalid_submissions"] == 0,
        "post_block_novelty": (
            novelty is None
            or novelty
            >= PREDEFINED_ENGINEERING_CRITERIA[
                "post_block_projected_novelty_rate_min"
            ]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def _paired_comparison(
    *,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    anchors: Sequence[str],
) -> dict[str, Any]:
    old_by_id = {str(row["anchor_id"]): row for row in old["per_anchor"]}
    new_by_id = {str(row["anchor_id"]): row for row in new["per_anchor"]}
    if list(old_by_id) != list(anchors) or list(new_by_id) != list(anchors):
        raise RuntimeError("Old/new summaries do not preserve the exact anchor order.")

    paired = []
    for anchor_id in anchors:
        before = old_by_id[anchor_id]
        after = new_by_id[anchor_id]
        paired.append(
            {
                "anchor_id": anchor_id,
                "old_success": bool(before["success"]),
                "new_success": bool(after["success"]),
                "outcome_transition": (
                    f"{bool(before['success'])}->{bool(after['success'])}"
                ),
                "old_stop_reason": before["stop_reason"],
                "new_stop_reason": after["stop_reason"],
                "old_queries_used": int(before["queries_used"]),
                "new_queries_used": int(after["queries_used"]),
                "old_attempts_to_success": before["attempts_to_success"],
                "new_attempts_to_success": after["attempts_to_success"],
                "old_public_sequence": [
                    step.get("public_label") for step in before.get("trajectory", [])
                ],
                "new_public_sequence": [
                    step.get("public_label") for step in after.get("trajectory", [])
                ],
                "new_selected_portfolio_ranks": [
                    step.get("selected_portfolio_rank")
                    for step in after.get("trajectory", [])
                ],
                "new_local_rejection_reasons": [
                    local.get("local_rejection_reason")
                    for step in after.get("trajectory", [])
                    for local in step.get("local_generation_records", [])
                    if local.get("local_rejection_reason")
                ],
            }
        )

    old_curve = old["asr_curve"]
    new_curve = new["asr_curve"]
    known = {"842574", "892990", "867193"}
    gate_ok, gate_failures = _engineering_gate(new)
    return {
        "status": "month6_a3_correction_development_only_not_dissertation_findings",
        "n_anchors": len(anchors),
        "exact_anchor_order_match": True,
        "only_intended_change": (
            "A3 prompt/response version plus deterministic ranked portfolio B=3"
        ),
        "asr_curve_old": old_curve,
        "asr_curve_new": new_curve,
        "asr_curve_delta": {
            key: float(new_curve[key]) - float(old_curve[key]) for key in old_curve
        },
        "asr5_delta": float(new_curve["ASR@5"]) - float(old_curve["ASR@5"]),
        "old_mean_queries_to_success": old["mean_queries_to_success"],
        "new_mean_queries_to_success": new["mean_queries_to_success"],
        "old_exhaustions": old["regeneration_exhaustions"],
        "new_exhaustions": new["regeneration_exhaustions"],
        "old_valid_submission_rate": old["primary_criteria"][
            "C_valid_submission_rate_over_attempted_query_records"
        ],
        "new_valid_submission_rate": new["primary_criteria"][
            "C_valid_submission_rate_over_attempted_query_records"
        ],
        "old_total_llm_calls": old["total_llm_calls"],
        "new_total_llm_calls": new["total_llm_calls"],
        "old_token_usage": old["token_usage"],
        "new_token_usage": new["token_usage"],
        "old_total_estimated_cost_usd": old["total_estimated_cost_usd"],
        "new_total_estimated_cost_usd": new["total_estimated_cost_usd"],
        "meets_existing_engineering_thresholds": gate_ok,
        "failed_engineering_thresholds": gate_failures,
        "known_exhaustion_cases": [
            row for row in paired if row["anchor_id"] in known
        ],
        "per_anchor": paired,
    }


def _write_human_report(parent: Path, comparison: Mapping[str, Any]) -> None:
    lines = [
        "A3 ranked-portfolio correction validation",
        "STATUS: MONTH-6 DEVELOPMENT/CORRECTION EVIDENCE ONLY",
        "DECISION STATUS: candidate for supervisor audit",
        "",
        f"ASR old: {comparison['asr_curve_old']}",
        f"ASR new: {comparison['asr_curve_new']}",
        f"ASR@5 delta: {comparison['asr5_delta']}",
        f"Exhaustions old/new: {comparison['old_exhaustions']}/"
        f"{comparison['new_exhaustions']}",
        f"Valid submission rate old/new: "
        f"{comparison['old_valid_submission_rate']}/"
        f"{comparison['new_valid_submission_rate']}",
        f"LLM calls old/new: {comparison['old_total_llm_calls']}/"
        f"{comparison['new_total_llm_calls']}",
        f"Estimated cost USD old/new: "
        f"{comparison['old_total_estimated_cost_usd']}/"
        f"{comparison['new_total_estimated_cost_usd']}",
        f"Existing engineering thresholds met: "
        f"{comparison['meets_existing_engineering_thresholds']}",
        f"Failed thresholds: {comparison['failed_engineering_thresholds']}",
        "",
        "Known former exhaustion cases:",
    ]
    for row in comparison["known_exhaustion_cases"]:
        lines.append(
            f"- {row['anchor_id']}: stop {row['old_stop_reason']} -> "
            f"{row['new_stop_reason']}; success {row['outcome_transition']}; "
            f"new local rejects={row['new_local_rejection_reasons']}"
        )
    lines.extend(
        [
            "",
            "DEVELOPMENT AND CORRECTION TRACE",
            "observed problem: three old T0 episodes stopped after a single illegal "
            "local proposal despite unproven remaining legal action space.",
            "diagnosis: the LLM lacked exact static-lock slot accounting and the "
            "single-candidate interface gave the deterministic executor no legal "
            "fallback within the same planning call.",
            "change: versioned ranked portfolio B=3; one LLM call/query; exact public "
            "slot accounting; deterministic first-legal selection; no Q or feedback "
            "for rejected/unselected alternatives.",
            "verification: see paired_comparison.json, per-query audit artefacts and "
            "the test command recorded in the supervisor handoff.",
            "remaining risk: development evidence is n=25 at temperature=0 and must "
            "be independently audited before any freeze/formal run.",
        ]
    )
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

    if not args.raw.exists():
        raise FileNotFoundError(f"BAF raw data not found: {args.raw}")
    old_summary_path = args.old_run / "summary.json"
    if not old_summary_path.exists() or not args.anchors_file.exists():
        raise FileNotFoundError("Frozen old T0 summary or anchor file is missing.")
    old = json.loads(old_summary_path.read_text(encoding="utf-8"))
    anchors_meta = json.loads(args.anchors_file.read_text(encoding="utf-8"))
    anchors = [str(item) for item in anchors_meta["anchor_ids"]]
    if len(anchors) != 25 or old["n_anchors"] != 25:
        raise RuntimeError("Validation is frozen to the exact old 25-anchor set.")
    if [str(row["anchor_id"]) for row in old["per_anchor"]] != anchors:
        raise RuntimeError("Old T0 summary anchor order differs from frozen metadata.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.parent_dir is not None:
        parent = args.parent_dir
        if parent.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {parent}")
        parent.mkdir(parents=True, exist_ok=False)
    else:
        parent = new_run_directory(
            f"a3_ranked_portfolio_validation_n25_m2_q5_t0_seed"
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
        prompt_version=PROMPT_VERSION_P1_RANKED_PORTFOLIO,
    )
    (parent / "outbound_preflight_manifest.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol = {
        "status": "month6_a3_correction_development_only_not_dissertation_findings",
        "prompt_version": PROMPT_VERSION_P1_RANKED_PORTFOLIO,
        "model": FORMAL_A3_MODEL_CONFIG.model,
        "temperature": 0.0,
        "top_p": FORMAL_A3_MODEL_CONFIG.top_p,
        "thinking_disabled": FORMAL_A3_MODEL_CONFIG.thinking_disabled,
        "Q": 5,
        "m": 2,
        "K": 10,
        "portfolio_cap": RANKED_PORTFOLIO_CAP,
        "llm_calls_per_query": 1,
        "max_parse_retries": 0,
        "feedback_mode": "label_only",
        "data_split": "dev_month6",
        "month7_opened": False,
        "anchors": anchors,
        "old_t0_run": str(args.old_run),
        "only_intended_change": (
            "A3 prompt/response version plus deterministic ranked portfolio B=3"
        ),
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
        "condition_RP1_t0_" + PROMPT_VERSION_P1_RANKED_PORTFOLIO
    )
    condition.mkdir(parents=True, exist_ok=False)
    new = run_variant(
        variant="RP1",
        anchor_ids=anchors,
        budget=budget,
        experiment_seed=DEFAULT_EXPERIMENT_SEED,
        run_dir=condition,
        raw_path=args.raw,
        artefact_dir=args.artefact_dir,
        temperature=0.0,
    )
    new["run_dir"] = str(condition)
    (condition / "summary.json").write_text(
        json.dumps(to_jsonable(new), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison = _paired_comparison(old=old, new=new, anchors=anchors)
    (parent / "paired_comparison.json").write_text(
        json.dumps(to_jsonable(comparison), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_human_report(parent, comparison)
    print((parent / "validation_report.txt").read_text(encoding="utf-8"))
    print(f"parent_dir: {parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
