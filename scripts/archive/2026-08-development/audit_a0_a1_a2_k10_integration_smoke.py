#!/usr/bin/env python3
"""Post-run audit for A0/A1/A2 K10 integration smoke (NOT findings).

Read-only over a completed smoke output directory.  Writes only:
  - audit_report.json
  - SMOKE_REPORT.md
inside that directory.  Does not rerun attackers or call any LLM/API.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

_IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_IMPL / "src"))

from attack_lab.types import to_jsonable  # noqa: E402

ATTACKERS = ["a0", "a1", "a2"]
Q_MAX, M_MAX, K = 5, 2, 10


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_smoke_directory(smoke: Path) -> dict[str, Any]:
    """Audit one completed smoke root; return the audit payload."""
    required = (
        "episode_diags.json",
        "arena_precompute.json",
        "smoke_config.json",
    )
    missing = [name for name in required if not (smoke / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Smoke directory {smoke} missing required artefacts: {missing}"
        )

    diags = load_json(smoke / "episode_diags.json")
    arena = load_json(smoke / "arena_precompute.json")
    config = load_json(smoke / "smoke_config.json")
    errors = (
        load_json(smoke / "run_errors.json")
        if (smoke / "run_errors.json").exists()
        else []
    )
    experiment_seed = int(config["experiment_seeds"][0])

    arena_issues: list[dict[str, Any]] = []
    for aid, meta in arena.items():
        fps: dict[str, str] = {}
        for att in ATTACKERS:
            d = next(
                x for x in diags if x["anchor_id"] == aid and x["attacker_id"] == att
            )
            fps[att] = d["pool_fingerprint"]
            if d["pool_fingerprint"] != meta["pool_fingerprint"]:
                arena_issues.append(
                    {
                        "anchor_id": aid,
                        "issue": "pool_fp_mismatch_vs_precompute",
                        "attacker": att,
                    }
                )
            if not d.get("same_arena_start_fingerprint_ok", False):
                arena_issues.append(
                    {
                        "anchor_id": aid,
                        "issue": "start_fingerprint_drift",
                        "attacker": att,
                    }
                )
            if d.get("K") != K:
                arena_issues.append(
                    {
                        "anchor_id": aid,
                        "issue": "K_mismatch",
                        "attacker": att,
                        "K": d.get("K"),
                    }
                )
            if d.get("governance_fingerprint") != config["governance_fingerprint"]:
                arena_issues.append(
                    {
                        "anchor_id": aid,
                        "issue": "gov_fp_mismatch",
                        "attacker": att,
                    }
                )
            if d.get("d1_artefact_id") != config["d1_artefact_id"]:
                arena_issues.append(
                    {
                        "anchor_id": aid,
                        "issue": "d1_id_mismatch",
                        "attacker": att,
                    }
                )
        if len(set(fps.values())) != 1:
            arena_issues.append(
                {
                    "anchor_id": aid,
                    "issue": "cross_attacker_pool_fp_disagree",
                    "fps": fps,
                }
            )

    month7_hits: list[str] = []
    skip_names = {
        "audit_report.json",
        "SMOKE_REPORT.md",
        "post_run_audit.py",
        "run_integration_smoke.py",
        "audit_a0_a1_a2_k10_integration_smoke.py",
        "audit_stdout.json",
    }
    for path in smoke.rglob("*"):
        if not path.is_file() or path.suffix not in {
            ".json",
            ".jsonl",
            ".txt",
            ".log",
            ".md",
        }:
            continue
        if path.name in skip_names:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"final_month7|data_split[^\n]{0,40}month\s*7|month_7/", text, re.I):
            month7_hits.append(str(path.relative_to(smoke)))

    leak_hits: list[dict[str, str]] = []
    for transcript in smoke.rglob("public_transcript.txt"):
        text = transcript.read_text(encoding="utf-8", errors="ignore")
        for pat in (
            r"risk_score",
            r"feature_importance",
            r"\bshap\b",
            r"gradient",
        ):
            if re.search(pat, text, re.I):
                leak_hits.append(
                    {
                        "file": str(transcript.relative_to(smoke)),
                        "pattern": pat,
                    }
                )
        # threshold only if paired with numeric defender context
        if re.search(r"internal_defence|risk_score", text, re.I) and re.search(
            r"threshold", text, re.I
        ):
            leak_hits.append(
                {
                    "file": str(transcript.relative_to(smoke)),
                    "pattern": "threshold+internal",
                }
            )

    per: dict[str, Any] = {}
    for att in ATTACKERS:
        rows = [d for d in diags if d["attacker_id"] == att]
        stop = Counter(d["stop_reason"] for d in rows)
        successes = sum(1 for d in rows if d["success"])
        asr5 = successes / len(rows) if rows else 0.0
        submitted = sum(d["submitted_candidates"] for d in rows)
        changed = sum(d["changed_raw_values"] for d in rows)
        refb = sum(d["reference_backed_values"] for d in rows)
        nonref = sum(d["non_reference_backed_values"] for d in rows)
        prov_fail = sum(d["provenance_failures"] for d in rows)
        prov_d1 = sum(d["provenance_fail_reached_d1"] for d in rows)
        d1_calls = sum(d["d1_calls"] for d in rows)
        q_viol = sum(d["q_gt_max"] for d in rows)
        m_viol = sum(d["m_violations"] for d in rows)
        dup = sum(d["duplicate_fingerprint_repeats"] for d in rows)
        static = sum(d["static_lock_violations"] for d in rows)
        local_rej = sum(d.get("local_rejections") or 0 for d in rows)
        local_att = sum(d.get("local_generation_attempts") or 0 for d in rows)
        q_exh = int(stop.get("q_exhausted", 0))
        no_feas = (
            int(stop.get("no_feasible_candidate", 0))
            + int(stop.get("no_feasible_plan", 0))
            + int(stop.get("local_generation_exhausted", 0))
        )
        act_exh = int(stop.get("action_space_exhaustion", 0)) + int(
            stop.get("action_space_exhausted", 0)
        )
        denom = refb + nonref
        prov_rate = (refb / denom) if denom else 1.0
        per[att] = {
            "episodes": len(rows),
            "submitted_candidates": submitted,
            "changed_raw_values": changed,
            "reference_backed_values": refb,
            "non_reference_backed_values": nonref,
            "provenance_rate": prov_rate,
            "provenance_failures": prov_fail,
            "provenance_fail_reached_d1": prov_d1,
            "D1_calls": d1_calls,
            "Q_gt_5_violations": q_viol,
            "m_gt_2_violations": m_viol,
            "duplicate_violations": dup,
            "static_lock_violations": static,
            "local_rejections": local_rej,
            "local_generation_attempts_total": local_att,
            "q_exhausted": q_exh,
            "no_feasible_candidate_or_plan": no_feas,
            "action_space_exhaustion": act_exh,
            "success_count": successes,
            "descriptive_ASR@5": asr5,
            "stop_reason_counts": dict(stop),
            "mean_q_used": sum(d["q_used"] for d in rows) / len(rows),
            "mean_d1_calls": d1_calls / len(rows),
            "mean_local_generation_attempts": (
                local_att / len(rows) if att == "a1" else None
            ),
        }

    feas_rows = []
    cannot5 = 0
    for aid, meta in arena.items():
        f = meta["feasibility"]
        c = meta["legal_action_choices"]["total_legal_reference_backed_choices"]
        feas_rows.append(
            {
                "anchor_id": aid,
                "legal_reference_backed_choices": c,
                "distinct_candidates_found": f["distinct_found"],
                "can_form_at_least_5": f["can_form_at_least_q"],
            }
        )
        if not f["can_form_at_least_q"]:
            cannot5 += 1

    by_anchor: dict[str, dict[str, Any]] = defaultdict(dict)
    for d in diags:
        by_anchor[d["anchor_id"]][d["attacker_id"]] = d

    easy = multi = fail = None
    for aid, amap in by_anchor.items():
        for att in ATTACKERS:
            d = amap[att]
            if easy is None and d["success"] and (d.get("attempts_to_success") or d["q_used"]) == 1:
                easy = (aid, att, d)
            if multi is None and d["success"] and (d.get("attempts_to_success") or 0) >= 3:
                multi = (aid, att, d)
            if fail is None and (not d["success"]) and d["stop_reason"] in {
                "q_exhausted",
                "no_feasible_candidate",
                "no_feasible_plan",
                "local_generation_exhausted",
                "action_space_exhaustion",
                "action_space_exhausted",
            }:
                fail = (aid, att, d)
    if easy is None:
        for aid, amap in by_anchor.items():
            for att, d in amap.items():
                if d["success"]:
                    easy = (aid, att, d)
                    break
            if easy:
                break
    if multi is None:
        cands = [
            (aid, att, d)
            for aid, amap in by_anchor.items()
            for att, d in amap.items()
            if d["success"]
        ]
        if cands:
            multi = max(cands, key=lambda t: t[2]["q_used"])
    if fail is None:
        cands = [
            (aid, att, d)
            for aid, amap in by_anchor.items()
            for att, d in amap.items()
            if not d["success"]
        ]
        if cands:
            fail = cands[0]

    def episode_paths(aid: str, att: str) -> dict[str, Any]:
        base = smoke / "episodes" / f"anchor_{aid}" / att / f"seed_{experiment_seed}"
        return {
            "episode_dir": str(base),
            "match_result": str(base / "match_result.json"),
            "trajectory": str(base / "trajectory.jsonl"),
            "public_transcript": str(base / "public_transcript.txt"),
            "smoke_diag": str(base / "smoke_episode_diag.json"),
            "exists": base.exists(),
        }

    def summarize_episode(aid: str, att: str, d: Mapping[str, Any]) -> dict[str, Any]:
        base = smoke / "episodes" / f"anchor_{aid}" / att / f"seed_{experiment_seed}"
        steps = []
        traj = base / "trajectory.jsonl"
        if traj.exists():
            for line in traj.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                step = json.loads(line)
                meta = step.get("research_meta") or {}
                steps.append(
                    {
                        "attempt": step.get("attempt"),
                        "public_label": (step.get("public_feedback") or {}).get(
                            "label"
                        ),
                        "valid": (step.get("validity") or {}).get("is_valid"),
                        "edit_cost": step.get("submitted_edit_cost"),
                        "edited_fields": meta.get("edited_fields"),
                        "reference_ids_used": meta.get("reference_ids_used"),
                        "provenance_status": (
                            meta.get("reference_provenance") or {}
                        ).get("status"),
                        "has_internal_defence": step.get("internal_defence")
                        is not None,
                    }
                )
        a1_local = None
        if att == "a1":
            callp = base / "a1_call_record.json"
            if callp.exists():
                call = load_json(callp)
                a1_local = {
                    "llm_call_count": call.get("llm_call_count"),
                    "retry_count": call.get("retry_count"),
                    "parse_status": call.get("parse_status"),
                    "n_frozen": call.get("n_frozen_candidates"),
                    "reject_counts": call.get("governance_reject_counts"),
                }
        return {
            "anchor_id": aid,
            "attacker_id": att,
            "success": d["success"],
            "stop_reason": d["stop_reason"],
            "q_used": d["q_used"],
            "d1_calls": d["d1_calls"],
            "attempts_to_success": d.get("attempts_to_success"),
            "local_generation_attempts": d.get("local_generation_attempts"),
            "local_rejections": d.get("local_rejections"),
            "pool_fingerprint": d["pool_fingerprint"],
            "paths": episode_paths(aid, att),
            "steps": steps,
            "a1_call": a1_local,
            "arena_legal_choices": arena[aid]["legal_action_choices"][
                "total_legal_reference_backed_choices"
            ],
            "arena_can_form_5": arena[aid]["feasibility"]["can_form_at_least_q"],
        }

    reps: dict[str, Any] = {}
    if easy:
        s = summarize_episode(*easy)
        s["role"] = "easy_pass"
        reps["easy_pass"] = s
    if multi:
        s = summarize_episode(*multi)
        s["role"] = "multi_query"
        reps["multi_query"] = s
    if fail:
        s = summarize_episode(*fail)
        s["role"] = "failure_exhaustion"
        reps["failure_exhaustion"] = s

    a1_rows = [d for d in diags if d["attacker_id"] == "a1"]
    a1_accounting = {
        "episodes": len(a1_rows),
        "total_local_generation_attempts": sum(
            d.get("local_generation_attempts") or 0 for d in a1_rows
        ),
        "total_local_rejections": sum(d.get("local_rejections") or 0 for d in a1_rows),
        "total_real_D1_queries": sum(d.get("real_D1_queries") or 0 for d in a1_rows),
        "total_q_used": sum(d["q_used"] for d in a1_rows),
        "episodes_where_local_attempts_equal_q_used": sum(
            1 for d in a1_rows if d.get("local_attempts_equal_q_used")
        ),
        "note": (
            "local_generation_attempts counts LLM calls before freeze; "
            "Q/D1 count only env.step scored/charged submissions."
        ),
        "examples_where_counts_differ": [
            {
                "anchor_id": d["anchor_id"],
                "local_generation_attempts": d.get("local_generation_attempts"),
                "local_rejections": d.get("local_rejections"),
                "real_D1_queries": d.get("real_D1_queries"),
                "q_used": d["q_used"],
            }
            for d in a1_rows
            if (d.get("local_generation_attempts") or 0)
            != (d.get("real_D1_queries") or 0)
        ][:15],
    }

    gates = {
        "reference_provenance_100pct": all(
            per[a]["non_reference_backed_values"] == 0
            and per[a]["provenance_fail_reached_d1"] == 0
            for a in ATTACKERS
        ),
        "non_reference_backed_eq_0": all(
            per[a]["non_reference_backed_values"] == 0 for a in ATTACKERS
        ),
        "Q_gt_5_violations_eq_0": all(
            per[a]["Q_gt_5_violations"] == 0 for a in ATTACKERS
        ),
        "m_gt_2_violations_eq_0": all(
            per[a]["m_gt_2_violations"] == 0 for a in ATTACKERS
        ),
        "Month7_usage_eq_0": len(month7_hits) == 0,
        "same_arena_pool_fp_consistent": len(arena_issues) == 0,
        "runtime_errors_eq_0": len(errors) == 0,
        "episodes_complete_75": len(diags) == 75,
        "public_transcript_no_d1_leaks": len(leak_hits) == 0,
    }

    violations = {
        "arena_issues": arena_issues,
        "month7_hits": month7_hits,
        "public_transcript_leak_hits": leak_hits,
        "runtime_errors": errors,
        "per_attacker_nonzero": {
            att: {
                k: per[att][k]
                for k in [
                    "non_reference_backed_values",
                    "provenance_failures",
                    "provenance_fail_reached_d1",
                    "Q_gt_5_violations",
                    "m_gt_2_violations",
                    "duplicate_violations",
                    "static_lock_violations",
                ]
                if per[att][k]
            }
            for att in ATTACKERS
        },
    }

    legal_vals = [r["legal_reference_backed_choices"] for r in feas_rows]
    legal_vals_sorted = sorted(legal_vals)
    audit = {
        "status": "integration_smoke_audit_not_dissertation_findings",
        "config_echo": {
            "N": config["N"],
            "K": config["K"],
            "m": config["m"],
            "Q": config["Q"],
            "experiment_seeds": config["experiment_seeds"],
            "reference_pool_seed": config["reference_pool_seed"],
            "require_reference_provenance": config[
                "require_reference_provenance"
            ],
        },
        "gates": gates,
        "same_arena": {
            "anchors": len(arena),
            "issues": arena_issues,
            "pass": len(arena_issues) == 0,
        },
        "per_attacker": per,
        "a1_local_vs_real_query": a1_accounting,
        "action_space_feasibility": {
            "anchors": len(feas_rows),
            "anchors_cannot_form_5_distinct_legal_candidates": cannot5,
            "min_legal_choices": min(legal_vals),
            "median_legal_choices": legal_vals_sorted[len(legal_vals_sorted) // 2],
            "max_legal_choices": max(legal_vals),
            "rows": feas_rows,
        },
        "representative_episodes": reps,
        "violations": violations,
        "output_path": str(smoke),
    }
    write_json(smoke / "audit_report.json", audit)

    lines = [
        "# A0/A1/A2 K10 Integration Smoke — AUDIT",
        "",
        "**NOT dissertation findings. Development/smoke only.**",
        "",
        f"Output: `{smoke}`",
        "",
        "## A. Configuration",
        f"- N={config['N']}, K={config['K']}, m={config['m']}, Q={config['Q']}",
        f"- experiment_seeds={config['experiment_seeds']}",
        f"- reference_pool_seed={config['reference_pool_seed']}",
        f"- require_reference_provenance={config['require_reference_provenance']}",
        "- A3 run: False",
        "",
        "## B. Same-arena integrity",
        f"- PASS={len(arena_issues) == 0}; issues={len(arena_issues)}",
        "",
        "## C. Per-attacker provenance / outcomes",
    ]
    for att in ATTACKERS:
        p = per[att]
        lines.extend(
            [
                f"### {att}",
                f"- episodes={p['episodes']} submitted={p['submitted_candidates']} "
                f"changed_raw={p['changed_raw_values']}",
                f"- reference_backed={p['reference_backed_values']} "
                f"non_reference_backed={p['non_reference_backed_values']} "
                f"provenance_rate={p['provenance_rate']:.6f}",
                f"- provenance_failures={p['provenance_failures']} "
                f"prov_fail_reached_d1={p['provenance_fail_reached_d1']}",
                f"- D1_calls={p['D1_calls']} Q>5={p['Q_gt_5_violations']} "
                f"m>2={p['m_gt_2_violations']} dup={p['duplicate_violations']} "
                f"static_lock={p['static_lock_violations']}",
                f"- local_rejections={p['local_rejections']} "
                f"q_exhausted={p['q_exhausted']} "
                f"no_feasible={p['no_feasible_candidate_or_plan']} "
                f"action_space_exhaustion={p['action_space_exhaustion']}",
                f"- success={p['success_count']} "
                f"descriptive_ASR@5={p['descriptive_ASR@5']:.4f}",
                f"- stops={p['stop_reason_counts']}",
                "",
            ]
        )
    lines.extend(
        [
            "## D. Budget integrity",
            *[
                f"- {att}: Q>5={per[att]['Q_gt_5_violations']} "
                f"m>2={per[att]['m_gt_2_violations']} "
                f"D1_calls={per[att]['D1_calls']} "
                f"mean_q_used={per[att]['mean_q_used']:.3f}"
                for att in ATTACKERS
            ],
            "",
            "## E. A1 local-generation vs real-query",
            "```json",
            json.dumps(to_jsonable(a1_accounting), indent=2),
            "```",
            "",
            "## F. Action-space feasibility",
            f"- anchors_cannot_form_5={cannot5}",
            f"- legal_choices min/median/max = "
            f"{audit['action_space_feasibility']['min_legal_choices']}/"
            f"{audit['action_space_feasibility']['median_legal_choices']}/"
            f"{audit['action_space_feasibility']['max_legal_choices']}",
            "",
            "## G. Representative episodes",
        ]
    )
    for role, s in reps.items():
        lines.extend(
            [
                f"### {role}: anchor={s['anchor_id']} attacker={s['attacker_id']}",
                f"- success={s['success']} stop={s['stop_reason']} "
                f"q_used={s['q_used']} d1={s['d1_calls']} "
                f"attempts_to_success={s.get('attempts_to_success')}",
                f"- local_generation_attempts={s.get('local_generation_attempts')} "
                f"local_rejections={s.get('local_rejections')}",
                f"- path=`{s['paths']['episode_dir']}`",
                f"- steps={json.dumps(to_jsonable(s['steps']))}",
                "",
            ]
        )
    lines.extend(
        [
            "## H. Violations",
            "```json",
            json.dumps(to_jsonable(violations), indent=2),
            "```",
            "",
            "## I. Gates",
            "```json",
            json.dumps(to_jsonable(gates), indent=2),
            "```",
            "",
            f"## Output path\n`{smoke}`\n",
        ]
    )
    (smoke / "SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-run audit for a completed A0/A1/A2 K10 integration smoke "
            "directory. Read-only over episode artefacts; writes audit_report.json "
            "and SMOKE_REPORT.md only."
        )
    )
    parser.add_argument(
        "smoke_dir",
        type=Path,
        help="Path to the completed smoke output directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    smoke = args.smoke_dir.expanduser().resolve()
    if not smoke.is_dir():
        raise SystemExit(f"Smoke directory does not exist: {smoke}")
    audit = audit_smoke_directory(smoke)
    gates = audit["gates"]
    all_pass = all(bool(v) for v in gates.values())
    summary = {
        "smoke_dir": str(smoke),
        "all_gates_passed": all_pass,
        "gates": gates,
        "per_attacker": {
            att: {
                "episodes": audit["per_attacker"][att]["episodes"],
                "success_count": audit["per_attacker"][att]["success_count"],
                "descriptive_ASR@5": audit["per_attacker"][att]["descriptive_ASR@5"],
                "non_reference_backed_values": audit["per_attacker"][att][
                    "non_reference_backed_values"
                ],
                "provenance_failures": audit["per_attacker"][att][
                    "provenance_failures"
                ],
                "D1_calls": audit["per_attacker"][att]["D1_calls"],
                "Q_gt_5_violations": audit["per_attacker"][att]["Q_gt_5_violations"],
                "m_gt_2_violations": audit["per_attacker"][att]["m_gt_2_violations"],
            }
            for att in ATTACKERS
        },
        "wrote": [
            str(smoke / "audit_report.json"),
            str(smoke / "SMOKE_REPORT.md"),
        ],
    }
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
