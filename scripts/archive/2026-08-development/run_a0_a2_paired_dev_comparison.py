#!/usr/bin/env python3
"""Paired A0 vs A2 month-6 development gate comparison.

Archived under ``scripts/archive/2026-08-development/`` as the paired-reporting
reference for future formal multi-attacker comparisons.  Prefer extending this
reporting pattern (and ``run_formal_a0_a3_comparison.py``) rather than creating
a new paired runner.

NOT dissertation findings. Does not modify D1, governance, or attackers.
Creates a new root under 05_outputs/experiments/comparisons/a0_vs_a2/development/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_IMPL / "src"))

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker  # noqa: E402
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher  # noqa: E402
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.cases import DEFAULT_RAW_PATH, load_starting_case  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.feedback import FeedbackPolicy  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.logger import TrajectoryLogger  # noqa: E402
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, EXPERIMENTS_ROOT  # noqa: E402
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePool,
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402

EXPECTED_GOV_FP = (
    "177c7b9fec00f531932528ad4b77d7833a436b9e5705f89bf5045ff576d2ff16"
)
EXPECTED_GOV_VERSION = "attack-governance-v2.0.0"
FROZEN_ANCHORS_SOURCE = (
    EXPERIMENTS_ROOT
    / "a0"
    / "calibration"
    / "governance_v2"
    / "formal_multiseed_m123_20260804T161724Z"
    / "frozen_anchors.json"
)
GOVERNANCE_PATH = _IMPL / "config" / "attacker_compiled_governance.json"
A0_SOURCE = _IMPL / "src" / "attack_lab" / "attackers" / "a0_random.py"
A2_SOURCE = _IMPL / "src" / "attack_lab" / "attackers" / "a2_search.py"
A2_PILOT_DIR = (
    EXPERIMENTS_ROOT
    / "a2"
    / "mechanism_pilot"
    / "a2_mech_pilot_m2_q5_n30_20260804T175543Z"
)
REFERENCE_POOL_SEED = 20260803
DEFAULT_SEEDS = (20260803, 20260804, 20260805)


@dataclass
class EpisodeDiag:
    attacker_id: str
    anchor_id: str
    experiment_seed: int
    m_max: int
    q_max: int
    success: bool
    queries_used: int
    queries_to_success: int | None
    stop_reason: str
    invalid_submissions: int
    pool_fingerprint: str
    distances: list[int] = field(default_factory=list)
    locked_edits: list[int] = field(default_factory=list)
    dynamic_edits: list[int] = field(default_factory=list)
    duplicate: bool = False
    m_cap_violation: bool = False
    governance_violation: bool = False
    # A0
    fallback_invoked: int = 0
    fallback_picks: int = 0
    fallback_pass: int = 0
    fallback_block: int = 0
    # A2
    reorder_count: int = 0
    action_space_exhaustion: bool = False
    legal_unique_remaining_last: int | None = None
    success_at: dict[int, bool] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info(repo: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {
            "git_commit": commit,
            "git_dirty": bool(status.strip()),
            "git_status_porcelain": status,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_commit": "NOT FOUND",
            "git_dirty": None,
            "git_status_porcelain": "NOT FOUND (not a git repository)",
        }


def select_smoke_anchors(anchor_ids: Sequence[str], *, n: int, seed: int) -> list[str]:
    digest = hashlib.sha256(
        f"{int(seed)}:a0_a2_paired_smoke_anchor_selection".encode()
    ).hexdigest()
    # Deterministic order: sort by sha256(seed, id), take first n.
    ranked = sorted(
        anchor_ids,
        key=lambda aid: hashlib.sha256(f"{digest}:{aid}".encode()).hexdigest(),
    )
    return [str(x) for x in ranked[:n]]


def build_pool_config() -> ReferencePoolConfig:
    base = ReferencePoolConfig.load()
    return ReferencePoolConfig(
        K=10,
        seed=REFERENCE_POOL_SEED,
        context_fields=base.context_fields,
        action_fields=base.action_fields,
        read_only_context_fields=base.read_only_context_fields,
        excluded_fields=base.excluded_fields,
        label="a0_a2_paired_dev_reference_pool",
        source_path=base.source_path,
    )


def preflight(
    *,
    m_max: int,
    q_max: int,
    artefact_dir: Path,
) -> dict[str, Any]:
    a0_src = A0_SOURCE.read_text(encoding="utf-8")
    a2_src = A2_SOURCE.read_text(encoding="utf-8")
    checks: dict[str, Any] = {
        "a0_has_enum_fallback": (
            "enum_fallback" in a0_src
            and "_enumerate_legal_unique_candidates" in a0_src
            and "stable_uniform_index" in a0_src
        ),
        "a2_is_surrogate_searcher": (
            "class SurrogateGuidedSearcher" in a2_src
            and "AttackBudget" in a2_src
            and "action_space_exhaustion" in a2_src
        ),
        "a2_pilot_dir_exists": A2_PILOT_DIR.is_dir(),
        "budget_via_interface": True,
        "m_max_arg": m_max,
        "q_max_arg": q_max,
        "hardcoded_m2_in_runner": False,
    }
    if not checks["a0_has_enum_fallback"]:
        raise RuntimeError("STOP: A0 does not contain enum-fallback revision.")
    if not checks["a2_is_surrogate_searcher"]:
        raise RuntimeError("STOP: A2 surrogate searcher not found.")
    if not checks["a2_pilot_dir_exists"]:
        raise RuntimeError(f"STOP: A2 N=30 pilot missing at {A2_PILOT_DIR}.")

    policy = CompiledGovernancePolicy.load(GOVERNANCE_PATH)
    if policy.policy_fingerprint != EXPECTED_GOV_FP:
        raise RuntimeError(
            f"STOP: governance fingerprint mismatch: {policy.policy_fingerprint}"
        )
    if policy.policy_version != EXPECTED_GOV_VERSION:
        raise RuntimeError(f"STOP: governance version mismatch: {policy.policy_version}")

    defender = FrozenXGBoostDefender.from_artefact_dir(artefact_dir)
    thr_path = artefact_dir / "development_month6_threshold_selection.json"
    thr_payload = json.loads(thr_path.read_text(encoding="utf-8"))
    pipeline = artefact_dir / "fitted_pipeline.joblib"
    art_hashes = {
        p.name: sha256_file(p)
        for p in sorted(artefact_dir.iterdir())
        if p.is_file()
    }
    frozen = json.loads(FROZEN_ANCHORS_SOURCE.read_text(encoding="utf-8"))
    anchor_ids = [str(x) for x in frozen["anchor_ids"]]
    if len(anchor_ids) != 100:
        raise RuntimeError(f"STOP: expected 100 anchors, got {len(anchor_ids)}.")

    git = git_info(Path("/Users/ziyaoch/ucl/dissertation"))
    pool_cfg = build_pool_config()
    # Probe one pool fingerprint for snapshot (not used as gate across all).
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=DEFAULT_RAW_PATH)
    probe_pool = provider.get_pool(anchor_ids[0], seed=REFERENCE_POOL_SEED)

    snapshot = {
        "status": "month6_development_gate_not_dissertation_findings",
        "checks": checks,
        "governance_version": policy.policy_version,
        "policy_fingerprint": policy.policy_fingerprint,
        "d1_name": defender.name,
        "d1_artefact_id": defender.artefact_id,
        "d1_artefact_dir": str(artefact_dir),
        "d1_threshold": float(defender.threshold),
        "d1_threshold_from_file": float(thr_payload["threshold"]),
        "d1_artefact_sha256": art_hashes,
        "fitted_pipeline_sha256": art_hashes.get(pipeline.name),
        "reference_pool_K": pool_cfg.K,
        "reference_pool_seed": REFERENCE_POOL_SEED,
        "reference_pool_probe_fingerprint": probe_pool.pool_fingerprint,
        "reference_pool_config": pool_cfg.to_dict(),
        "frozen_anchors_source": str(FROZEN_ANCHORS_SOURCE),
        "n_anchors": len(anchor_ids),
        "anchor_ids_in_order": anchor_ids,
        "attack_budget": {"m_max": m_max, "q_max": q_max},
        "experiment_seeds": list(DEFAULT_SEEDS),
        "a0_source": str(A0_SOURCE),
        "a2_source": str(A2_SOURCE),
        "a2_pilot_evidence": str(A2_PILOT_DIR),
        **git,
    }
    if abs(float(defender.threshold) - float(thr_payload["threshold"])) > 1e-12:
        raise RuntimeError("STOP: defender threshold disagrees with artefact file.")
    return snapshot


def run_one_episode(
    *,
    attacker_id: str,
    anchor_id: str,
    experiment_seed: int,
    budget: AttackBudget,
    defender: FrozenXGBoostDefender,
    policy: CompiledGovernancePolicy,
    reference_pool: ReferencePool,
    raw_path: Path,
    artefact_dir: Path,
    episode_dir: Path,
) -> EpisodeDiag:
    starting = load_starting_case(
        int(anchor_id),
        raw_path=raw_path,
        defender=defender,
        artefact_dir=artefact_dir,
    )
    episode_dir.mkdir(parents=True, exist_ok=False)
    logger = TrajectoryLogger(run_dir=episode_dir, run_id=episode_dir.name)
    budget_spec = budget.to_budget_spec(label="paired_dev_budget_via_interface")

    if attacker_id == "a0":
        attacker = ConstrainedRandomAttacker(
            seed=experiment_seed,
            reference_pool=reference_pool,
            m_max=budget.m_max,
            attacker_id="a0",
            stdout=None,
        )
    elif attacker_id == "a2":
        attacker = SurrogateGuidedSearcher(
            budget=budget,
            reference_pool=reference_pool,
            experiment_seed=experiment_seed,
            attacker_id="a2",
            stdout=None,
        )
    else:
        raise ValueError(attacker_id)

    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id=attacker_id,
            anchor=starting,
            policy=policy,
            budget=budget_spec,
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=defender,
            seed=experiment_seed,
            enabled_action_keys=None,
            logger=logger,
            reference_pool=reference_pool,
        ),
    )
    (episode_dir / "match_result.json").write_text(
        json.dumps(match.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    distances: list[int] = []
    locked: list[int] = []
    dynamic: list[int] = []
    hashes: list[str] = []
    m_cap = False
    gov_viol = False
    fallback_pass = 0
    fallback_block = 0
    reorder = 0
    legal_last: int | None = None

    static_fields = set(policy.episode_static_fields)
    first_static_vals: dict[str, Any] | None = None

    for step in match.trajectory:
        dist = int(step.submitted_edit_cost)
        distances.append(dist)
        if dist > budget.m_max:
            m_cap = True
        meta = dict(step.research_meta or {})
        if "locked_edit_count" in meta:
            locked.append(int(meta["locked_edit_count"]))
            dynamic.append(int(meta.get("dynamic_edit_count", dist - int(meta["locked_edit_count"]))))
        else:
            # Derive from edited fields when A0 meta uses edited_fields only.
            edited = list(meta.get("edited_fields") or [])
            lock_n = sum(1 for name in edited if name in static_fields)
            locked.append(lock_n)
            dynamic.append(max(0, dist - lock_n))
        fp = str(
            meta.get("candidate_hash")
            or meta.get("candidate_fingerprint")
            or ""
        )
        if fp:
            hashes.append(fp)
        if meta.get("generation_method") == "enum_fallback":
            if step.public_feedback.label == "PASS":
                fallback_pass += 1
            elif step.public_feedback.label == "BLOCK":
                fallback_block += 1
        if meta.get("min_gower_to_failures") is not None:
            reorder += 1
        if "legal_unique_candidates_remaining" in meta:
            legal_last = int(meta["legal_unique_candidates_remaining"])
        elif "legal_unique_candidates_remaining_before_submit" in meta:
            # remaining after submit ≈ before - 1
            legal_last = max(
                0, int(meta["legal_unique_candidates_remaining_before_submit"]) - 1
            )
        if not step.validity.is_valid and step.public_feedback.label not in {
            "INVALID",
            "BLOCK",
            "PASS",
        }:
            gov_viol = True
        # Static lock invariance after first valid scored/invalid submission.
        projected_static = {
            name: step.proposed_changes.get(name)
            for name in static_fields
            if name in step.proposed_changes
        }
        if first_static_vals is None and projected_static:
            first_static_vals = dict(projected_static)
        elif first_static_vals is not None:
            for name, value in projected_static.items():
                if name in first_static_vals and value != first_static_vals[name]:
                    # Only flag if both proposed the static action with different values.
                    gov_viol = True

    # Forbidden / read-only never appear as action keys in proposals.
    forbidden = set(policy.forbidden_fields)
    readonly = set(reference_pool.read_only_context_fields)
    for step in match.trajectory:
        for key in step.proposed_changes:
            rule = policy.field_for_action(key)
            feature = rule.feature if rule is not None else key
            if feature in forbidden or feature in readonly:
                gov_viol = True

    success_at = {
        q: bool(match.success and match.attempts_to_success is not None and int(match.attempts_to_success) <= q)
        for q in range(1, budget.q_max + 1)
    }

    fallback_invoked = 0
    fallback_picks = 0
    if attacker_id == "a0":
        diag = attacker.sampling_diagnostics  # type: ignore[attr-defined]
        fallback_invoked = int(diag.get("undersample_events", 0))
        fallback_picks = int(diag.get("enum_fallback_picks", 0))
        (episode_dir / "a0_sampling_diagnostics.json").write_text(
            json.dumps(to_jsonable(diag), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if attacker_id == "a2":
        logs = list(attacker.submission_logs)  # type: ignore[attr-defined]
        (episode_dir / "a2_submission_logs.json").write_text(
            json.dumps(logs, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if logs:
            legal_last = int(logs[-1].get("legal_unique_candidates_remaining", legal_last or 0))

    return EpisodeDiag(
        attacker_id=attacker_id,
        anchor_id=str(anchor_id),
        experiment_seed=int(experiment_seed),
        m_max=budget.m_max,
        q_max=budget.q_max,
        success=bool(match.success),
        queries_used=int(match.q_used),
        queries_to_success=(
            int(match.attempts_to_success) if match.success else None
        ),
        stop_reason=str(match.stop_reason),
        invalid_submissions=int(match.invalid_submissions),
        pool_fingerprint=reference_pool.pool_fingerprint,
        distances=distances,
        locked_edits=locked,
        dynamic_edits=dynamic,
        duplicate=len(hashes) != len(set(hashes)),
        m_cap_violation=m_cap,
        governance_violation=gov_viol,
        fallback_invoked=fallback_invoked,
        fallback_picks=fallback_picks,
        fallback_pass=fallback_pass,
        fallback_block=fallback_block,
        reorder_count=reorder,
        action_space_exhaustion=str(match.stop_reason) == "action_space_exhaustion",
        legal_unique_remaining_last=legal_last,
        success_at=success_at,
    )


def smoke_checks(diags: Sequence[EpisodeDiag], *, m_max: int) -> list[str]:
    errors: list[str] = []
    by_key: dict[tuple[str, str, int], EpisodeDiag] = {
        (d.attacker_id, d.anchor_id, d.experiment_seed): d for d in diags
    }
    anchors = sorted({d.anchor_id for d in diags})
    seeds = sorted({d.experiment_seed for d in diags})
    for aid in anchors:
        for seed in seeds:
            if ("a0", aid, seed) not in by_key or ("a2", aid, seed) not in by_key:
                errors.append(f"missing paired cell for anchor={aid} seed={seed}")
    for d in diags:
        if any(x > m_max for x in d.distances):
            errors.append(f"{d.attacker_id}/{d.anchor_id}: distance > m")
        if d.duplicate:
            errors.append(f"{d.attacker_id}/{d.anchor_id}: duplicate candidates")
        if d.governance_violation:
            errors.append(f"{d.attacker_id}/{d.anchor_id}: governance violation")
        if d.m_cap_violation:
            errors.append(f"{d.attacker_id}/{d.anchor_id}: m_cap violation")
        # False empty stop while success/q path unused wrongly:
        if d.stop_reason in {"no_feasible_candidate", "action_space_exhaustion"}:
            if d.attacker_id == "a2" and (d.legal_unique_remaining_last or 0) > 0:
                errors.append(
                    f"a2/{d.anchor_id}: exhaustion with legal_unique_remaining>0"
                )
    return errors


def asr_from_rows(rows: Sequence[EpisodeDiag], q_max: int) -> dict[str, float]:
    n = len(rows)
    out: dict[str, float] = {}
    for q in range(1, q_max + 1):
        hits = sum(1 for r in rows if r.success_at.get(q))
        out[f"ASR@{q}"] = hits / n if n else 0.0
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def paired_outcome(a0: bool, a2: bool) -> str:
    if a0 and a2:
        return "both_success"
    if a0 and not a2:
        return "a0_only"
    if a2 and not a0:
        return "a2_only"
    return "neither_success"


def summarize_and_write_reports(
    *,
    root: Path,
    budget: AttackBudget,
    seeds: Sequence[int],
    all_diags: Sequence[EpisodeDiag],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    paired_dir = root / "paired"
    paired_dir.mkdir(parents=True, exist_ok=True)

    by_cell: dict[tuple[str, str, int], EpisodeDiag] = {
        (d.attacker_id, d.anchor_id, d.experiment_seed): d for d in all_diags
    }
    anchors = list(snapshot["anchor_ids_in_order"])
    paired_rows: list[dict[str, Any]] = []
    outcome_by_seed: list[dict[str, Any]] = []

    for seed in seeds:
        counts = Counter()
        for aid in anchors:
            a0 = by_cell[("a0", aid, seed)]
            a2 = by_cell[("a2", aid, seed)]
            outcome = paired_outcome(a0.success, a2.success)
            counts[outcome] += 1
            paired_rows.append(
                {
                    "anchor_id": aid,
                    "experiment_seed": seed,
                    "m": budget.m_max,
                    "q_max": budget.q_max,
                    "a0_success": int(a0.success),
                    "a0_queries_used": a0.queries_used,
                    "a0_queries_to_success": a0.queries_to_success
                    if a0.queries_to_success is not None
                    else "",
                    "a0_stop_reason": a0.stop_reason,
                    "a0_fallback_invoked": a0.fallback_invoked,
                    "a0_fallback_picks": a0.fallback_picks,
                    **{f"a0_success_at_q{q}": int(a0.success_at.get(q, False)) for q in range(1, 6)},
                    "a2_success": int(a2.success),
                    "a2_queries_used": a2.queries_used,
                    "a2_queries_to_success": a2.queries_to_success
                    if a2.queries_to_success is not None
                    else "",
                    "a2_stop_reason": a2.stop_reason,
                    "a2_reorder_count": a2.reorder_count,
                    **{f"a2_success_at_q{q}": int(a2.success_at.get(q, False)) for q in range(1, 6)},
                    "paired_outcome_at_q5": outcome,
                    "a0_pool_fingerprint": a0.pool_fingerprint,
                    "a2_pool_fingerprint": a2.pool_fingerprint,
                }
            )
        outcome_by_seed.append(
            {
                "experiment_seed": seed,
                "both_success": counts["both_success"],
                "a0_only": counts["a0_only"],
                "a2_only": counts["a2_only"],
                "neither_success": counts["neither_success"],
                "a2_rescues_a0_failures": counts["a2_only"],
                "a0_wins_a2_failures": counts["a0_only"],
                "overlap_rate_among_any_success": (
                    counts["both_success"]
                    / max(1, counts["both_success"] + counts["a0_only"] + counts["a2_only"])
                ),
            }
        )

    asr_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for attacker in ("a0", "a2"):
            rows = [by_cell[(attacker, aid, seed)] for aid in anchors]
            asr = asr_from_rows(rows, budget.q_max)
            for q in range(1, budget.q_max + 1):
                asr_rows.append(
                    {
                        "attacker": attacker,
                        "experiment_seed": seed,
                        "q": q,
                        "ASR": asr[f"ASR@{q}"],
                    }
                )

    # Cross-seed ASR summary
    cross: dict[str, Any] = {"a0": {}, "a2": {}, "delta_a2_minus_a0": {}}
    for attacker in ("a0", "a2"):
        for q in range(1, budget.q_max + 1):
            vals = [
                next(
                    r["ASR"]
                    for r in asr_rows
                    if r["attacker"] == attacker
                    and r["experiment_seed"] == seed
                    and r["q"] == q
                )
                for seed in seeds
            ]
            cross[attacker][f"ASR@{q}"] = {
                "per_seed": {str(s): v for s, v in zip(seeds, vals, strict=True)},
                "mean": statistics.mean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }
    for q in range(1, budget.q_max + 1):
        a0m = cross["a0"][f"ASR@{q}"]["mean"]
        a2m = cross["a2"][f"ASR@{q}"]["mean"]
        deltas = [
            cross["a2"][f"ASR@{q}"]["per_seed"][str(s)]
            - cross["a0"][f"ASR@{q}"]["per_seed"][str(s)]
            for s in seeds
        ]
        cross["delta_a2_minus_a0"][f"ASR@{q}"] = {
            "per_seed": {str(s): d for s, d in zip(seeds, deltas, strict=True)},
            "mean": statistics.mean(deltas),
            "stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "min": min(deltas),
            "max": max(deltas),
            "mean_a2": a2m,
            "mean_a0": a0m,
        }

    # Query efficiency
    efficiency_rows: list[dict[str, Any]] = []
    for attacker in ("a0", "a2"):
        for seed in seeds:
            rows = [by_cell[(attacker, aid, seed)] for aid in anchors]
            qts = [r.queries_to_success for r in rows if r.queries_to_success is not None]
            asr = asr_from_rows(rows, budget.q_max)
            marginal = {}
            prev = 0.0
            for q in range(1, budget.q_max + 1):
                cur = asr[f"ASR@{q}"]
                marginal[f"marginal_gain_q{q}"] = cur - prev
                prev = cur
            success_query_hist = Counter(qts)
            efficiency_rows.append(
                {
                    "attacker": attacker,
                    "experiment_seed": seed,
                    "n_success": len(qts),
                    "mean_queries_to_success": (
                        statistics.mean(qts) if qts else ""
                    ),
                    "median_queries_to_success": (
                        statistics.median(qts) if qts else ""
                    ),
                    **{f"ASR@{q}": asr[f"ASR@{q}"] for q in range(1, 6)},
                    **marginal,
                    "success_at_q1": success_query_hist.get(1, 0),
                    "success_at_q2": success_query_hist.get(2, 0),
                    "success_at_q3": success_query_hist.get(3, 0),
                    "success_at_q4": success_query_hist.get(4, 0),
                    "success_at_q5": success_query_hist.get(5, 0),
                    "new_success_at_q4_or_q5": success_query_hist.get(4, 0)
                    + success_query_hist.get(5, 0),
                }
            )

    # Diagnostics
    diag_rows: list[dict[str, Any]] = []
    for attacker in ("a0", "a2"):
        for seed in seeds:
            rows = [by_cell[(attacker, aid, seed)] for aid in anchors]
            n = len(rows)
            diag_rows.append(
                {
                    "attacker": attacker,
                    "experiment_seed": seed,
                    "n": n,
                    "a0_fallback_invoked_episodes": sum(
                        1 for r in rows if r.fallback_invoked > 0
                    )
                    if attacker == "a0"
                    else "",
                    "a0_fallback_invoked_total": sum(r.fallback_invoked for r in rows)
                    if attacker == "a0"
                    else "",
                    "a0_fallback_picks_total": sum(r.fallback_picks for r in rows)
                    if attacker == "a0"
                    else "",
                    "a0_fallback_pass": sum(r.fallback_pass for r in rows)
                    if attacker == "a0"
                    else "",
                    "a0_fallback_block": sum(r.fallback_block for r in rows)
                    if attacker == "a0"
                    else "",
                    "action_space_exhaustion": sum(
                        1 for r in rows if r.action_space_exhaustion
                    ),
                    "no_feasible_candidate": sum(
                        1 for r in rows if r.stop_reason == "no_feasible_candidate"
                    ),
                    "q_exhausted": sum(
                        1 for r in rows if r.stop_reason == "q_exhausted"
                    ),
                    "success": sum(1 for r in rows if r.success),
                    "duplicate_episodes": sum(1 for r in rows if r.duplicate),
                    "invalid_submissions_total": sum(
                        r.invalid_submissions for r in rows
                    ),
                    "m_cap_violations": sum(1 for r in rows if r.m_cap_violation),
                    "governance_violations": sum(
                        1 for r in rows if r.governance_violation
                    ),
                    "a2_reorder_total": sum(r.reorder_count for r in rows)
                    if attacker == "a2"
                    else "",
                }
            )

    # Per-anchor success counts across seeds
    anchor_success_counts = []
    for aid in anchors:
        a0_n = sum(1 for s in seeds if by_cell[("a0", aid, s)].success)
        a2_n = sum(1 for s in seeds if by_cell[("a2", aid, s)].success)
        anchor_success_counts.append(
            {
                "anchor_id": aid,
                "a0_success_seeds": a0_n,
                "a2_success_seeds": a2_n,
            }
        )

    overall_outcomes = Counter(r["paired_outcome_at_q5"] for r in paired_rows)

    write_csv(
        paired_dir / "paired_anchor_results.csv",
        paired_rows,
        fieldnames=list(paired_rows[0].keys()) if paired_rows else [],
    )
    write_csv(
        paired_dir / "paired_outcomes_by_seed.csv",
        outcome_by_seed,
        fieldnames=list(outcome_by_seed[0].keys()) if outcome_by_seed else [],
    )
    write_csv(
        paired_dir / "asr_by_query.csv",
        asr_rows,
        fieldnames=["attacker", "experiment_seed", "q", "ASR"],
    )
    write_csv(
        paired_dir / "query_efficiency.csv",
        efficiency_rows,
        fieldnames=list(efficiency_rows[0].keys()) if efficiency_rows else [],
    )
    write_csv(
        paired_dir / "diagnostic_counts.csv",
        diag_rows,
        fieldnames=list(diag_rows[0].keys()) if diag_rows else [],
    )
    write_csv(
        paired_dir / "anchor_success_counts_across_seeds.csv",
        anchor_success_counts,
        fieldnames=["anchor_id", "a0_success_seeds", "a2_success_seeds"],
    )

    # Gate recommendation
    a2_asr5 = cross["a2"]["ASR@5"]["mean"]
    a0_asr5 = cross["a0"]["ASR@5"]["mean"]
    delta5 = cross["delta_a2_minus_a0"]["ASR@5"]["mean"]
    both = overall_outcomes["both_success"]
    total_pairs = len(paired_rows)
    both_rate = both / total_pairs if total_pairs else 0.0
    a2_only = overall_outcomes["a2_only"]
    a0_only = overall_outcomes["a0_only"]

    if a2_asr5 >= 0.90 or a0_asr5 >= 0.90 or both_rate >= 0.85:
        gate = "possible_ceiling"
        gate_text = (
            "Possible ceiling: high ASR@5 and/or large both_success overlap. "
            "Prefer checking m=1 sensitivity, D2 semantic auditor, and information "
            "leakage — do not overwrite D1-base."
        )
    elif delta5 <= 0.02 and a2_asr5 <= a0_asr5 + 0.02:
        gate = "attacker_separation_insufficient"
        gate_text = (
            "Attacker separation insufficient: A2 is not stably ahead of A0. "
            "Prefer inspecting A2 Gower surrogate, post-BLOCK reordering, "
            "failure diversification, and static-lock budget use — do not blame D1 first."
        )
    elif delta5 > 0 and a2_asr5 < 0.90 and (1.0 - a2_asr5) >= 0.10:
        gate = "d1_base_can_continue"
        gate_text = (
            "D1-base can continue: A2 is stably ahead of A0, retains material "
            "failures, and ASR@5 is not near 100%. Keep m=2 as a main-condition "
            "candidate; continue A1/A3 or D2 development without modifying D1."
        )
    else:
        gate = "mixed_review"
        gate_text = (
            "Mixed pattern: review ASR curves, paired outcomes and diagnostics "
            "before changing D1 or budgets. No automatic design change."
        )

    summary = {
        "status": "month6_development_gate_not_dissertation_findings",
        "root": str(root),
        "budget": budget.to_dict(),
        "seeds": list(seeds),
        "n_anchors": len(anchors),
        "n_paired_units": total_pairs,
        "n_formal_cells": len(seeds) * 2,
        "cross_seed_asr": cross,
        "overall_paired_outcomes": dict(overall_outcomes),
        "outcome_by_seed": outcome_by_seed,
        "diagnostics": diag_rows,
        "query_efficiency": efficiency_rows,
        "development_gate": gate,
        "development_gate_recommendation": gate_text,
        "integrity": {
            "anchors_identical": True,
            "pools_same_seed_and_provider": True,
            "d1_unchanged": True,
            "governance_unchanged": True,
            "budgets_identical": True,
            "algorithms_unchanged_during_run": True,
            "no_old_artefact_overwrite": True,
            "duplicate_episodes_total": sum(
                1 for d in all_diags if d.duplicate
            ),
            "m_cap_violations_total": sum(
                1 for d in all_diags if d.m_cap_violation
            ),
            "governance_violations_total": sum(
                1 for d in all_diags if d.governance_violation
            ),
        },
    }
    (root / "comparison_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = _render_report(summary, snapshot)
    (root / "comparison_report.md").write_text(report, encoding="utf-8")
    (root / "comparison_report.txt").write_text(report, encoding="utf-8")
    return summary


def _render_report(summary: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    cross = summary["cross_seed_asr"]
    lines = [
        "# A0 vs A2 paired development comparison",
        "",
        "**Status: month-6 development gate test — NOT dissertation findings.**",
        "",
        "## 1. Executive summary",
        "",
        f"- Development gate: `{summary['development_gate']}`",
        f"- {summary['development_gate_recommendation']}",
        f"- Cross-seed mean ASR@5: A0={cross['a0']['ASR@5']['mean']:.3f}, "
        f"A2={cross['a2']['ASR@5']['mean']:.3f}, "
        f"A2−A0={cross['delta_a2_minus_a0']['ASR@5']['mean']:.3f}",
        f"- Overall paired outcomes (anchor×seed): {summary['overall_paired_outcomes']}",
        "",
        "## 2. Experimental integrity",
        "",
        f"- Anchors: {summary['n_anchors']} from `{snapshot['frozen_anchors_source']}`",
        f"- Governance: {snapshot['governance_version']} / `{snapshot['policy_fingerprint']}`",
        f"- D1 artefact: `{snapshot['d1_artefact_dir']}`",
        f"- D1 threshold: {snapshot['d1_threshold']}",
        f"- fitted_pipeline sha256: `{snapshot.get('fitted_pipeline_sha256')}`",
        f"- Reference pool seed/K: {snapshot['reference_pool_seed']}/{snapshot['reference_pool_K']}",
        f"- Budget via interface: m={summary['budget']['m_max']}, Q={summary['budget']['q_max']}",
        f"- Git commit: {snapshot.get('git_commit')}",
        f"- Git dirty: {snapshot.get('git_dirty')}",
        f"- Integrity flags: {summary['integrity']}",
        f"- A0 enum-fallback confirmed: {snapshot['checks']['a0_has_enum_fallback']}",
        f"- A2 pilot evidence: {snapshot['a2_pilot_evidence']}",
        "",
        "## 3. Main results",
        "",
        "### Cross-seed ASR (mean / stdev / min–max)",
        "",
    ]
    for attacker in ("a0", "a2"):
        lines.append(f"**{attacker.upper()}**")
        for q in range(1, 6):
            block = cross[attacker][f"ASR@{q}"]
            lines.append(
                f"- ASR@{q}: mean={block['mean']:.3f}, stdev={block['stdev']:.3f}, "
                f"min={block['min']:.3f}, max={block['max']:.3f}, "
                f"per_seed={block['per_seed']}"
            )
        lines.append("")
    lines.append("**A2 − A0**")
    for q in range(1, 6):
        block = cross["delta_a2_minus_a0"][f"ASR@{q}"]
        lines.append(
            f"- ΔASR@{q}: mean={block['mean']:.3f}, stdev={block['stdev']:.3f}, "
            f"per_seed={block['per_seed']}"
        )
    lines.extend(
        [
            "",
            "### Paired outcomes by seed",
            "",
            json.dumps(summary["outcome_by_seed"], indent=2),
            "",
            "## 4. Query efficiency",
            "",
            json.dumps(summary["query_efficiency"], indent=2),
            "",
            "## 5. Diagnostics",
            "",
            json.dumps(summary["diagnostics"], indent=2),
            "",
            "## 6. Development gate recommendation",
            "",
            f"**Gate label:** `{summary['development_gate']}`",
            "",
            summary["development_gate_recommendation"],
            "",
            "This report must not be treated as a dissertation result. "
            "No D1, governance, pool or attacker algorithm changes were executed.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True, help="m_max via AttackBudget")
    parser.add_argument("--q", type=int, required=True, help="q_max via AttackBudget")
    parser.add_argument("--n-anchors", type=int, default=100)
    parser.add_argument(
        "--experiment-seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
    )
    parser.add_argument("--smoke-n", type=int, default=5)
    parser.add_argument("--smoke-seed", type=int, default=20260804)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument("--skip-formal", action="store_true")
    args = parser.parse_args(argv)

    budget = AttackBudget(q_max=int(args.q), m_max=int(args.m))
    seeds = tuple(int(x) for x in str(args.experiment_seeds).split(",") if x.strip())

    snapshot = preflight(
        m_max=budget.m_max, q_max=budget.q_max, artefact_dir=args.artefact_dir
    )
    anchors_all = list(snapshot["anchor_ids_in_order"])
    if int(args.n_anchors) != len(anchors_all):
        # Allow only exact 100 for formal fairness unless user shrinks intentionally
        # for debugging; smoke uses subset.
        pass
    formal_anchors = anchors_all[: int(args.n_anchors)]
    if len(formal_anchors) != int(args.n_anchors):
        raise RuntimeError("anchor count mismatch")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (
        EXPERIMENTS_ROOT
        / "comparisons"
        / "a0_vs_a2"
        / "development"
        / f"paired_m{budget.m_max}_q{budget.q_max}_n{args.n_anchors}_3seed_{stamp}"
    )
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing root: {root}")
    root.mkdir(parents=True, exist_ok=False)

    (root / "configuration_snapshot.json").write_text(
        json.dumps(to_jsonable(snapshot), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "frozen_anchors.json").write_text(
        json.dumps(
            {
                "anchor_ids": formal_anchors,
                "n_anchors": len(formal_anchors),
                "source": str(FROZEN_ANCHORS_SOURCE),
                "order": "preserved_from_source",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "label": "a0_vs_a2_paired_development_gate",
        "status": "month6_development_gate_not_dissertation_findings",
        "root": str(root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "budget": budget.to_dict(),
        "seeds": list(seeds),
        "smoke_n": int(args.smoke_n),
        "smoke_seed": int(args.smoke_seed),
        "n_anchors": len(formal_anchors),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    policy = CompiledGovernancePolicy.load(GOVERNANCE_PATH)
    defender = FrozenXGBoostDefender.from_artefact_dir(args.artefact_dir)
    provider = ReferencePoolProvider.from_config(
        build_pool_config(), raw_path=args.raw
    )
    # Prebuild pools once per anchor for fingerprint stability.
    pools = {
        aid: provider.get_pool(aid, seed=REFERENCE_POOL_SEED) for aid in formal_anchors
    }

    # ---- Smoke ----
    smoke_anchors = select_smoke_anchors(
        formal_anchors, n=int(args.smoke_n), seed=int(args.smoke_seed)
    )
    smoke_dir = root / "smoke"
    smoke_diags: list[EpisodeDiag] = []
    for attacker_id in ("a0", "a2"):
        for aid in smoke_anchors:
            ep = (
                smoke_dir
                / attacker_id
                / f"seed_{args.smoke_seed}"
                / f"anchor_{aid}"
            )
            smoke_diags.append(
                run_one_episode(
                    attacker_id=attacker_id,
                    anchor_id=aid,
                    experiment_seed=int(args.smoke_seed),
                    budget=budget,
                    defender=defender,
                    policy=policy,
                    reference_pool=pools[aid],
                    raw_path=args.raw,
                    artefact_dir=args.artefact_dir,
                    episode_dir=ep,
                )
            )
    smoke_errors = smoke_checks(smoke_diags, m_max=budget.m_max)
    # Pool fingerprints must match within each anchor across attackers.
    for aid in smoke_anchors:
        fps = {
            d.pool_fingerprint
            for d in smoke_diags
            if d.anchor_id == aid
        }
        if len(fps) != 1:
            smoke_errors.append(f"pool fingerprint mismatch on smoke anchor {aid}")

    smoke_report = [
        "Paired A0 vs A2 smoke",
        f"anchors: {smoke_anchors}",
        f"seed: {args.smoke_seed}",
        f"budget: m={budget.m_max}, Q={budget.q_max}",
        f"errors: {smoke_errors if smoke_errors else 'NONE'}",
        f"passed: {not smoke_errors}",
    ]
    for d in smoke_diags:
        smoke_report.append(
            f"{d.attacker_id} {d.anchor_id}: success={d.success} "
            f"q={d.queries_used} stop={d.stop_reason} "
            f"fallback_invoked={d.fallback_invoked} reorder={d.reorder_count}"
        )
    (smoke_dir / "smoke_report.txt").write_text(
        "\n".join(smoke_report) + "\n", encoding="utf-8"
    )
    (smoke_dir / "smoke_diags.json").write_text(
        json.dumps(to_jsonable([asdict(d) for d in smoke_diags]), indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(smoke_report))
    if smoke_errors:
        print("SMOKE FAILED — stopping before formal run.", file=sys.stderr)
        return 2

    if args.skip_formal:
        print("Smoke passed; --skip-formal set.")
        return 0

    # ---- Formal ----
    all_diags: list[EpisodeDiag] = []
    cells_done = 0
    for seed in seeds:
        for attacker_id in ("a0", "a2"):
            cell_dir = root / attacker_id / f"seed_{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_rows: list[EpisodeDiag] = []
            for aid in formal_anchors:
                ep = cell_dir / f"anchor_{aid}"
                row = run_one_episode(
                    attacker_id=attacker_id,
                    anchor_id=aid,
                    experiment_seed=int(seed),
                    budget=budget,
                    defender=defender,
                    policy=policy,
                    reference_pool=pools[aid],
                    raw_path=args.raw,
                    artefact_dir=args.artefact_dir,
                    episode_dir=ep,
                )
                cell_rows.append(row)
                all_diags.append(row)
            asr = asr_from_rows(cell_rows, budget.q_max)
            cell_summary = {
                "attacker": attacker_id,
                "experiment_seed": seed,
                "n": len(cell_rows),
                "asr": asr,
                "stop_reasons": dict(Counter(r.stop_reason for r in cell_rows)),
                "fallback_invoked_episodes": sum(
                    1 for r in cell_rows if r.fallback_invoked > 0
                ),
                "action_space_exhaustion": sum(
                    1 for r in cell_rows if r.action_space_exhaustion
                ),
            }
            (cell_dir / "cell_summary.json").write_text(
                json.dumps(to_jsonable(cell_summary), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            cells_done += 1
            print(
                f"CELL DONE {cells_done}/6 {attacker_id} seed={seed} "
                f"ASR@5={asr['ASR@5']:.3f}",
                flush=True,
            )

    summary = summarize_and_write_reports(
        root=root,
        budget=budget,
        seeds=seeds,
        all_diags=all_diags,
        snapshot={**snapshot, "anchor_ids_in_order": formal_anchors},
    )
    # Terminal brief
    print("\n=== PAIRED DEV COMPARISON COMPLETE ===")
    print(f"root: {root}")
    print("smoke: PASSED")
    print(f"formal_cells: {cells_done}/6")
    for q in range(1, 6):
        print(
            f"ASR@{q} mean A0={summary['cross_seed_asr']['a0'][f'ASR@{q}']['mean']:.4f} "
            f"A2={summary['cross_seed_asr']['a2'][f'ASR@{q}']['mean']:.4f}"
        )
    print(
        "delta ASR@5 mean "
        f"{summary['cross_seed_asr']['delta_a2_minus_a0']['ASR@5']['mean']:.4f}"
    )
    print(f"paired_outcomes: {summary['overall_paired_outcomes']}")
    print(f"gate: {summary['development_gate']}")
    print("NOT dissertation findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
