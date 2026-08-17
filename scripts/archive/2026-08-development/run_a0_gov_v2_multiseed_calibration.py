#!/usr/bin/env python3
"""A0 governance-v2 multi-seed calibration runner (evaluation only).

Formal grid under attack-governance-v2.0.0 and B=(Q,m):

- Reuses the frozen 100 TP-BLOCK anchors from the governance-v1 m-curve run.
- Same frozen D1 artefact and threshold.
- Fixed reference-pool seed (pools independent of experiment seed).
- Experiment seeds control only A0 RNG via
  stable_hash(experiment_seed, anchor_id, attacker_id).
- Writes under ``05_outputs/experiments/a0/calibration/governance_v2/``.

Metric vocabulary (do not confuse):

- ``submission_edit_distance``: edits on one candidate vs the original anchor
  (must be <= m). This is the scientific candidate distance.
- ``episode_cumulative_edits``: sum of charged submission distances in an
  episode (reporting/efficiency only; NOT a candidate distance).
- ``submission_locked_edits`` / ``submission_dynamic_edits``: per-candidate
  split of episode-static vs per-attempt fields differing from the anchor.
- ``episode_cumulative_*``: sums of the above across charged submissions.

Calibration only — not dissertation findings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_IMPL_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _IMPL_ROOT / "scripts"
_SRC = _IMPL_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker  # noqa: E402
from attack_lab.budget import BudgetSpec  # noqa: E402
from attack_lab.cases import DEFAULT_RAW_PATH, load_starting_case  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.feedback import FeedbackPolicy  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy, GovernanceError  # noqa: E402
from attack_lab.logger import TrajectoryLogger  # noqa: E402
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator  # noqa: E402
from attack_lab.paths import (  # noqa: E402
    DEFAULT_C1_ARTEFACT_DIR,
    EXPERIMENTS_ROOT,
    new_run_directory,
)
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePool,
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402

ATTACKER = "a0"
DEFENCE = "d1"
Q_MAX = 5
REFERENCE_POOL_K = 10
REFERENCE_POOL_SEED = 20260803
FORMAL_M_LIST: tuple[int, ...] = (1, 2, 3)
FORMAL_EXPERIMENT_SEEDS: tuple[int, ...] = (20260803, 20260804, 20260805)
SMOKE_M = 1
SMOKE_N = 5
SMOKE_EXPERIMENT_SEED = 20260803

EXPECTED_POLICY_VERSION = "attack-governance-v2.0.0"
EXPECTED_POLICY_FINGERPRINT = (
    "177c7b9fec00f531932528ad4b77d7833a436b9e5705f89bf5045ff576d2ff16"
)
A0_SEED_RULE = "stable_hash(experiment_seed, anchor_id, attacker_id)"
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260804

DEFAULT_GOVERNANCE = _IMPL_ROOT / "config" / "attacker_compiled_governance.json"
LEGACY_ANCHOR_SOURCE = (
    EXPERIMENTS_ROOT
    / "a0"
    / "calibration"
    / "a0_m_curve_20260803T214158Z"
    / "frozen_anchors.json"
)
GOV_V2_CALIBRATION_ROOT = EXPERIMENTS_ROOT / "a0" / "calibration" / "governance_v2"

FAILURE_BUCKETS = (
    "q_exhausted",
    "m_exceeded",
    "no_feasible_candidate",
    "invalid_candidate",
)


@dataclass(frozen=True)
class SubmissionEditRecord:
    attempt: int
    submission_edit_distance: int
    submission_locked_edits: int
    submission_dynamic_edits: int
    dynamic_slots_remaining: int
    edited_fields: tuple[str, ...]
    scored: bool
    success: bool


@dataclass(frozen=True)
class AnchorEpisodeRow:
    anchor_id: str
    m_max: int
    experiment_seed: int
    success: bool
    queries_used: int
    episode_cumulative_edits: int
    stop_reason: str
    failure_bucket: str | None
    invalid_submissions: int
    reference_pool_fingerprint: str
    queries_to_success: int | None
    # Successful candidate distance (anchor-relative), not episode cumulative.
    success_submission_edit_distance: int | None
    episode_cumulative_locked_edits: int
    episode_cumulative_dynamic_edits: int
    mean_submission_edit_distance: float | None
    mean_submission_locked_edits: float | None
    mean_submission_dynamic_edits: float | None
    first_submission_locked_edits: int | None
    first_submission_dynamic_slots_remaining: int | None
    field_edit_counts_episode: dict[str, int] = field(default_factory=dict)
    success_field_edit_counts: dict[str, int] = field(default_factory=dict)
    submissions: tuple[SubmissionEditRecord, ...] = ()


@dataclass(frozen=True)
class CellMetrics:
    m_max: int
    experiment_seed: int
    n_anchors: int
    successes: int
    asr_at_q: dict[int, float]
    mean_queries_to_success: float | None
    median_queries_to_success: float | None
    # Mean of successful candidates' submission_edit_distance (NOT cumulative).
    mean_success_submission_edit_distance: float | None
    total_episode_cumulative_edits: int
    mean_episode_cumulative_edits: float
    mean_episode_cumulative_locked_edits: float
    mean_episode_cumulative_dynamic_edits: float
    mean_per_submission_edit_distance: float | None
    mean_per_submission_locked_edits: float | None
    mean_per_submission_dynamic_edits: float | None
    field_edit_frequency_all: dict[str, int]
    field_edit_frequency_success_candidates: dict[str, int]
    failure_buckets: dict[str, int]
    failure_reasons: dict[str, int]
    max_observed_submission_edit_distance: int
    m_cap_violations: int
    per_anchor: tuple[AnchorEpisodeRow, ...]


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return float(math.sqrt(var))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(sample.mean())
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": n,
    }


def git_commit_hash() -> str:
    for cwd in (_IMPL_ROOT, _IMPL_ROOT.parent):
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            value = completed.stdout.strip()
            if value:
                return value
        except (OSError, subprocess.CalledProcessError):
            continue
    return "UNAVAILABLE"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_anchor_ids(path: Path, *, n: int | None) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [int(x) for x in payload["anchor_ids"]]
    if not ids:
        raise SystemExit(f"ERROR: no anchors in {path}")
    if n is not None:
        if n < 1 or n > len(ids):
            raise SystemExit(f"ERROR: n_anchors={n} out of range 1..{len(ids)}")
        return ids[:n]
    return ids


def build_pool_config() -> ReferencePoolConfig:
    base = ReferencePoolConfig.load()
    return ReferencePoolConfig(
        K=REFERENCE_POOL_K,
        seed=REFERENCE_POOL_SEED,
        context_fields=base.context_fields,
        action_fields=base.action_fields,
        read_only_context_fields=base.read_only_context_fields,
        excluded_fields=base.excluded_fields,
        label="a0_gov_v2_multiseed_reference_pool",
        source_path=base.source_path,
    )


def prebuild_reference_pools(
    *,
    provider: ReferencePoolProvider,
    anchor_ids: Sequence[int],
) -> dict[str, ReferencePool]:
    pools: dict[str, ReferencePool] = {}
    for case_id in anchor_ids:
        key = str(case_id)
        pool = provider.get_pool(key, seed=REFERENCE_POOL_SEED)
        if pool.K != REFERENCE_POOL_K:
            raise RuntimeError(f"Expected K={REFERENCE_POOL_K}, got {pool.K}.")
        pools[key] = pool
    return pools


def aggregate_pool_fingerprint(pools: dict[str, ReferencePool]) -> str:
    payload = {
        str(anchor_id): pool.pool_fingerprint
        for anchor_id, pool in sorted(pools.items(), key=lambda item: item[0])
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def classify_failure_bucket(
    *,
    success: bool,
    stop_reason: str,
    invalid_submissions: int,
    scored_defender_queries: int,
) -> str | None:
    if success:
        return None
    if stop_reason in FAILURE_BUCKETS:
        return stop_reason
    if stop_reason in {"insufficient_edit_budget", "no_feasible_candidate"}:
        return "no_feasible_candidate"
    if stop_reason == "q_exhausted":
        return "q_exhausted"
    if stop_reason == "m_exceeded":
        return "m_exceeded"
    if invalid_submissions > 0 and scored_defender_queries == 0:
        return "invalid_candidate"
    return stop_reason


def extract_submission_records(
    *,
    match_trajectory: Sequence[Any],
    episode_static: set[str],
    m_max: int,
) -> tuple[SubmissionEditRecord, ...]:
    records: list[SubmissionEditRecord] = []
    for step in match_trajectory:
        event = step.budget_event
        if event is None or event.budget_rejected:
            continue
        edited = tuple(event.edited_fields)
        locked = sum(1 for name in edited if name in episode_static)
        dynamic = sum(1 for name in edited if name not in episode_static)
        distance = int(event.submitted_edit_cost)
        if distance != locked + dynamic:
            # Prefer explicit submitted cost; keep split consistent with fields.
            distance = locked + dynamic
        records.append(
            SubmissionEditRecord(
                attempt=int(step.attempt),
                submission_edit_distance=distance,
                submission_locked_edits=locked,
                submission_dynamic_edits=dynamic,
                dynamic_slots_remaining=max(0, int(m_max) - locked),
                edited_fields=edited,
                scored=bool(event.scored_defender_query),
                success=bool(step.success),
            )
        )
    return tuple(records)


def run_episode(
    *,
    case_id: int,
    m_max: int,
    experiment_seed: int,
    defender: FrozenXGBoostDefender,
    governance_policy: CompiledGovernancePolicy,
    reference_pool: ReferencePool,
    raw_path: Path,
    artefact_dir: Path,
    episode_dir: Path,
    q_max: int,
) -> AnchorEpisodeRow:
    starting = load_starting_case(
        case_id,
        raw_path=raw_path,
        defender=defender,
        artefact_dir=artefact_dir,
    )
    if starting.label != 1 or starting.initial_decision != "BLOCK":
        raise RuntimeError(
            f"Anchor {case_id} failed filter "
            f"(label={starting.label}, decision={starting.initial_decision})."
        )

    budget = BudgetSpec.development_dummy(
        q_max=q_max,
        m_max=m_max,
        label="a0_gov_v2_multiseed_dummy_budget",
    )
    logger = TrajectoryLogger(run_dir=episode_dir, run_id=episode_dir.name)
    episode_dir.mkdir(parents=True, exist_ok=False)

    attacker = ConstrainedRandomAttacker(
        seed=experiment_seed,
        reference_pool=reference_pool,
        m_max=m_max,
        attacker_id=ATTACKER,
        stdout=None,
    )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id=ATTACKER,
            anchor=starting,
            policy=governance_policy,
            budget=budget,
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

    episode_static = set(governance_policy.episode_static_fields)
    submissions = extract_submission_records(
        match_trajectory=match.trajectory,
        episode_static=episode_static,
        m_max=m_max,
    )
    for sub in submissions:
        if sub.submission_edit_distance > m_max:
            raise RuntimeError(
                f"m-cap violation on anchor {case_id}: "
                f"submission_edit_distance={sub.submission_edit_distance} > m={m_max}"
            )

    field_counts: Counter[str] = Counter()
    for sub in submissions:
        field_counts.update(sub.edited_fields)
    success_fields: dict[str, int] = {}
    success_distance: int | None = None
    for sub in submissions:
        if sub.success:
            success_fields = {name: 1 for name in sub.edited_fields}
            success_distance = sub.submission_edit_distance
            break

    qts = int(match.attempts_to_success) if match.success else None
    cumulative = int(match.total_edits_used)
    locked_cum = sum(sub.submission_locked_edits for sub in submissions)
    dynamic_cum = sum(sub.submission_dynamic_edits for sub in submissions)
    distances = [sub.submission_edit_distance for sub in submissions]
    locked_subs = [sub.submission_locked_edits for sub in submissions]
    dynamic_subs = [sub.submission_dynamic_edits for sub in submissions]
    first = submissions[0] if submissions else None
    bucket = classify_failure_bucket(
        success=bool(match.success),
        stop_reason=str(match.stop_reason),
        invalid_submissions=int(match.invalid_submissions),
        scored_defender_queries=int(match.scored_defender_queries),
    )
    return AnchorEpisodeRow(
        anchor_id=match.anchor_id,
        m_max=m_max,
        experiment_seed=experiment_seed,
        success=bool(match.success),
        queries_used=int(match.q_used),
        episode_cumulative_edits=cumulative,
        stop_reason=str(match.stop_reason),
        failure_bucket=bucket,
        invalid_submissions=int(match.invalid_submissions),
        reference_pool_fingerprint=reference_pool.pool_fingerprint,
        queries_to_success=qts,
        success_submission_edit_distance=success_distance,
        episode_cumulative_locked_edits=locked_cum,
        episode_cumulative_dynamic_edits=dynamic_cum,
        mean_submission_edit_distance=_mean([float(x) for x in distances]),
        mean_submission_locked_edits=_mean([float(x) for x in locked_subs]),
        mean_submission_dynamic_edits=_mean([float(x) for x in dynamic_subs]),
        first_submission_locked_edits=(
            first.submission_locked_edits if first is not None else None
        ),
        first_submission_dynamic_slots_remaining=(
            first.dynamic_slots_remaining if first is not None else None
        ),
        field_edit_counts_episode=dict(sorted(field_counts.items())),
        success_field_edit_counts=dict(sorted(success_fields.items())),
        submissions=submissions,
    )


def summarise_cell(
    *,
    m_max: int,
    experiment_seed: int,
    rows: Sequence[AnchorEpisodeRow],
    q_max: int,
) -> CellMetrics:
    n = len(rows)
    successes = sum(1 for row in rows if row.success)
    asr_at_q = {
        q: (
            sum(
                1
                for row in rows
                if row.success
                and row.queries_to_success is not None
                and row.queries_to_success <= q
            )
            / n
            if n
            else 0.0
        )
        for q in range(1, q_max + 1)
    }
    succ_q = [
        float(row.queries_to_success)
        for row in rows
        if row.success and row.queries_to_success is not None
    ]
    succ_dist = [
        float(row.success_submission_edit_distance)
        for row in rows
        if row.success and row.success_submission_edit_distance is not None
    ]
    raw_failures = Counter(row.stop_reason for row in rows if not row.success)
    buckets = {name: 0 for name in FAILURE_BUCKETS}
    for row in rows:
        if row.success or row.failure_bucket is None:
            continue
        if row.failure_bucket in buckets:
            buckets[row.failure_bucket] += 1

    field_all: Counter[str] = Counter()
    field_success: Counter[str] = Counter()
    all_submission_distances: list[float] = []
    all_submission_locked: list[float] = []
    all_submission_dynamic: list[float] = []
    max_dist = 0
    violations = 0
    for row in rows:
        field_all.update(row.field_edit_counts_episode)
        field_success.update(row.success_field_edit_counts)
        for sub in row.submissions:
            all_submission_distances.append(float(sub.submission_edit_distance))
            all_submission_locked.append(float(sub.submission_locked_edits))
            all_submission_dynamic.append(float(sub.submission_dynamic_edits))
            max_dist = max(max_dist, sub.submission_edit_distance)
            if sub.submission_edit_distance > m_max:
                violations += 1

    return CellMetrics(
        m_max=m_max,
        experiment_seed=experiment_seed,
        n_anchors=n,
        successes=successes,
        asr_at_q=asr_at_q,
        mean_queries_to_success=_mean(succ_q),
        median_queries_to_success=_median(succ_q) if succ_q else None,
        mean_success_submission_edit_distance=_mean(succ_dist),
        total_episode_cumulative_edits=int(
            sum(row.episode_cumulative_edits for row in rows)
        ),
        mean_episode_cumulative_edits=(
            sum(row.episode_cumulative_edits for row in rows) / n if n else 0.0
        ),
        mean_episode_cumulative_locked_edits=(
            sum(row.episode_cumulative_locked_edits for row in rows) / n if n else 0.0
        ),
        mean_episode_cumulative_dynamic_edits=(
            sum(row.episode_cumulative_dynamic_edits for row in rows) / n if n else 0.0
        ),
        mean_per_submission_edit_distance=_mean(all_submission_distances),
        mean_per_submission_locked_edits=_mean(all_submission_locked),
        mean_per_submission_dynamic_edits=_mean(all_submission_dynamic),
        field_edit_frequency_all=dict(sorted(field_all.items())),
        field_edit_frequency_success_candidates=dict(sorted(field_success.items())),
        failure_buckets=buckets,
        failure_reasons=dict(sorted(raw_failures.items())),
        max_observed_submission_edit_distance=max_dist,
        m_cap_violations=violations,
        per_anchor=tuple(rows),
    )


def stratified_asr(
    rows: Sequence[AnchorEpisodeRow],
    *,
    key_fn,
) -> dict[str, dict[str, float | int]]:
    groups: dict[Any, list[AnchorEpisodeRow]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        groups[key].append(row)
    out: dict[str, dict[str, float | int]] = {}
    for key in sorted(groups):
        group = groups[key]
        n = len(group)
        succ = sum(1 for row in group if row.success)
        out[str(key)] = {
            "n": n,
            "successes": succ,
            "ASR": (succ / n) if n else 0.0,
        }
    return out


def cross_seed_summary(
    cells: Sequence[CellMetrics],
    *,
    q_max: int,
) -> dict[str, Any]:
    by_m: dict[int, list[CellMetrics]] = defaultdict(list)
    for cell in cells:
        by_m[cell.m_max].append(cell)

    summary: dict[str, Any] = {"by_m": {}, "metric_notes": {
        "submission_edit_distance": (
            "Per-candidate edits vs original anchor; must be <= m."
        ),
        "episode_cumulative_edits": (
            "Sum of charged submission distances within an episode; "
            "efficiency/reporting only; NOT a candidate distance."
        ),
        "mean_episode_cumulative_dynamic_edits": (
            "Episode-level cumulative dynamic-field edit charges; "
            "not per-candidate distance."
        ),
        "mean_per_submission_dynamic_edits": (
            "Average dynamic edits on individual charged submissions."
        ),
        "dynamic_slots_remaining": "m - submission_locked_edits on a candidate.",
    }}

    for m_max in sorted(by_m):
        group = sorted(by_m[m_max], key=lambda c: c.experiment_seed)
        seed_rows = {
            str(c.experiment_seed): {
                "successes": c.successes,
                "n_anchors": c.n_anchors,
                **{f"ASR@{q}": c.asr_at_q[q] for q in range(1, q_max + 1)},
                "mean_queries_to_success": c.mean_queries_to_success,
                "mean_success_submission_edit_distance": (
                    c.mean_success_submission_edit_distance
                ),
                "mean_episode_cumulative_edits": c.mean_episode_cumulative_edits,
                "mean_episode_cumulative_locked_edits": (
                    c.mean_episode_cumulative_locked_edits
                ),
                "mean_episode_cumulative_dynamic_edits": (
                    c.mean_episode_cumulative_dynamic_edits
                ),
                "mean_per_submission_edit_distance": (
                    c.mean_per_submission_edit_distance
                ),
                "mean_per_submission_locked_edits": (
                    c.mean_per_submission_locked_edits
                ),
                "mean_per_submission_dynamic_edits": (
                    c.mean_per_submission_dynamic_edits
                ),
                "failure_buckets": c.failure_buckets,
                "m_cap_violations": c.m_cap_violations,
                "max_observed_submission_edit_distance": (
                    c.max_observed_submission_edit_distance
                ),
            }
            for c in group
        }
        asr_stats: dict[str, Any] = {}
        for q in range(1, q_max + 1):
            vals = [float(c.asr_at_q[q]) for c in group]
            asr_stats[f"ASR@{q}"] = {
                "values_by_seed": {
                    str(c.experiment_seed): c.asr_at_q[q] for c in group
                },
                "mean": _mean(vals),
                "std": _std(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "bootstrap_95ci": bootstrap_mean_ci(vals, seed=BOOTSTRAP_SEED + m_max + q),
            }

        # Pool all anchors across seeds for stratification (paired same anchors).
        all_rows = [row for c in group for row in c.per_anchor]
        locked_strata = stratified_asr(
            all_rows,
            key_fn=lambda r: r.first_submission_locked_edits,
        )
        dynamic_slot_strata = stratified_asr(
            all_rows,
            key_fn=lambda r: r.first_submission_dynamic_slots_remaining,
        )

        summary["by_m"][str(m_max)] = {
            "m": m_max,
            "n_seeds": len(group),
            "per_seed": seed_rows,
            "asr_summary": asr_stats,
            "asr_by_first_submission_locked_edits": locked_strata,
            "asr_by_first_submission_dynamic_slots_remaining": dynamic_slot_strata,
            "aggregate_failure_buckets": {
                name: int(sum(c.failure_buckets.get(name, 0) for c in group))
                for name in FAILURE_BUCKETS
            },
            "total_m_cap_violations": int(sum(c.m_cap_violations for c in group)),
        }
    return summary


def format_report(
    *,
    cells: Sequence[CellMetrics],
    metadata: dict[str, Any],
    cross: dict[str, Any],
    q_max: int,
) -> str:
    lines = [
        "A0 Governance-v2 Multi-seed Calibration",
        "",
        "NOTE: Calibration only. Not dissertation findings.",
        "",
        "Metric notes:",
        "- submission_edit_distance = candidate vs original anchor (must be <= m)",
        "- episode_cumulative_* = sum over charged submissions (NOT candidate distance)",
        "",
        f"mode: {metadata['mode']}",
        f"status: {metadata['status']}",
        f"attacker: {metadata['attacker']}",
        f"defence: {metadata['defence']}",
        f"governance_version: {metadata['governance_version']}",
        f"policy_fingerprint: {metadata['policy_fingerprint']}",
        f"n_anchors: {metadata['n_anchors']}",
        f"reference_pool_K: {metadata['reference_pool_K']}",
        f"reference_pool_seed: {metadata['reference_pool_seed']}",
        f"Q_max: {metadata['Q_max']}",
        f"m_list: {metadata['m_list']}",
        f"experiment_seeds: {metadata['experiment_seeds']}",
        f"a0_seed_rule: {metadata['a0_seed_rule']}",
        f"anchor_source: {metadata['anchor_source']}",
        f"timestamp_utc: {metadata['timestamp_utc']}",
        f"commit_hash: {metadata['commit_hash']}",
        f"reference_pool_fingerprint: {metadata['reference_pool_fingerprint']}",
        f"frozen_threshold: {metadata['frozen_threshold']}",
        "",
    ]
    for metrics in cells:
        lines.append(f"m={metrics.m_max} seed={metrics.experiment_seed}")
        lines.append(f"  successes: {metrics.successes}/{metrics.n_anchors}")
        for q in range(1, q_max + 1):
            lines.append(f"  ASR@{q}: {metrics.asr_at_q[q]:.2%}")
        if metrics.mean_queries_to_success is None:
            lines.append("  mean queries-to-success: n/a")
            lines.append("  median queries-to-success: n/a")
        else:
            lines.append(
                f"  mean queries-to-success: {metrics.mean_queries_to_success:.2f}"
            )
            lines.append(
                f"  median queries-to-success: {metrics.median_queries_to_success:.2f}"
            )
        if metrics.mean_success_submission_edit_distance is None:
            lines.append("  mean success submission_edit_distance: n/a")
        else:
            lines.append(
                "  mean success submission_edit_distance: "
                f"{metrics.mean_success_submission_edit_distance:.2f}"
            )
        lines.append(
            "  mean episode_cumulative_edits: "
            f"{metrics.mean_episode_cumulative_edits:.2f}"
        )
        lines.append(
            "  mean episode_cumulative_locked_edits: "
            f"{metrics.mean_episode_cumulative_locked_edits:.2f}"
        )
        lines.append(
            "  mean episode_cumulative_dynamic_edits: "
            f"{metrics.mean_episode_cumulative_dynamic_edits:.2f}"
        )
        lines.append(
            "  mean per_submission_edit_distance: "
            f"{metrics.mean_per_submission_edit_distance}"
        )
        lines.append(
            "  mean per_submission_locked_edits: "
            f"{metrics.mean_per_submission_locked_edits}"
        )
        lines.append(
            "  mean per_submission_dynamic_edits: "
            f"{metrics.mean_per_submission_dynamic_edits}"
        )
        lines.append(
            "  max_observed_submission_edit_distance: "
            f"{metrics.max_observed_submission_edit_distance}"
        )
        lines.append(f"  m_cap_violations: {metrics.m_cap_violations}")
        lines.append("  field edit frequency (all charged submissions):")
        if metrics.field_edit_frequency_all:
            for name, count in metrics.field_edit_frequency_all.items():
                lines.append(f"    {name}: {count}")
        else:
            lines.append("    (none)")
        lines.append("  field edit frequency (success candidates only):")
        if metrics.field_edit_frequency_success_candidates:
            for name, count in metrics.field_edit_frequency_success_candidates.items():
                lines.append(f"    {name}: {count}")
        else:
            lines.append("    (none)")
        lines.append("  failure buckets:")
        for name in FAILURE_BUCKETS:
            lines.append(f"    {name}: {metrics.failure_buckets.get(name, 0)}")
        lines.append("")

    lines.append("=== Cross-seed summary by m ===")
    for m_key, block in cross.get("by_m", {}).items():
        lines.append(f"m={m_key}")
        for q_label, stats in block["asr_summary"].items():
            ci = stats["bootstrap_95ci"]
            lines.append(
                f"  {q_label}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"min={stats['min']:.4f} max={stats['max']:.4f} "
                f"95%CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
            )
        lines.append(
            f"  aggregate failures: {block['aggregate_failure_buckets']}"
        )
        lines.append(
            f"  total m_cap_violations: {block['total_m_cap_violations']}"
        )
        lines.append("  ASR by first_submission_locked_edits:")
        for key, val in block["asr_by_first_submission_locked_edits"].items():
            lines.append(
                f"    locked={key}: n={val['n']} successes={val['successes']} "
                f"ASR={val['ASR']:.2%}"
            )
        lines.append("  ASR by first_submission_dynamic_slots_remaining:")
        for key, val in block[
            "asr_by_first_submission_dynamic_slots_remaining"
        ].items():
            lines.append(
                f"    dynamic_slots_remaining={key}: n={val['n']} "
                f"successes={val['successes']} ASR={val['ASR']:.2%}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_curve_csv(path: Path, cells: Sequence[CellMetrics], q_max: int) -> None:
    fieldnames = [
        "m",
        "experiment_seed",
        "n_anchors",
        "successes",
        *[f"ASR@{q}" for q in range(1, q_max + 1)],
        "mean_queries_to_success",
        "median_queries_to_success",
        "mean_success_submission_edit_distance",
        "mean_episode_cumulative_edits",
        "mean_episode_cumulative_locked_edits",
        "mean_episode_cumulative_dynamic_edits",
        "mean_per_submission_edit_distance",
        "mean_per_submission_locked_edits",
        "mean_per_submission_dynamic_edits",
        "max_observed_submission_edit_distance",
        "m_cap_violations",
        "q_exhausted",
        "no_feasible_candidate",
        "invalid_candidate",
        "m_exceeded",
        "field_edit_frequency_all_json",
        "field_edit_frequency_success_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in cells:
            row: dict[str, Any] = {
                "m": metrics.m_max,
                "experiment_seed": metrics.experiment_seed,
                "n_anchors": metrics.n_anchors,
                "successes": metrics.successes,
                "mean_queries_to_success": (
                    ""
                    if metrics.mean_queries_to_success is None
                    else f"{metrics.mean_queries_to_success:.6f}"
                ),
                "median_queries_to_success": (
                    ""
                    if metrics.median_queries_to_success is None
                    else f"{metrics.median_queries_to_success:.6f}"
                ),
                "mean_success_submission_edit_distance": (
                    ""
                    if metrics.mean_success_submission_edit_distance is None
                    else f"{metrics.mean_success_submission_edit_distance:.6f}"
                ),
                "mean_episode_cumulative_edits": (
                    f"{metrics.mean_episode_cumulative_edits:.6f}"
                ),
                "mean_episode_cumulative_locked_edits": (
                    f"{metrics.mean_episode_cumulative_locked_edits:.6f}"
                ),
                "mean_episode_cumulative_dynamic_edits": (
                    f"{metrics.mean_episode_cumulative_dynamic_edits:.6f}"
                ),
                "mean_per_submission_edit_distance": (
                    ""
                    if metrics.mean_per_submission_edit_distance is None
                    else f"{metrics.mean_per_submission_edit_distance:.6f}"
                ),
                "mean_per_submission_locked_edits": (
                    ""
                    if metrics.mean_per_submission_locked_edits is None
                    else f"{metrics.mean_per_submission_locked_edits:.6f}"
                ),
                "mean_per_submission_dynamic_edits": (
                    ""
                    if metrics.mean_per_submission_dynamic_edits is None
                    else f"{metrics.mean_per_submission_dynamic_edits:.6f}"
                ),
                "max_observed_submission_edit_distance": (
                    metrics.max_observed_submission_edit_distance
                ),
                "m_cap_violations": metrics.m_cap_violations,
                "q_exhausted": metrics.failure_buckets.get("q_exhausted", 0),
                "no_feasible_candidate": metrics.failure_buckets.get(
                    "no_feasible_candidate", 0
                ),
                "invalid_candidate": metrics.failure_buckets.get(
                    "invalid_candidate", 0
                ),
                "m_exceeded": metrics.failure_buckets.get("m_exceeded", 0),
                "field_edit_frequency_all_json": json.dumps(
                    metrics.field_edit_frequency_all, sort_keys=True
                ),
                "field_edit_frequency_success_json": json.dumps(
                    metrics.field_edit_frequency_success_candidates, sort_keys=True
                ),
            }
            for q in range(1, q_max + 1):
                row[f"ASR@{q}"] = f"{metrics.asr_at_q[q]:.6f}"
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A0 governance-v2 multi-seed calibration runner."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run m=1, N=5, single experiment seed smoke only.",
    )
    parser.add_argument(
        "--anchor-source",
        type=Path,
        default=LEGACY_ANCHOR_SOURCE,
        help="frozen_anchors.json from the governance-v1 calibration.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_path = DEFAULT_RAW_PATH
    artefact_dir = DEFAULT_C1_ARTEFACT_DIR
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    commit = git_commit_hash()

    if args.smoke:
        mode = "smoke"
        m_list = (SMOKE_M,)
        experiment_seeds = (SMOKE_EXPERIMENT_SEED,)
        n_anchors = SMOKE_N
        status = "smoke_structure_check_not_formal"
        run_prefix = f"smoke_m{SMOKE_M}_n{SMOKE_N}"
    else:
        mode = "formal_grid"
        m_list = FORMAL_M_LIST
        experiment_seeds = FORMAL_EXPERIMENT_SEEDS
        n_anchors = None
        status = "draft_calibration_not_dissertation_findings"
        run_prefix = "formal_multiseed_m123"

    if not raw_path.is_file():
        print(f"ERROR: raw BAF file not found: {raw_path}", file=sys.stderr)
        return 2
    if not artefact_dir.is_dir():
        print(f"ERROR: D1 artefact dir not found: {artefact_dir}", file=sys.stderr)
        return 1
    if not args.anchor_source.is_file():
        print(f"ERROR: anchor source not found: {args.anchor_source}", file=sys.stderr)
        return 2

    try:
        governance_policy = CompiledGovernancePolicy.load(DEFAULT_GOVERNANCE)
    except (OSError, json.JSONDecodeError, GovernanceError) as exc:
        print(f"ERROR: cannot load governance policy: {exc}", file=sys.stderr)
        return 2

    if governance_policy.policy_version != EXPECTED_POLICY_VERSION:
        print(
            "ERROR: expected governance "
            f"{EXPECTED_POLICY_VERSION}, got {governance_policy.policy_version}",
            file=sys.stderr,
        )
        return 2
    if governance_policy.policy_fingerprint != EXPECTED_POLICY_FINGERPRINT:
        print(
            "ERROR: policy fingerprint mismatch.\n"
            f"  expected: {EXPECTED_POLICY_FINGERPRINT}\n"
            f"  got:      {governance_policy.policy_fingerprint}",
            file=sys.stderr,
        )
        return 2
    if len(governance_policy.per_attempt_fields) != 13:
        print("ERROR: expected 13 per-attempt fields.", file=sys.stderr)
        return 2
    if len(governance_policy.episode_static_fields) != 5:
        print("ERROR: expected 5 episode-static fields.", file=sys.stderr)
        return 2

    try:
        defender = FrozenXGBoostDefender.from_artefact_dir(artefact_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot load frozen D1: {exc}", file=sys.stderr)
        return 1

    try:
        anchor_ids = load_frozen_anchor_ids(args.anchor_source, n=n_anchors)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot load anchors: {exc}", file=sys.stderr)
        return 1
    if mode == "formal_grid" and len(anchor_ids) != 100:
        print(
            f"ERROR: formal grid requires 100 anchors; got {len(anchor_ids)}.",
            file=sys.stderr,
        )
        return 2

    pool_config = build_pool_config()
    try:
        pool_config.validate_against_governance(governance_policy.action_fields)
        provider = ReferencePoolProvider.from_config(pool_config, raw_path=raw_path)
        pools = prebuild_reference_pools(provider=provider, anchor_ids=anchor_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: reference pool setup failed: {exc}", file=sys.stderr)
        return 1

    for case_id in anchor_ids[: min(3, len(anchor_ids))]:
        again = provider.get_pool(str(case_id), seed=REFERENCE_POOL_SEED)
        if again.pool_fingerprint != pools[str(case_id)].pool_fingerprint:
            print(
                "ERROR: reference pool not stable under fixed pool seed.",
                file=sys.stderr,
            )
            return 1

    pool_set_fingerprint = aggregate_pool_fingerprint(pools)
    governance_file_fingerprint = file_sha256(DEFAULT_GOVERNANCE)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"{run_prefix}_{stamp}",
        parent=GOV_V2_CALIBRATION_ROOT,
        stage="experiments",
    )
    print(f"run_dir={run_dir}", file=sys.stderr)

    metadata: dict[str, Any] = {
        "label": "a0_gov_v2_multiseed_calibration",
        "mode": mode,
        "status": status,
        "stage": "experiments",
        "budget_protocol": "B=(Q,m)",
        "attacker": ATTACKER,
        "defence": DEFENCE,
        "governance_version": governance_policy.policy_version,
        "n_anchors": len(anchor_ids),
        "reference_pool_K": REFERENCE_POOL_K,
        "reference_pool_seed": REFERENCE_POOL_SEED,
        "Q_max": Q_MAX,
        "m_list": list(m_list),
        "experiment_seeds": list(experiment_seeds),
        "expected_cells": len(m_list) * len(experiment_seeds),
        "a0_seed_rule": A0_SEED_RULE,
        "pool_seed_rule": "fixed_reference_pool_seed_independent_of_experiment_seed",
        "anchor_source": str(args.anchor_source),
        "anchor_filter": "fraud_bool==1 AND frozen_D1_BLOCK (reused from v1 calibration)",
        "frozen_anchor_ids": [str(x) for x in anchor_ids],
        "timestamp_utc": timestamp_utc,
        "commit_hash": commit,
        "governance_fingerprint": governance_file_fingerprint,
        "governance_source_path": str(DEFAULT_GOVERNANCE),
        "policy_fingerprint": governance_policy.policy_fingerprint,
        "policy_source_sha256": governance_policy.source_sha256,
        "per_attempt_fields": list(governance_policy.per_attempt_fields),
        "episode_static_fields": list(governance_policy.episode_static_fields),
        "reference_pool_fingerprint": pool_set_fingerprint,
        "defender_artefact_dir": str(artefact_dir),
        "defender_artefact_id": defender.artefact_id,
        "frozen_threshold": defender.threshold,
        "run_dir": str(run_dir),
        "does_not_overwrite_v1_or_smoke": True,
        "metric_vocabulary": {
            "submission_edit_distance": "candidate vs original anchor; <= m",
            "episode_cumulative_edits": (
                "sum of charged submission distances; not candidate distance"
            ),
        },
    }

    (run_dir / "frozen_anchors.json").write_text(
        json.dumps(
            {
                "source": str(args.anchor_source),
                "n_anchors": len(anchor_ids),
                "anchor_ids": [str(x) for x in anchor_ids],
                "filter": metadata["anchor_filter"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(to_jsonable(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cells: list[CellMetrics] = []
    for m_max in m_list:
        for experiment_seed in experiment_seeds:
            print(
                f"=== m={m_max} experiment_seed={experiment_seed} ===",
                file=sys.stderr,
            )
            cell_dir = run_dir / f"m{m_max}_seed{experiment_seed}"
            cell_dir.mkdir(parents=True, exist_ok=False)
            rows: list[AnchorEpisodeRow] = []
            for index, case_id in enumerate(anchor_ids, start=1):
                print(
                    f"[m={m_max} seed={experiment_seed}] "
                    f"({index}/{len(anchor_ids)}) anchor={case_id}",
                    file=sys.stderr,
                )
                row = run_episode(
                    case_id=case_id,
                    m_max=m_max,
                    experiment_seed=experiment_seed,
                    defender=defender,
                    governance_policy=governance_policy,
                    reference_pool=pools[str(case_id)],
                    raw_path=raw_path,
                    artefact_dir=artefact_dir,
                    episode_dir=cell_dir / f"anchor_{case_id}",
                    q_max=Q_MAX,
                )
                rows.append(row)
            metrics = summarise_cell(
                m_max=m_max,
                experiment_seed=experiment_seed,
                rows=rows,
                q_max=Q_MAX,
            )
            if metrics.m_cap_violations:
                print(
                    f"ERROR: m_cap_violations={metrics.m_cap_violations} "
                    f"for m={m_max} seed={experiment_seed}",
                    file=sys.stderr,
                )
                return 1
            if m_max == 1 and metrics.max_observed_submission_edit_distance > 1:
                print(
                    "ERROR: m=1 observed submission_edit_distance > 1.",
                    file=sys.stderr,
                )
                return 1
            cells.append(metrics)
            (cell_dir / "cell_summary.json").write_text(
                json.dumps(
                    to_jsonable(
                        {
                            "m_max": metrics.m_max,
                            "experiment_seed": metrics.experiment_seed,
                            "n_anchors": metrics.n_anchors,
                            "successes": metrics.successes,
                            "ASR_at_q": {
                                str(k): v for k, v in metrics.asr_at_q.items()
                            },
                            "mean_queries_to_success": metrics.mean_queries_to_success,
                            "median_queries_to_success": (
                                metrics.median_queries_to_success
                            ),
                            "mean_success_submission_edit_distance": (
                                metrics.mean_success_submission_edit_distance
                            ),
                            "mean_episode_cumulative_edits": (
                                metrics.mean_episode_cumulative_edits
                            ),
                            "mean_episode_cumulative_locked_edits": (
                                metrics.mean_episode_cumulative_locked_edits
                            ),
                            "mean_episode_cumulative_dynamic_edits": (
                                metrics.mean_episode_cumulative_dynamic_edits
                            ),
                            "mean_per_submission_edit_distance": (
                                metrics.mean_per_submission_edit_distance
                            ),
                            "mean_per_submission_locked_edits": (
                                metrics.mean_per_submission_locked_edits
                            ),
                            "mean_per_submission_dynamic_edits": (
                                metrics.mean_per_submission_dynamic_edits
                            ),
                            "field_edit_frequency_all": (
                                metrics.field_edit_frequency_all
                            ),
                            "field_edit_frequency_success_candidates": (
                                metrics.field_edit_frequency_success_candidates
                            ),
                            "failure_buckets": metrics.failure_buckets,
                            "failure_reasons": metrics.failure_reasons,
                            "max_observed_submission_edit_distance": (
                                metrics.max_observed_submission_edit_distance
                            ),
                            "m_cap_violations": metrics.m_cap_violations,
                            "per_anchor": [asdict(row) for row in metrics.per_anchor],
                        }
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    if len(cells) != len(m_list) * len(experiment_seeds):
        print(
            f"ERROR: expected {len(m_list)*len(experiment_seeds)} cells, "
            f"got {len(cells)}",
            file=sys.stderr,
        )
        return 1

    cross = cross_seed_summary(cells, q_max=Q_MAX)
    report = format_report(
        cells=cells, metadata=metadata, cross=cross, q_max=Q_MAX
    )
    print(report)

    summary = {
        **metadata,
        "cells": [
            {
                "m": c.m_max,
                "experiment_seed": c.experiment_seed,
                "n_anchors": c.n_anchors,
                "successes": c.successes,
                "ASR_at_1": c.asr_at_q[1],
                "ASR_at_2": c.asr_at_q[2],
                "ASR_at_3": c.asr_at_q[3],
                "ASR_at_4": c.asr_at_q[4],
                "ASR_at_5": c.asr_at_q[5],
                "ASR_at_q": {str(k): v for k, v in c.asr_at_q.items()},
                "mean_queries_to_success": c.mean_queries_to_success,
                "median_queries_to_success": c.median_queries_to_success,
                "mean_success_submission_edit_distance": (
                    c.mean_success_submission_edit_distance
                ),
                "total_episode_cumulative_edits": c.total_episode_cumulative_edits,
                "mean_episode_cumulative_edits": c.mean_episode_cumulative_edits,
                "mean_episode_cumulative_locked_edits": (
                    c.mean_episode_cumulative_locked_edits
                ),
                "mean_episode_cumulative_dynamic_edits": (
                    c.mean_episode_cumulative_dynamic_edits
                ),
                "mean_per_submission_edit_distance": (
                    c.mean_per_submission_edit_distance
                ),
                "mean_per_submission_locked_edits": (
                    c.mean_per_submission_locked_edits
                ),
                "mean_per_submission_dynamic_edits": (
                    c.mean_per_submission_dynamic_edits
                ),
                "field_edit_frequency_all": c.field_edit_frequency_all,
                "field_edit_frequency_success_candidates": (
                    c.field_edit_frequency_success_candidates
                ),
                "failure_buckets": c.failure_buckets,
                "failure_reasons": c.failure_reasons,
                "max_observed_submission_edit_distance": (
                    c.max_observed_submission_edit_distance
                ),
                "m_cap_violations": c.m_cap_violations,
                "per_anchor": [asdict(row) for row in c.per_anchor],
            }
            for c in cells
        ],
        "cross_seed_summary": cross,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "cross_seed_summary.json").write_text(
        json.dumps(to_jsonable(cross), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "calibration_report.txt").write_text(report, encoding="utf-8")
    write_curve_csv(run_dir / "a0_gov_v2_curve.csv", cells, Q_MAX)

    required = (
        "summary.json",
        "cross_seed_summary.json",
        "calibration_report.txt",
        "a0_gov_v2_curve.csv",
        "metadata.json",
        "frozen_anchors.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            print(f"ERROR: missing artefact {name}", file=sys.stderr)
            return 1

    print(f"Artefacts: {run_dir}", file=sys.stderr)
    if args.smoke:
        print(
            "SMOKE COMPLETE — stopping before formal nine-cell grid.",
            file=sys.stderr,
        )
    else:
        print(
            f"FORMAL COMPLETE — {len(cells)} cells finished.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
