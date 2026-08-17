#!/usr/bin/env python3
"""A0 (Q,m) budget-curve calibration runner (evaluation only).

Draft calibration under B=(Q,m):
- Q_max fixed
- m swept
- same anchors, reference pools, and experiment seed across m values
- episode RNG via stable_hash(experiment_seed, anchor_id, attacker_id)

Writes under ``05_outputs/experiments/a0/calibration/a0_m_curve_<timestamp>/``.

Calibration only — not dissertation findings.  Does not modify A0 strategy,
governance, reference-pool implementation, D1, or the dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
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
from attack_lab.cases import (  # noqa: E402
    DEFAULT_RAW_PATH,
    discover_true_positive_case_ids,
    load_starting_case,
)
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

# --- draft calibration configuration ---
ATTACKER = "a0"
DEFENCE = "d1"
N_ANCHORS = 100
REFERENCE_POOL_K = 10
Q_MAX = 5
SEED = 20260803
M_LIST: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

A0_SEED_RULE = "stable_hash(experiment_seed, anchor_id, attacker_id)"

FAILURE_BUCKETS = (
    "q_exhausted",
    "m_exceeded",
    "no_feasible_candidate",
    "invalid_candidate",
)

DEFAULT_GOVERNANCE = _IMPL_ROOT / "config" / "attacker_compiled_governance.json"
EXPERIMENT_CALIBRATION_ROOT = EXPERIMENTS_ROOT / "a0" / "calibration"


@dataclass(frozen=True)
class AnchorEpisodeRow:
    anchor_id: str
    m_max: int
    success: bool
    queries_used: int
    edits_used: int
    stop_reason: str
    failure_bucket: str | None
    invalid_submissions: int
    reference_pool_fingerprint: str
    queries_to_success: int | None
    edits_to_success: int | None


@dataclass(frozen=True)
class MCurveMetrics:
    m_max: int
    n_anchors: int
    successes: int
    asr_at_5: float
    asr_at_q: dict[int, float]
    mean_queries_to_success: float | None
    median_queries_to_success: float | None
    mean_edits_before_success: float | None
    total_edits_consumed: int
    failure_reasons: dict[str, int]
    failure_buckets: dict[str, int]
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


def select_frozen_anchors(
    *,
    n_anchors: int,
    seed: int,
    artefact_dir: Path,
) -> list[int]:
    eligible = discover_true_positive_case_ids(artefact_dir)
    if len(eligible) < n_anchors:
        raise SystemExit(
            f"ERROR: need at least {n_anchors} eligible anchors "
            f"(fraud_bool=1 and frozen D1 BLOCK); found {len(eligible)}."
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(eligible, dtype=int), size=n_anchors, replace=False)
    return [int(value) for value in chosen]


def build_pool_config(*, k: int, seed: int) -> ReferencePoolConfig:
    base = ReferencePoolConfig.load()
    return ReferencePoolConfig(
        K=k,
        seed=seed,
        context_fields=base.context_fields,
        action_fields=base.action_fields,
        read_only_context_fields=base.read_only_context_fields,
        excluded_fields=base.excluded_fields,
        label="a0_m_curve_calibration_reference_pool",
        source_path=base.source_path,
    )


def prebuild_reference_pools(
    *,
    provider: ReferencePoolProvider,
    anchor_ids: Sequence[int],
    seed: int,
) -> dict[str, ReferencePool]:
    pools: dict[str, ReferencePool] = {}
    for case_id in anchor_ids:
        key = str(case_id)
        pool = provider.get_pool(key, seed=seed)
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
    if stop_reason == "m_exceeded":
        return "m_exceeded"
    if stop_reason == "q_exhausted":
        return "q_exhausted"
    if invalid_submissions > 0 and scored_defender_queries == 0:
        return "invalid_candidate"
    return stop_reason


def run_episode(
    *,
    case_id: int,
    m_max: int,
    defender: FrozenXGBoostDefender,
    governance_policy: CompiledGovernancePolicy,
    reference_pool: ReferencePool,
    raw_path: Path,
    artefact_dir: Path,
    episode_dir: Path,
    experiment_seed: int,
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
        label="a0_m_curve_calibration_dummy_budget",
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
    qts = int(match.attempts_to_success) if match.success else None
    edits_used = int(match.total_edits_used)
    bucket = classify_failure_bucket(
        success=bool(match.success),
        stop_reason=str(match.stop_reason),
        invalid_submissions=int(match.invalid_submissions),
        scored_defender_queries=int(match.scored_defender_queries),
    )
    return AnchorEpisodeRow(
        anchor_id=match.anchor_id,
        m_max=m_max,
        success=bool(match.success),
        queries_used=int(match.q_used),
        edits_used=edits_used,
        stop_reason=str(match.stop_reason),
        failure_bucket=bucket,
        invalid_submissions=int(match.invalid_submissions),
        reference_pool_fingerprint=reference_pool.pool_fingerprint,
        queries_to_success=qts,
        edits_to_success=edits_used if match.success else None,
    )


def summarise_m(
    *,
    m_max: int,
    rows: Sequence[AnchorEpisodeRow],
    q_max: int,
) -> MCurveMetrics:
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
    succ_e = [
        float(row.edits_to_success)
        for row in rows
        if row.success and row.edits_to_success is not None
    ]
    raw_failures = Counter(row.stop_reason for row in rows if not row.success)
    buckets = {name: 0 for name in FAILURE_BUCKETS}
    other_failures = Counter()
    for row in rows:
        if row.success or row.failure_bucket is None:
            continue
        if row.failure_bucket in buckets:
            buckets[row.failure_bucket] += 1
        else:
            other_failures[row.failure_bucket] += 1
    failure_reasons = dict(sorted(raw_failures.items()))
    if other_failures:
        failure_reasons["other_mapped"] = dict(sorted(other_failures.items()))
    return MCurveMetrics(
        m_max=m_max,
        n_anchors=n,
        successes=successes,
        asr_at_5=asr_at_q.get(q_max, 0.0),
        asr_at_q=asr_at_q,
        mean_queries_to_success=(sum(succ_q) / len(succ_q)) if succ_q else None,
        median_queries_to_success=_median(succ_q) if succ_q else None,
        mean_edits_before_success=(sum(succ_e) / len(succ_e)) if succ_e else None,
        total_edits_consumed=int(sum(row.edits_used for row in rows)),
        failure_reasons=failure_reasons,
        failure_buckets=buckets,
        per_anchor=tuple(rows),
    )


def format_report(
    *,
    metrics_by_m: Sequence[MCurveMetrics],
    metadata: dict[str, Any],
    q_max: int,
) -> str:
    lines = [
        "A0 (Q,m) Budget Curve Calibration (DRAFT)",
        "",
        "NOTE: Calibration experiment only. Not dissertation findings.",
        "",
        f"attacker: {metadata['attacker']}",
        f"defence: {metadata['defence']}",
        f"n_anchors: {metadata['n_anchors']}",
        f"reference_pool_K: {metadata['reference_pool_K']}",
        f"Q_max: {metadata['Q_max']}",
        f"seed: {metadata['seed']}",
        f"m_list: {metadata['m_list']}",
        f"a0_seed_rule: {metadata['a0_seed_rule']}",
        f"timestamp_utc: {metadata['timestamp_utc']}",
        f"commit_hash: {metadata['commit_hash']}",
        f"governance_fingerprint: {metadata['governance_fingerprint']}",
        f"policy_fingerprint: {metadata['policy_fingerprint']}",
        f"reference_pool_fingerprint: {metadata['reference_pool_fingerprint']}",
        "",
    ]
    for metrics in metrics_by_m:
        lines.append(f"m={metrics.m_max}")
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
        if metrics.mean_edits_before_success is None:
            lines.append("  mean edits before success: n/a")
        else:
            lines.append(
                f"  mean edits before success: {metrics.mean_edits_before_success:.2f}"
            )
        lines.append(f"  total edits consumed: {metrics.total_edits_consumed}")
        lines.append("  failure buckets:")
        for name in FAILURE_BUCKETS:
            lines.append(f"    {name}: {metrics.failure_buckets.get(name, 0)}")
        lines.append("  raw stop_reason breakdown:")
        if metrics.failure_reasons:
            for reason, count in metrics.failure_reasons.items():
                lines.append(f"    {reason}: {count}")
        else:
            lines.append("    (none)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_curve_csv(
    path: Path, metrics_by_m: Sequence[MCurveMetrics], q_max: int
) -> None:
    fieldnames = [
        "m",
        "successes",
        *[f"ASR@{q}" for q in range(1, q_max + 1)],
        "mean_queries_to_success",
        "median_queries_to_success",
        "mean_edits_before_success",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_by_m:
            row: dict[str, Any] = {
                "m": metrics.m_max,
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
                "mean_edits_before_success": (
                    ""
                    if metrics.mean_edits_before_success is None
                    else f"{metrics.mean_edits_before_success:.6f}"
                ),
            }
            for q in range(1, q_max + 1):
                row[f"ASR@{q}"] = f"{metrics.asr_at_q[q]:.6f}"
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    raw_path = DEFAULT_RAW_PATH
    artefact_dir = DEFAULT_C1_ARTEFACT_DIR
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    commit = git_commit_hash()

    if not raw_path.is_file():
        print(f"ERROR: raw BAF file not found: {raw_path}", file=sys.stderr)
        return 2
    if not artefact_dir.is_dir():
        print(f"ERROR: D1 artefact dir not found: {artefact_dir}", file=sys.stderr)
        return 1

    try:
        governance_policy = CompiledGovernancePolicy.load(DEFAULT_GOVERNANCE)
    except (OSError, json.JSONDecodeError, GovernanceError) as exc:
        print(f"ERROR: cannot load governance policy: {exc}", file=sys.stderr)
        return 2

    try:
        defender = FrozenXGBoostDefender.from_artefact_dir(artefact_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot load frozen D1: {exc}", file=sys.stderr)
        return 1

    try:
        anchor_ids = select_frozen_anchors(
            n_anchors=N_ANCHORS,
            seed=SEED,
            artefact_dir=artefact_dir,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: anchor selection failed: {exc}", file=sys.stderr)
        return 1

    pool_config = build_pool_config(k=REFERENCE_POOL_K, seed=SEED)
    try:
        provider = ReferencePoolProvider.from_config(pool_config, raw_path=raw_path)
        pools = prebuild_reference_pools(
            provider=provider, anchor_ids=anchor_ids, seed=SEED
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: reference pool setup failed: {exc}", file=sys.stderr)
        return 1

    pool_set_fingerprint = aggregate_pool_fingerprint(pools)
    governance_fingerprint = file_sha256(DEFAULT_GOVERNANCE)
    policy_fingerprint = governance_policy.policy_fingerprint

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"a0_m_curve_{stamp}",
        parent=EXPERIMENT_CALIBRATION_ROOT,
        stage="experiments",
    )
    print(f"run_dir={run_dir}", file=sys.stderr)

    metadata: dict[str, Any] = {
        "label": "a0_m_curve_calibration",
        "status": "draft_calibration_not_dissertation_findings",
        "stage": "experiments",
        "budget_protocol": "B=(Q,m)",
        "attacker": ATTACKER,
        "defence": DEFENCE,
        "n_anchors": len(anchor_ids),
        "reference_pool_K": REFERENCE_POOL_K,
        "Q_max": Q_MAX,
        "seed": SEED,
        "m_list": list(M_LIST),
        "a0_seed_rule": A0_SEED_RULE,
        "reference_pool_seed": SEED,
        "anchor_filter": "fraud_bool==1 AND frozen_D1_BLOCK",
        "frozen_anchor_ids": [str(x) for x in anchor_ids],
        "timestamp_utc": timestamp_utc,
        "commit_hash": commit,
        "governance_fingerprint": governance_fingerprint,
        "governance_source_path": str(DEFAULT_GOVERNANCE),
        "policy_fingerprint": policy_fingerprint,
        "policy_source_sha256": governance_policy.source_sha256,
        "reference_pool_fingerprint": pool_set_fingerprint,
        "defender_artefact_dir": str(artefact_dir),
        "defender_artefact_id": defender.artefact_id,
        "frozen_threshold": defender.threshold,
        "run_dir": str(run_dir),
    }

    (run_dir / "frozen_anchors.json").write_text(
        json.dumps(
            {
                "seed": SEED,
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

    metrics_by_m: list[MCurveMetrics] = []
    for m_max in M_LIST:
        print(f"=== m_max={m_max} ===", file=sys.stderr)
        m_dir = run_dir / f"m{m_max}"
        m_dir.mkdir(parents=True, exist_ok=False)
        rows: list[AnchorEpisodeRow] = []
        for index, case_id in enumerate(anchor_ids, start=1):
            print(
                f"[m={m_max}] ({index}/{len(anchor_ids)}) anchor={case_id}",
                file=sys.stderr,
            )
            row = run_episode(
                case_id=case_id,
                m_max=m_max,
                defender=defender,
                governance_policy=governance_policy,
                reference_pool=pools[str(case_id)],
                raw_path=raw_path,
                artefact_dir=artefact_dir,
                episode_dir=m_dir / f"anchor_{case_id}",
                experiment_seed=SEED,
                q_max=Q_MAX,
            )
            rows.append(row)
        metrics = summarise_m(m_max=m_max, rows=rows, q_max=Q_MAX)
        metrics_by_m.append(metrics)
        (m_dir / "m_summary.json").write_text(
            json.dumps(
                to_jsonable(
                    {
                        "m_max": metrics.m_max,
                        "n_anchors": metrics.n_anchors,
                        "successes": metrics.successes,
                        "ASR_at_q": {
                            str(k): v for k, v in metrics.asr_at_q.items()
                        },
                        "mean_queries_to_success": metrics.mean_queries_to_success,
                        "median_queries_to_success": metrics.median_queries_to_success,
                        "mean_edits_before_success": metrics.mean_edits_before_success,
                        "total_edits_consumed": metrics.total_edits_consumed,
                        "failure_buckets": metrics.failure_buckets,
                        "failure_reasons": metrics.failure_reasons,
                        "per_anchor": [asdict(row) for row in metrics.per_anchor],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    report = format_report(
        metrics_by_m=metrics_by_m, metadata=metadata, q_max=Q_MAX
    )
    print(report)

    summary = {
        **metadata,
        "by_m": [
            {
                "m": m.m_max,
                "n_anchors": m.n_anchors,
                "successes": m.successes,
                "ASR_at_1": m.asr_at_q[1],
                "ASR_at_2": m.asr_at_q[2],
                "ASR_at_3": m.asr_at_q[3],
                "ASR_at_4": m.asr_at_q[4],
                "ASR_at_5": m.asr_at_q[5],
                "ASR_at_q": {str(k): v for k, v in m.asr_at_q.items()},
                "mean_queries_to_success": m.mean_queries_to_success,
                "median_queries_to_success": m.median_queries_to_success,
                "mean_edits_before_success": m.mean_edits_before_success,
                "total_edits_consumed": m.total_edits_consumed,
                "failure_buckets": m.failure_buckets,
                "failure_reasons": m.failure_reasons,
                "per_anchor": [asdict(row) for row in m.per_anchor],
            }
            for m in metrics_by_m
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "calibration_report.txt").write_text(report, encoding="utf-8")
    write_curve_csv(run_dir / "a0_m_curve.csv", metrics_by_m, Q_MAX)

    print(f"Artefacts: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
