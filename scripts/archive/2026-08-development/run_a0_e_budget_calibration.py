#!/usr/bin/env python3
"""A0 E-budget calibration runner (evaluation only).

Sweeps E_max while holding anchors, reference-pool rules, Q_max and seed fixed.
Does not modify A0, governance, budget accounting, or the orchestrator.
"""

from __future__ import annotations

import csv
import json
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
    SCRATCH_CALIBRATION_ROOT,
    new_run_directory,
)
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePool,
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402

# --- calibration configuration (development only; not a scientific freeze) ---
ATTACKER = "a0"
DEFENCE = "d1"
N_ANCHORS = 20
REFERENCE_POOL_K = 10
Q_MAX = 5
SEED = 20260803
E_LIST: tuple[int, ...] = (1, 2, 3, 4, 5, 7, 9)

DEFAULT_GOVERNANCE = _IMPL_ROOT / "config" / "attacker_compiled_governance.json"
CALIBRATION_ROOT = SCRATCH_CALIBRATION_ROOT


@dataclass(frozen=True)
class AnchorEpisodeRow:
    anchor_id: str
    e_max: int
    success: bool
    queries_used: int
    edits_used: int
    stop_reason: str
    reference_pool_fingerprint: str
    queries_to_success: int | None
    edits_to_success: int | None


@dataclass(frozen=True)
class EBudgetMetrics:
    e_max: int
    n_anchors: int
    successes: int
    asr_at_5: float
    asr_at_q: dict[int, float]
    mean_queries_to_success: float | None
    median_queries_to_success: float | None
    mean_edits_before_success: float | None
    failure_reasons: dict[str, int]
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


def select_frozen_anchors(
    *,
    n_anchors: int,
    seed: int,
    artefact_dir: Path,
) -> list[int]:
    """Freeze one fraud+BLOCK anchor set shared across all E values."""
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
        label="a0_e_budget_calibration_reference_pool",
        source_path=base.source_path,
    )


def prebuild_reference_pools(
    *,
    provider: ReferencePoolProvider,
    anchor_ids: Sequence[int],
    seed: int,
) -> dict[str, ReferencePool]:
    """Build once per anchor so every E reuses the identical pool."""
    pools: dict[str, ReferencePool] = {}
    for case_id in anchor_ids:
        key = str(case_id)
        pool = provider.get_pool(key, seed=seed)
        if pool.K != REFERENCE_POOL_K:
            raise RuntimeError(f"Expected K={REFERENCE_POOL_K}, got {pool.K}.")
        pools[key] = pool
    return pools


def run_episode(
    *,
    case_id: int,
    e_max: int,
    defender: FrozenXGBoostDefender,
    governance_policy: CompiledGovernancePolicy,
    reference_pool: ReferencePool,
    raw_path: Path,
    artefact_dir: Path,
    episode_dir: Path,
    seed: int,
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
        e_max=e_max,
        label="a0_e_budget_calibration_not_scientific_freeze",
    )
    logger = TrajectoryLogger(
        run_dir=episode_dir,
        run_id=episode_dir.name,
    )
    episode_dir.mkdir(parents=True, exist_ok=False)

    attacker = ConstrainedRandomAttacker(
        seed=seed,
        reference_pool=reference_pool,
        m_max=e_max,
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
            seed=seed,
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
    return AnchorEpisodeRow(
        anchor_id=match.anchor_id,
        e_max=e_max,
        success=bool(match.success),
        queries_used=int(match.scored_defender_queries),
        edits_used=int(match.e_used),
        stop_reason=str(match.stop_reason),
        reference_pool_fingerprint=reference_pool.pool_fingerprint,
        queries_to_success=qts,
        edits_to_success=int(match.e_used) if match.success else None,
    )


def summarise_e(
    *,
    e_max: int,
    rows: Sequence[AnchorEpisodeRow],
    q_max: int,
) -> EBudgetMetrics:
    n = len(rows)
    successes = sum(1 for row in rows if row.success)
    asr_at_q: dict[int, float] = {}
    for q in range(1, q_max + 1):
        ok = sum(
            1
            for row in rows
            if row.success
            and row.queries_to_success is not None
            and row.queries_to_success <= q
        )
        asr_at_q[q] = ok / n if n else 0.0

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
    failures = Counter(
        row.stop_reason for row in rows if not row.success
    )
    return EBudgetMetrics(
        e_max=e_max,
        n_anchors=n,
        successes=successes,
        asr_at_5=asr_at_q.get(q_max, 0.0),
        asr_at_q=asr_at_q,
        mean_queries_to_success=(sum(succ_q) / len(succ_q)) if succ_q else None,
        median_queries_to_success=_median(succ_q) if succ_q else None,
        mean_edits_before_success=(sum(succ_e) / len(succ_e)) if succ_e else None,
        failure_reasons=dict(sorted(failures.items())),
        per_anchor=tuple(rows),
    )


def format_report(
    *,
    metrics_by_e: Sequence[EBudgetMetrics],
    anchor_ids: Sequence[int],
    seed: int,
    k: int,
    q_max: int,
) -> str:
    lines = [
        "A0 E Budget Calibration",
        "",
        f"attacker: {ATTACKER}",
        f"defence: {DEFENCE}",
        f"n_anchors: {len(anchor_ids)}",
        f"reference_pool_K: {k}",
        f"Q_max: {q_max}",
        f"seed: {seed}",
        f"E_LIST: {list(E_LIST)}",
        f"frozen_anchors: {', '.join(str(x) for x in anchor_ids)}",
        "",
        "NOTE: Development calibration only; not dissertation findings.",
        "",
    ]
    for metrics in metrics_by_e:
        lines.append(f"E={metrics.e_max}")
        lines.append("ASR:")
        lines.append(f"  successes: {metrics.successes}/{metrics.n_anchors}")
        lines.append(f"  ASR@5: {metrics.asr_at_5:.2%}")
        for q in range(1, q_max + 1):
            lines.append(f"  ASR@{q}: {metrics.asr_at_q[q]:.2%}")
        lines.append("Queries:")
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
        if metrics.failure_reasons:
            lines.append("  failure reasons:")
            for reason, count in metrics.failure_reasons.items():
                lines.append(f"    {reason}: {count}")
        else:
            lines.append("  failure reasons: (none)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_curve_csv(path: Path, metrics_by_e: Sequence[EBudgetMetrics], q_max: int) -> None:
    fieldnames = [
        "E_max",
        "n_anchors",
        "successes",
        "ASR_at_5",
        *[f"ASR_at_{q}" for q in range(1, q_max + 1)],
        "mean_queries_to_success",
        "median_queries_to_success",
        "mean_edits_before_success",
        "failure_reasons_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_by_e:
            row: dict[str, Any] = {
                "E_max": metrics.e_max,
                "n_anchors": metrics.n_anchors,
                "successes": metrics.successes,
                "ASR_at_5": f"{metrics.asr_at_5:.6f}",
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
                "failure_reasons_json": json.dumps(
                    metrics.failure_reasons, sort_keys=True
                ),
            }
            for q in range(1, q_max + 1):
                row[f"ASR_at_{q}"] = f"{metrics.asr_at_q[q]:.6f}"
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    raw_path = DEFAULT_RAW_PATH
    artefact_dir = DEFAULT_C1_ARTEFACT_DIR

    if DEFENCE != "d1" or ATTACKER != "a0":
        print("ERROR: calibration runner is fixed to a0 vs d1.", file=sys.stderr)
        return 2
    if not raw_path.is_file():
        print(f"ERROR: raw BAF file not found: {raw_path}", file=sys.stderr)
        return 2
    if not artefact_dir.is_dir():
        print(f"ERROR: D1 artefact dir not found: {artefact_dir}", file=sys.stderr)
        return 2

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

    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"a0_e_budget_{stamp}",
        parent=CALIBRATION_ROOT,
        stage="scratch",
    )
    print(f"run_dir={run_dir}", file=sys.stderr)

    (run_dir / "frozen_anchors.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_anchors": len(anchor_ids),
                "anchor_ids": [str(x) for x in anchor_ids],
                "filter": "fraud_bool==1 AND frozen_D1_BLOCK",
                "reference_pool_K": REFERENCE_POOL_K,
                "Q_max": Q_MAX,
                "E_LIST": list(E_LIST),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_by_e: list[EBudgetMetrics] = []
    for e_max in E_LIST:
        print(f"=== E_max={e_max} ===", file=sys.stderr)
        e_dir = run_dir / f"E{e_max}"
        e_dir.mkdir(parents=True, exist_ok=False)
        rows: list[AnchorEpisodeRow] = []
        for index, case_id in enumerate(anchor_ids, start=1):
            print(
                f"[E={e_max}] ({index}/{len(anchor_ids)}) anchor={case_id}",
                file=sys.stderr,
            )
            row = run_episode(
                case_id=case_id,
                e_max=e_max,
                defender=defender,
                governance_policy=governance_policy,
                reference_pool=pools[str(case_id)],
                raw_path=raw_path,
                artefact_dir=artefact_dir,
                episode_dir=e_dir / f"anchor_{case_id}",
                seed=SEED,
                q_max=Q_MAX,
            )
            rows.append(row)
        metrics = summarise_e(e_max=e_max, rows=rows, q_max=Q_MAX)
        metrics_by_e.append(metrics)
        (e_dir / "e_summary.json").write_text(
            json.dumps(
                to_jsonable(
                    {
                        "e_max": metrics.e_max,
                        "n_anchors": metrics.n_anchors,
                        "successes": metrics.successes,
                        "ASR_at_5": metrics.asr_at_5,
                        "ASR_at_q": {
                            str(k): v for k, v in metrics.asr_at_q.items()
                        },
                        "mean_queries_to_success": metrics.mean_queries_to_success,
                        "median_queries_to_success": metrics.median_queries_to_success,
                        "mean_edits_before_success": metrics.mean_edits_before_success,
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
        metrics_by_e=metrics_by_e,
        anchor_ids=anchor_ids,
        seed=SEED,
        k=REFERENCE_POOL_K,
        q_max=Q_MAX,
    )
    print(report)

    summary = {
        "label": "a0_e_budget_calibration",
        "not_dissertation_findings": True,
        "development_calibration_only": True,
        "attacker": ATTACKER,
        "defence": DEFENCE,
        "n_anchors": len(anchor_ids),
        "reference_pool_K": REFERENCE_POOL_K,
        "Q_max": Q_MAX,
        "seed": SEED,
        "E_LIST": list(E_LIST),
        "frozen_anchor_ids": [str(x) for x in anchor_ids],
        "by_E": [
            {
                "e_max": m.e_max,
                "n_anchors": m.n_anchors,
                "successes": m.successes,
                "ASR_at_5": m.asr_at_5,
                "ASR_at_q": {str(k): v for k, v in m.asr_at_q.items()},
                "mean_queries_to_success": m.mean_queries_to_success,
                "median_queries_to_success": m.median_queries_to_success,
                "mean_edits_before_success": m.mean_edits_before_success,
                "failure_reasons": m.failure_reasons,
                "per_anchor": [asdict(row) for row in m.per_anchor],
            }
            for m in metrics_by_e
        ],
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "calibration_report.txt").write_text(report, encoding="utf-8")
    write_curve_csv(run_dir / "a0_e_budget_curve.csv", metrics_by_e, Q_MAX)

    print(f"Artefacts: {run_dir}", file=sys.stderr)
    print("该结果仅用于 development calibration，不作为论文结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
