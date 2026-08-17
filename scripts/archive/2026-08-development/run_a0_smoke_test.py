#!/usr/bin/env python3
"""Minimal A0 smoke test against frozen D1 (evaluation only).

Pipeline:
  fraud_bool=1 AND D1 BLOCK -> select anchors -> A0 + reference pool -> metrics.

This script does not modify A0, governance, or defence logic.  It only wires
existing attack_lab components for a development smoke evaluation.
"""

from __future__ import annotations

import json
import sys
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
    AttackLabCaseError,
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
    SCRATCH_SMOKE_ROOT,
    new_run_directory,
)
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402

# --- smoke-test configuration (not a scientific freeze) ---
ATTACKER = "a0"
DEFENCE = "d1"
N_ANCHORS = 5
REFERENCE_POOL_K = 1
Q_MAX = 5
E_MAX = 5
SEED = 20260803

DEFAULT_GOVERNANCE = (
    _IMPL_ROOT / "config" / "attacker_compiled_governance.json"
)


@dataclass(frozen=True)
class AnchorSmokeResult:
    anchor_id: str
    success: bool
    queries_used: int
    edits_used: int
    stop_reason: str
    queries_to_success: int | None
    scored_defender_queries: int


def select_blocked_fraud_anchors(
    *,
    n_anchors: int,
    seed: int,
    artefact_dir: Path,
) -> list[int]:
    """Select anchors with fraud_bool=1 and frozen D1 BLOCK (via score CSV)."""
    eligible = discover_true_positive_case_ids(artefact_dir)
    if len(eligible) < n_anchors:
        raise SystemExit(
            f"ERROR: need at least {n_anchors} eligible anchors "
            f"(fraud_bool=1 and frozen D1 BLOCK); found {len(eligible)}."
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(eligible, dtype=int), size=n_anchors, replace=False)
    return [int(value) for value in chosen]


def build_smoke_pool_config(*, k: int, seed: int) -> ReferencePoolConfig:
    """Load experiment field lists; override K/seed for this smoke run only."""
    base = ReferencePoolConfig.load()
    return ReferencePoolConfig(
        K=k,
        seed=seed,
        context_fields=base.context_fields,
        action_fields=base.action_fields,
        read_only_context_fields=base.read_only_context_fields,
        excluded_fields=base.excluded_fields,
        label="a0_smoke_test_reference_pool_config",
        source_path=base.source_path,
    )


def run_one_anchor(
    *,
    case_id: int,
    defender: FrozenXGBoostDefender,
    governance_policy: CompiledGovernancePolicy,
    pool_provider: ReferencePoolProvider,
    raw_path: Path,
    artefact_dir: Path,
    run_parent: Path,
    seed: int,
    q_max: int,
    e_max: int,
) -> AnchorSmokeResult:
    starting = load_starting_case(
        case_id,
        raw_path=raw_path,
        defender=defender,
        artefact_dir=artefact_dir,
    )
    if starting.initial_decision != "BLOCK" or starting.label != 1:
        raise AttackLabCaseError(
            f"Anchor {case_id} failed smoke precondition "
            f"(label={starting.label}, decision={starting.initial_decision})."
        )

    reference_pool = pool_provider.get_pool(starting.case_id, seed=seed)
    if reference_pool.K != REFERENCE_POOL_K:
        raise RuntimeError(
            f"Reference pool K={reference_pool.K}, expected {REFERENCE_POOL_K}."
        )

    budget = BudgetSpec.development_dummy(
        q_max=q_max,
        e_max=e_max,
        label="a0_smoke_test_budget_not_scientific_freeze",
    )
    # stop_on_success defaults True: episode ends on first PASS.
    logger = TrajectoryLogger(
        run_dir=run_parent / f"anchor_{starting.case_id}",
        run_id=f"anchor_{starting.case_id}",
    )
    logger.run_dir.mkdir(parents=True, exist_ok=False)

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
    (logger.run_dir / "match_result.json").write_text(
        json.dumps(match.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AnchorSmokeResult(
        anchor_id=match.anchor_id,
        success=bool(match.success),
        queries_used=int(match.scored_defender_queries),
        edits_used=int(match.e_used),
        stop_reason=str(match.stop_reason),
        queries_to_success=(
            int(match.attempts_to_success) if match.success else None
        ),
        scored_defender_queries=int(match.scored_defender_queries),
    )


def format_report(
    results: Sequence[AnchorSmokeResult],
    *,
    k: int,
    q_max: int,
    e_max: int,
) -> str:
    n = len(results)
    n_success = sum(1 for row in results if row.success)
    asr = (n_success / n) if n else 0.0
    success_q = [
        float(row.queries_to_success)
        for row in results
        if row.success and row.queries_to_success is not None
    ]
    all_q = [float(row.queries_used) for row in results]
    mean_success_q = sum(success_q) / len(success_q) if success_q else float("nan")
    mean_all_q = sum(all_q) / len(all_q) if all_q else float("nan")

    lines = [
        "=== A0 Smoke Test ===",
        f"anchors: {n}",
        f"reference_pool_K: {k}",
        f"Q_max: {q_max}",
        f"E_max: {e_max}",
        "",
        "Results:",
        f"success: {n_success}/{n}",
        f"ASR: {asr:.0%}" if n else "ASR: n/a",
        "",
    ]
    if success_q:
        lines.extend(
            [
                "successful attacks:",
                f"mean queries-to-success: {mean_success_q:.2f}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "successful attacks:",
                "mean queries-to-success: n/a (no successes)",
                "",
            ]
        )
    lines.extend(
        [
            "all episodes:",
            f"mean queries: {mean_all_q:.2f}",
            "",
            "Per-anchor:",
        ]
    )
    for row in results:
        lines.append(
            f"  anchor_id={row.anchor_id} success={row.success} "
            f"queries_used={row.queries_used} edits_used={row.edits_used} "
            f"stop_reason={row.stop_reason}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # fixed smoke configuration; CLI flags intentionally omitted
    raw_path = DEFAULT_RAW_PATH
    artefact_dir = DEFAULT_C1_ARTEFACT_DIR
    governance_path = DEFAULT_GOVERNANCE

    if not raw_path.is_file():
        print(f"ERROR: raw BAF file not found: {raw_path}", file=sys.stderr)
        return 2
    if not artefact_dir.is_dir():
        print(f"ERROR: D1 artefact dir not found: {artefact_dir}", file=sys.stderr)
        return 2
    if DEFENCE != "d1" or ATTACKER != "a0":
        print("ERROR: smoke test is fixed to attacker=a0, defence=d1.", file=sys.stderr)
        return 2

    try:
        governance_policy = CompiledGovernancePolicy.load(governance_path)
    except (OSError, json.JSONDecodeError, GovernanceError) as exc:
        print(f"ERROR: cannot load governance policy: {exc}", file=sys.stderr)
        return 2

    try:
        defender = FrozenXGBoostDefender.from_artefact_dir(artefact_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot load frozen D1: {exc}", file=sys.stderr)
        return 1

    try:
        anchor_ids = select_blocked_fraud_anchors(
            n_anchors=N_ANCHORS,
            seed=SEED,
            artefact_dir=artefact_dir,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: anchor selection failed: {exc}", file=sys.stderr)
        return 1

    pool_config = build_smoke_pool_config(k=REFERENCE_POOL_K, seed=SEED)
    try:
        pool_provider = ReferencePoolProvider.from_config(
            pool_config, raw_path=raw_path
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: reference pool provider failed: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"a0_smoke_K{REFERENCE_POOL_K}_{stamp}",
        parent=SCRATCH_SMOKE_ROOT,
        stage="scratch",
    )

    results: list[AnchorSmokeResult] = []
    for case_id in anchor_ids:
        print(f"Running A0 on anchor {case_id} ...", file=sys.stderr)
        try:
            row = run_one_anchor(
                case_id=case_id,
                defender=defender,
                governance_policy=governance_policy,
                pool_provider=pool_provider,
                raw_path=raw_path,
                artefact_dir=artefact_dir,
                run_parent=run_dir,
                seed=SEED,
                q_max=Q_MAX,
                e_max=E_MAX,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: episode failed for anchor {case_id}: {exc}", file=sys.stderr)
            return 1
        results.append(row)

    report = format_report(
        results, k=REFERENCE_POOL_K, q_max=Q_MAX, e_max=E_MAX
    )
    print(report)

    n_success = sum(1 for row in results if row.success)
    success_q = [
        row.queries_to_success
        for row in results
        if row.success and row.queries_to_success is not None
    ]
    summary: dict[str, Any] = {
        "label": "a0_smoke_test",
        "not_dissertation_findings": True,
        "attacker": ATTACKER,
        "defence": DEFENCE,
        "n_anchors": N_ANCHORS,
        "reference_pool_K": REFERENCE_POOL_K,
        "Q_max": Q_MAX,
        "E_max": E_MAX,
        "seed": SEED,
        "selected_anchor_ids": [str(value) for value in anchor_ids],
        "per_anchor": [asdict(row) for row in results],
        "ASR": n_success / len(results) if results else None,
        "mean_queries_to_success": (
            sum(success_q) / len(success_q) if success_q else None
        ),
        "mean_queries_overall": (
            sum(row.queries_used for row in results) / len(results) if results else None
        ),
        "run_dir": str(run_dir),
    }
    summary_path = run_dir / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = run_dir / "smoke_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Artefacts: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
