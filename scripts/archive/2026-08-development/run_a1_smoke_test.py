#!/usr/bin/env python3
"""A1 smoke tests: 1-anchor and 5-anchor (not dissertation findings).

Reuses Attack Lab anchors, K=10 reference pools, governance v2, Q=5, m=2.
Does not modify A0, A2, D1, governance, budgets, or existing outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_IMPL / "src"))

from attack_lab.attackers.a1_planner import (  # noqa: E402
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    SUPPORTED_PROMPT_VERSIONS,
    OneShotLLMPlanner,
)
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.cases import DEFAULT_RAW_PATH, load_starting_case  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.feedback import FeedbackPolicy  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.logger import TrajectoryLogger  # noqa: E402
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator  # noqa: E402
from attack_lab.paths import (  # noqa: E402
    DEFAULT_C1_ARTEFACT_DIR,
    EXPERIMENTS_ROOT,
    new_run_directory,
)
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402

FROZEN_ANCHORS_SOURCE = (
    EXPERIMENTS_ROOT
    / "a0"
    / "calibration"
    / "governance_v2"
    / "formal_multiseed_m123_20260804T161724Z"
    / "frozen_anchors.json"
)
GOVERNANCE = _IMPL / "config" / "attacker_compiled_governance.json"
REFERENCE_POOL_SEED = 20260803
DEFAULT_EXPERIMENT_SEED = 20260804


def select_smoke_anchors(
    anchor_ids: Sequence[str],
    *,
    n: int,
    experiment_seed: int,
) -> list[str]:
    """Stable seeded selection from the frozen 100 — no outcome conditioning."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n > len(anchor_ids):
        raise ValueError(f"Requested n={n} exceeds available anchors={len(anchor_ids)}.")
    digest = hashlib.sha256(
        f"{int(experiment_seed)}:a1_smoke_anchor_selection".encode("utf-8")
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    ordered = [str(item) for item in anchor_ids]
    rng.shuffle(ordered)
    return ordered[:n]


def _asr_curve(successes_at: list[int | None], q_max: int, n: int) -> dict[str, float]:
    curve: dict[str, float] = {}
    for q in range(1, q_max + 1):
        hits = sum(
            1 for value in successes_at if value is not None and int(value) <= q
        )
        curve[f"ASR@{q}"] = hits / n if n else 0.0
    return curve


def run_batch(
    *,
    anchor_ids: Sequence[str],
    budget: AttackBudget,
    experiment_seed: int,
    run_dir: Path,
    raw_path: Path,
    artefact_dir: Path,
    label: str,
    prompt_version: str,
) -> dict[str, Any]:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"BAF raw data not found at {raw_path}. "
            "Mount the external drive before running A1 smoke tests."
        )

    policy = CompiledGovernancePolicy.load(GOVERNANCE)
    defender = FrozenXGBoostDefender.from_artefact_dir(artefact_dir)
    pool_cfg = ReferencePoolConfig.load()
    pool_cfg = ReferencePoolConfig(
        K=10,
        seed=REFERENCE_POOL_SEED,
        context_fields=pool_cfg.context_fields,
        action_fields=pool_cfg.action_fields,
        read_only_context_fields=pool_cfg.read_only_context_fields,
        excluded_fields=pool_cfg.excluded_fields,
        label="a1_smoke_reference_pool",
        source_path=pool_cfg.source_path,
    )
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=raw_path)
    budget_spec = budget.to_budget_spec(label="a1_smoke_budget_interface")

    per_anchor: list[dict[str, Any]] = []
    stop_counts: Counter[str] = Counter()
    governance_schema_failures = 0
    parse_statuses: Counter[str] = Counter()
    total_estimated_cost = 0.0
    total_retries = 0

    for anchor_id in anchor_ids:
        starting = load_starting_case(
            int(anchor_id),
            raw_path=raw_path,
            defender=defender,
            artefact_dir=artefact_dir,
        )
        pool = provider.get_pool(str(anchor_id), seed=REFERENCE_POOL_SEED)
        logger = TrajectoryLogger(
            run_dir=run_dir / f"anchor_{anchor_id}",
            run_id=f"a1_{anchor_id}",
        )
        logger.run_dir.mkdir(parents=True, exist_ok=True)
        attacker = OneShotLLMPlanner(
            experiment_seed=experiment_seed,
            reference_pool=pool,
            budget=budget,
            attacker_id="a1",
            prompt_version=prompt_version,
            thinking_disabled=True,
            stdout=sys.stdout,
        )
        match = MatchOrchestrator().run_episode(
            attacker,
            MatchConfig(
                attacker_id="a1",
                anchor=starting,
                policy=policy,
                budget=budget_spec,
                feedback_policy=FeedbackPolicy(mode="label_only"),
                defender=defender,
                seed=experiment_seed,
                enabled_action_keys=None,
                logger=logger,
                reference_pool=pool,
            ),
        )
        (logger.run_dir / "match_result.json").write_text(
            json.dumps(match.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        call = attacker.call_record
        if call is not None:
            parse_statuses[call.parse_status] += 1
            total_estimated_cost += float(call.estimated_cost_usd)
            total_retries += int(call.retry_count)
            (logger.run_dir / "a1_call_record.json").write_text(
                json.dumps(call.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        for step in match.trajectory:
            if not step.validity.is_valid:
                governance_schema_failures += 1
            if step.submitted_edit_cost > budget.m_max:
                governance_schema_failures += 1

        stop_counts[match.stop_reason] += 1
        per_anchor.append(
            {
                "anchor_id": anchor_id,
                "success": match.success,
                "stop_reason": match.stop_reason,
                "queries_used": match.q_used,
                "attempts_to_success": match.attempts_to_success,
                "invalid_submissions": match.invalid_submissions,
                "frozen_count": (
                    call.n_frozen_candidates if call is not None else None
                ),
                "parse_status": call.parse_status if call is not None else None,
                "retry_count": call.retry_count if call is not None else None,
                "prompt_tokens": call.prompt_tokens if call is not None else None,
                "completion_tokens": (
                    call.completion_tokens if call is not None else None
                ),
                "estimated_cost_usd": (
                    call.estimated_cost_usd if call is not None else None
                ),
                "latency_ms": call.latency_ms if call is not None else None,
                "governance_reject_counts": (
                    dict(call.governance_reject_counts) if call is not None else {}
                ),
                "prompt_version": (
                    call.prompt_version if call is not None else prompt_version
                ),
                "prompt_hash": call.prompt_hash if call is not None else None,
                "llm_call_count": call.llm_call_count if call is not None else None,
            }
        )

    n = len(anchor_ids)
    asr = _asr_curve(
        [row["attempts_to_success"] for row in per_anchor], budget.q_max, n
    )
    return {
        "label": label,
        "status": "a1_smoke_not_dissertation_findings",
        "n_anchors": n,
        "attack_budget": budget.to_dict(),
        "experiment_seed": experiment_seed,
        "reference_pool_seed": REFERENCE_POOL_SEED,
        "K": 10,
        "governance_version": policy.policy_version,
        "policy_fingerprint": policy.policy_fingerprint,
        "model": "deepseek-v4-flash",
        "thinking_disabled": True,
        "prompt_version": prompt_version,
        "asr_curve": asr,
        "n_success": sum(1 for row in per_anchor if row["success"]),
        "stop_reason_counts": dict(stop_counts),
        "parse_status_counts": dict(parse_statuses),
        "governance_or_schema_failure_events": governance_schema_failures,
        "total_estimated_cost_usd": total_estimated_cost,
        "total_retries": total_retries,
        "per_anchor": per_anchor,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-anchors",
        type=int,
        required=True,
        choices=(1, 5),
        help="Smoke size only: 1 or 5 anchors. Full 100-anchor run is out of scope.",
    )
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--experiment-seed", type=int, default=DEFAULT_EXPERIMENT_SEED)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument("--frozen-anchors", type=Path, default=FROZEN_ANCHORS_SOURCE)
    parser.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION_V2,
        choices=sorted(SUPPORTED_PROMPT_VERSIONS),
        help=(
            "A1 prompt version. Default is v2 diversified; v1 remains available "
            "and existing v1 smoke directories are never overwritten."
        ),
    )
    args = parser.parse_args(argv)

    if int(args.m) != 2 or int(args.q) != 5:
        raise SystemExit("A1 smoke is locked to Q=5 and m=2 for this stage.")

    budget = AttackBudget(q_max=int(args.q), m_max=int(args.m))
    frozen = json.loads(args.frozen_anchors.read_text(encoding="utf-8"))
    all_ids = [str(item) for item in frozen["anchor_ids"]]
    selected = select_smoke_anchors(
        all_ids, n=int(args.n_anchors), experiment_seed=int(args.experiment_seed)
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_tag = str(args.prompt_version).replace("a1_oneshot_", "")
    run_name = (
        f"a1_smoke_{version_tag}_n{args.n_anchors}_m{budget.m_max}_q{budget.q_max}_"
        f"seed{args.experiment_seed}_{stamp}"
    )
    smoke_parent = (
        EXPERIMENTS_ROOT / "a1" / "smoke_v2"
        if args.prompt_version == PROMPT_VERSION_V2
        else EXPERIMENTS_ROOT / "a1" / "smoke"
    )
    run_dir = new_run_directory(
        run_name,
        parent=smoke_parent,
        stage="experiments",
    )

    (run_dir / "smoke_anchors.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "stable_sha256(experiment_seed:a1_smoke_anchor_selection) "
                    "seeded shuffle of frozen 100 TP/BLOCK anchors"
                ),
                "source_frozen_anchors": str(args.frozen_anchors),
                "experiment_seed": int(args.experiment_seed),
                "n_anchors": int(args.n_anchors),
                "anchor_ids": selected,
                "prompt_version": str(args.prompt_version),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_batch(
        anchor_ids=selected,
        budget=budget,
        experiment_seed=int(args.experiment_seed),
        run_dir=run_dir,
        raw_path=args.raw,
        artefact_dir=args.artefact_dir,
        label=f"a1_smoke_{version_tag}_n{args.n_anchors}",
        prompt_version=str(args.prompt_version),
    )
    summary["run_dir"] = str(run_dir)
    summary["smoke_anchors_path"] = str(run_dir / "smoke_anchors.json")

    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"A1 smoke n={args.n_anchors} prompt={args.prompt_version} "
        "(NOT dissertation findings)",
        f"run_dir: {run_dir}",
        f"budget: Q={budget.q_max}, m={budget.m_max}",
        f"prompt_version: {args.prompt_version}",
        f"anchors: {selected}",
        f"ASR curve: {summary['asr_curve']}",
        f"stop_reasons: {summary['stop_reason_counts']}",
        f"parse_statuses: {summary['parse_status_counts']}",
        f"governance_or_schema_failure_events: "
        f"{summary['governance_or_schema_failure_events']}",
        f"total_estimated_cost_usd: {summary['total_estimated_cost_usd']:.6f}",
        f"total_retries: {summary['total_retries']}",
        "status: a1_smoke_not_dissertation_findings",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "smoke_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
