#!/usr/bin/env python3
"""A2 mechanism-verification smoke + 30-anchor pilot (not dissertation findings).

Does not modify D1, governance, reference pools, or A0. Writes a new output
directory under 05_outputs/experiments/a2/.
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

from attack_lab.attackers.a2_search import SurrogateGuidedSearcher  # noqa: E402
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


def select_pilot_anchors(
    anchor_ids: Sequence[str],
    *,
    n: int,
    experiment_seed: int,
) -> list[str]:
    """Stable seeded selection from the frozen 100 — no A0 outcome conditioning."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n > len(anchor_ids):
        raise ValueError(f"Requested n={n} exceeds available anchors={len(anchor_ids)}.")
    digest = hashlib.sha256(
        f"{int(experiment_seed)}:a2_mech_pilot_anchor_selection".encode("utf-8")
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
) -> dict[str, Any]:
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
        label="a2_mech_pilot_reference_pool",
        source_path=pool_cfg.source_path,
    )
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=raw_path)
    budget_spec = budget.to_budget_spec(label="a2_mech_pilot_budget_interface")

    per_anchor: list[dict[str, Any]] = []
    stop_counts: Counter[str] = Counter()
    static_costs: list[int] = []
    duplicate_flags = 0
    governance_violations = 0
    queries_to_success: list[int] = []

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
            run_id=f"a2_{anchor_id}",
        )
        logger.run_dir.mkdir(parents=True, exist_ok=True)
        attacker = SurrogateGuidedSearcher(
            budget=budget,
            reference_pool=pool,
            experiment_seed=experiment_seed,
            attacker_id="a2",
            stdout=None,
        )
        match = MatchOrchestrator().run_episode(
            attacker,
            MatchConfig(
                attacker_id="a2",
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
        (logger.run_dir / "a2_submission_logs.json").write_text(
            json.dumps(list(attacker.submission_logs), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        hashes = [
            str(step.research_meta.get("candidate_hash", ""))
            for step in match.trajectory
            if step.research_meta
        ]
        if len(hashes) != len(set(hashes)):
            duplicate_flags += 1
        for step in match.trajectory:
            if not step.validity.is_valid and step.public_feedback.label != "INVALID":
                governance_violations += 1
            if step.submitted_edit_cost > budget.m_max:
                governance_violations += 1
            meta = step.research_meta or {}
            if "locked_edit_count" in meta:
                static_costs.append(int(meta["locked_edit_count"]))

        stop_counts[match.stop_reason] += 1
        if match.success and match.attempts_to_success is not None:
            queries_to_success.append(int(match.attempts_to_success))

        per_anchor.append(
            {
                "anchor_id": anchor_id,
                "success": match.success,
                "stop_reason": match.stop_reason,
                "queries_used": match.q_used,
                "attempts_to_success": match.attempts_to_success,
                "invalid_submissions": match.invalid_submissions,
                "first_static_lock_cost": (
                    int(match.trajectory[0].research_meta["locked_edit_count"])
                    if match.trajectory
                    and match.trajectory[0].research_meta.get("locked_edit_count")
                    is not None
                    else None
                ),
                "submission_logs": list(attacker.submission_logs),
            }
        )

    n = len(anchor_ids)
    asr = _asr_curve(
        [row["attempts_to_success"] for row in per_anchor], budget.q_max, n
    )
    summary = {
        "label": label,
        "status": "mechanism_verification_pilot_not_dissertation_findings",
        "n_anchors": n,
        "attack_budget": budget.to_dict(),
        "experiment_seed": experiment_seed,
        "reference_pool_seed": REFERENCE_POOL_SEED,
        "K": 10,
        "governance_version": policy.policy_version,
        "policy_fingerprint": policy.policy_fingerprint,
        "asr_curve": asr,
        "mean_queries_to_success": (
            float(np.mean(queries_to_success)) if queries_to_success else None
        ),
        "n_success": sum(1 for row in per_anchor if row["success"]),
        "stop_reason_counts": dict(stop_counts),
        "static_lock_cost_counts": dict(Counter(static_costs)),
        "duplicate_candidate_episodes": duplicate_flags,
        "governance_violation_events": governance_violations,
        "action_space_exhaustion": int(stop_counts.get("action_space_exhaustion", 0)),
        "q_exhausted": int(stop_counts.get("q_exhausted", 0)),
        "ceiling_near_100pct": bool(asr.get(f"ASR@{budget.q_max}", 0.0) >= 0.99),
        "per_anchor": per_anchor,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--n-anchors", type=int, default=None)
    parser.add_argument("--experiment-seed", type=int, default=DEFAULT_EXPERIMENT_SEED)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument(
        "--frozen-anchors",
        type=Path,
        default=FROZEN_ANCHORS_SOURCE,
    )
    args = parser.parse_args(argv)

    n_default = 5 if args.mode == "smoke" else 30
    n_anchors = int(args.n_anchors) if args.n_anchors is not None else n_default
    budget = AttackBudget(q_max=int(args.q), m_max=int(args.m))

    frozen = json.loads(args.frozen_anchors.read_text(encoding="utf-8"))
    all_ids = [str(item) for item in frozen["anchor_ids"]]
    selected = select_pilot_anchors(
        all_ids, n=n_anchors, experiment_seed=int(args.experiment_seed)
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"a2_mech_{args.mode}_m{budget.m_max}_q{budget.q_max}_n{n_anchors}_{stamp}"
    run_dir = new_run_directory(
        run_name,
        parent=EXPERIMENTS_ROOT / "a2" / "mechanism_pilot",
        stage="experiments",
    )

    (run_dir / "pilot_anchors.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "stable_sha256(experiment_seed:a2_mech_pilot_anchor_selection) "
                    "seeded shuffle of frozen 100 TP/BLOCK anchors; "
                    "no A0 outcome conditioning"
                ),
                "source_frozen_anchors": str(args.frozen_anchors),
                "experiment_seed": int(args.experiment_seed),
                "n_anchors": n_anchors,
                "anchor_ids": selected,
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
        label=f"a2_mech_{args.mode}",
    )
    summary["run_dir"] = str(run_dir)
    summary["pilot_anchors_path"] = str(run_dir / "pilot_anchors.json")

    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"A2 mechanism {args.mode} (NOT dissertation findings)",
        f"run_dir: {run_dir}",
        f"budget: Q={budget.q_max}, m={budget.m_max}",
        f"n_anchors: {n_anchors}",
        f"experiment_seed: {args.experiment_seed}",
        f"ASR curve: {summary['asr_curve']}",
        f"mean_queries_to_success: {summary['mean_queries_to_success']}",
        f"stop_reasons: {summary['stop_reason_counts']}",
        f"static_lock_cost_counts: {summary['static_lock_cost_counts']}",
        f"action_space_exhaustion: {summary['action_space_exhaustion']}",
        f"duplicate_candidate_episodes: {summary['duplicate_candidate_episodes']}",
        f"governance_violation_events: {summary['governance_violation_events']}",
        f"ceiling_near_100pct: {summary['ceiling_near_100pct']}",
        "status: mechanism_verification_pilot_not_dissertation_findings",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "pilot_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
