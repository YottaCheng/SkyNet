"""Command-line entry point for the attack laboratory."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.cases import (
    DEFAULT_RAW_PATH,
    AttackLabCaseError,
    assert_month6_only,
    discover_true_positive_case_ids,
    load_starting_case,
)
from attack_lab.defender import FrozenXGBoostDefender
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.governance import CompiledGovernancePolicy, GovernanceError
from attack_lab.human import HumanAttacker
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory
from attack_lab.reference_pool import (
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import EpisodeResult, to_jsonable
from attack_lab.validator import ConstraintValidator
from baf_data.config import FROZEN_CONFIG

DEFAULT_COMPILED_GOVERNANCE = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "attacker_compiled_governance.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m attack_lab",
        description=(
            "Development attack laboratory against the frozen D1 "
            "statistical defence (C1 XGBoost)."
        ),
    )
    parser.add_argument(
        "--attacker",
        choices=("human", "a0", "a2"),
        default="human",
        help="Attacker mode (default: human).",
    )
    parser.add_argument(
        "--defence",
        choices=("d1",),
        default="d1",
        help="Defence configuration (default: d1 = frozen C1 XGBoost).",
    )
    parser.add_argument(
        "--feedback",
        choices=("label_only",),
        default="label_only",
        help="Public feedback mode (default: label_only).",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Stable month-6 source row_id (true positive under frozen D1).",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Comma-separated month-6 true-positive case IDs for batch mode.",
    )
    parser.add_argument(
        "--mutable-fields",
        default=None,
        help=(
            "Comma-separated governance action keys enabled for this episode. "
            "They may only restrict, never expand, the compiled policy."
        ),
    )
    parser.add_argument(
        "--governance-policy",
        type=Path,
        default=DEFAULT_COMPILED_GOVERNANCE,
        help=(
            "Compiled months-0-5 governance policy. Attack execution fails "
            "closed if it is absent or invalid."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum valid/invalid attempts for this episode. "
            "Required unless --dev-config supplies it. "
            "Maps to development-dummy Q_max when --q-max is omitted."
        ),
    )
    parser.add_argument(
        "--q-max",
        "--q",
        dest="q_max",
        type=int,
        default=None,
        help=(
            "Development-only per-attacker Q_max (--q alias). "
            "Not a final scientific freeze. Defaults to --max-attempts."
        ),
    )
    parser.add_argument(
        "--m-max",
        "--m",
        dest="m_max",
        type=int,
        default=None,
        help=(
            "Per-candidate max feature edits relative to the original anchor (m). "
            "--m is an alias. Required for attackers a0/a2 unless --dev-config."
        ),
    )
    parser.add_argument(
        "--n-anchors",
        type=int,
        default=None,
        help="Optional batch size hint for pilot runners (CLI metadata).",
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=None,
        help="Alias of --seed for A0/A2 experiment-level seeding.",
    )
    parser.add_argument(
        "--e-max",
        type=int,
        default=None,
        help=(
            "Deprecated alias for --m-max (archived runners only). "
            "No longer a cumulative episode edit budget."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Experiment seed (required for attackers a0/a2). "
            "Episode seed uses stable_hash(seed, anchor_id, attacker_id). "
            "Prefer --experiment-seed as an explicit alias."
        ),
    )
    parser.add_argument(
        "--a0-domains",
        type=Path,
        default=None,
        help=(
            "LEGACY ignored. Official A0 samples from compiled governance; "
            "this flag is accepted only for backward-compatible CLI parsing."
        ),
    )
    parser.add_argument(
        "--dev-config",
        type=Path,
        default=None,
        help=(
            "Optional development-only JSON with keys "
            "'mutable_fields' (list[str]), 'max_attempts' (int), and "
            "optionally 'numeric_domains'."
        ),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to BAF Base.csv (default: dissertation raw path).",
    )
    parser.add_argument(
        "--artefact-dir",
        type=Path,
        default=DEFAULT_C1_ARTEFACT_DIR,
        help="Directory containing frozen C1 pipeline and threshold artefacts.",
    )
    parser.add_argument(
        "--split",
        default="dev_month6",
        help="Data split (only month-6 development is permitted).",
    )
    parser.add_argument(
        "--list-tp-cases",
        action="store_true",
        help="List true-positive case IDs from frozen month-6 scores and exit.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional run directory name under 05_outputs/scratch/debug/ "
            "(exploratory default; formal experiments require stage=experiments)."
        ),
    )
    return parser


def _resolve_m_max(args: argparse.Namespace) -> int | None:
    if getattr(args, "m_max", None) is not None:
        return int(args.m_max)
    if args.dev_config is not None:
        payload = json.loads(args.dev_config.read_text(encoding="utf-8"))
        if "m_max" in payload:
            return int(payload["m_max"])
        if "e_max" in payload:
            # Deprecated: treat archived e_max as per-candidate m_max.
            return int(payload["e_max"])
    return None


def _resolve_episode_controls(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...] | None, int]:
    mutable: list[str] | None = None
    max_attempts: int | None = args.max_attempts
    dev_payload: dict[str, Any] = {}

    if args.dev_config is not None:
        dev_payload = json.loads(args.dev_config.read_text(encoding="utf-8"))
        if "mutable_fields" in dev_payload:
            mutable = list(dev_payload["mutable_fields"])
        if "max_attempts" in dev_payload and max_attempts is None:
            max_attempts = int(dev_payload["max_attempts"])

    if args.mutable_fields is not None:
        mutable = [part.strip() for part in args.mutable_fields.split(",") if part.strip()]

    if not mutable:
        if args.attacker in {"a0", "a2"}:
            # Official A0/A2 default to the full compiled governance action set.
            mutable = None
        else:
            raise SystemExit(
                "ERROR: --mutable-fields is required (or supply it via --dev-config). "
                "No final scientific mutable-field policy is assumed."
            )
    if max_attempts is None and args.q_max is not None:
        max_attempts = int(args.q_max)
    if max_attempts is None:
        raise SystemExit(
            "ERROR: --max-attempts is required (or supply it via --dev-config). "
            "No final scientific attempt budget is assumed."
        )
    if max_attempts < 1:
        raise SystemExit("ERROR: --max-attempts must be >= 1.")
    if mutable is None:
        return None, int(max_attempts)
    return tuple(mutable), int(max_attempts)


def _resolve_case_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.case_ids:
        ids.extend(
            part.strip() for part in args.case_ids.split(",") if part.strip()
        )
    if args.case_id:
        ids.append(str(args.case_id).strip())
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for case_id in ids:
        if case_id not in seen:
            seen.add(case_id)
            ordered.append(case_id)
    return ordered


def _run_single_episode(
    *,
    case_id: str,
    args: argparse.Namespace,
    mutable_fields: tuple[str, ...] | None,
    max_attempts: int,
    defender: FrozenXGBoostDefender,
    governance_policy: CompiledGovernancePolicy,
    run_id: str | None,
    parent: Path | None,
    seed: int | None,
) -> EpisodeResult:
    starting = load_starting_case(
        case_id,
        raw_path=args.raw,
        defender=defender,
        artefact_dir=args.artefact_dir,
    )
    logger = TrajectoryLogger.create(run_id, parent=parent)

    q_max = int(args.q_max) if args.q_max is not None else int(max_attempts)
    m_max = _resolve_m_max(args)
    if m_max is None and getattr(args, "e_max", None) is not None:
        # Archived CLI compatibility: --e-max now means per-candidate m_max.
        m_max = int(args.e_max)
    if args.attacker in {"a0", "a2"} and m_max is None:
        raise SystemExit(
            "ERROR: --m / --m-max is required for attackers a0/a2 "
            "(or supply m_max via --dev-config)."
        )
    if m_max is None:
        # Development default for non-budgeted attackers; not a scientific freeze.
        m_max = 1_000_000
    attack_budget = AttackBudget(q_max=q_max, m_max=int(m_max))
    budget = attack_budget.to_budget_spec(
        label="attack_budget_via_cli_interface"
    )

    enabled_action_keys = (
        None
        if mutable_fields is None
        else mutable_fields
    )
    manifest: dict[str, Any] = {
        "case_id": starting.case_id,
        "data_split": starting.data_split,
        "defender_name": defender.name,
        "artefact_id": defender.artefact_id,
        "artefact_dir": str(defender.artefact_dir),
        "frozen_threshold": defender.threshold,
        "feedback_mode": args.feedback,
        "enabled_action_keys": (
            list(enabled_action_keys)
            if enabled_action_keys is not None
            else list(governance_policy.available_action_keys)
        ),
        "max_attempts": max_attempts,
        "budget_spec": budget.to_dict(),
        "success_rule": (
            "development_smoke_test: initial BLOCK -> valid PASS "
            "under frozen threshold"
        ),
        "attacker": args.attacker,
        "defence": args.defence,
        "policy_fingerprint": governance_policy.policy_fingerprint,
        "not_dissertation_findings": True,
    }

    print(f"Run output directory: {logger.run_dir}", file=sys.stderr)

    if args.attacker == "human":
        if enabled_action_keys is None:
            raise SystemExit(
                "ERROR: human attacker requires explicit --mutable-fields."
            )
        validator = ConstraintValidator.from_policy(
            governance_policy,
            enabled_action_keys=enabled_action_keys,
            data_config=FROZEN_CONFIG,
        )
        env = AttackEnvironment(
            starting_case=starting,
            defender=defender,
            validator=validator,
            feedback_policy=FeedbackPolicy(mode=args.feedback),
            max_attempts=max_attempts,
            logger=logger,
            budget=budget,
        )
        logger.write_manifest(manifest)
        HumanAttacker(env=env, stdin=sys.stdin, stdout=sys.stdout).run()
        print(f"Artefacts written under: {logger.run_dir}", file=sys.stderr)
        return env.result()

    if args.attacker == "a0":
        if seed is None:
            raise SystemExit("ERROR: --seed is required for attacker a0.")
        if args.a0_domains is not None:
            print(
                "WARNING: --a0-domains is legacy and ignored; "
                "official A0 samples from compiled governance.",
                file=sys.stderr,
            )
        pool_config = ReferencePoolConfig.load()
        reference_pool = ReferencePoolProvider.from_config(
            pool_config, raw_path=args.raw
        ).get_pool(starting.case_id)
        assert m_max is not None
        manifest.update(
            {
                "seed": seed,
                "m_max": m_max,
                "a0_sampling": "frozen_qm_reference_pool_sequence",
                "a0_independence_rule": (
                    "up to Q candidates are generated and frozen before any "
                    "D1 feedback; each candidate satisfies edit_distance <= m "
                    "relative to the original anchor; feedback labels never "
                    "alter the frozen sequence; episode_seed = "
                    "stable_hash(experiment_seed, anchor_id, attacker_id)"
                ),
                "reference_pool": {
                    "K": reference_pool.K,
                    "generation_seed": reference_pool.generation_seed,
                    "pool_fingerprint": reference_pool.pool_fingerprint,
                    "config": pool_config.to_dict(),
                },
                "submission_path": (
                    "MatchOrchestrator -> AttackEnvironment -> "
                    "BudgetLedger / validator / D1"
                ),
            }
        )
        logger.write_manifest(manifest)
        match = MatchOrchestrator().run_episode(
            ConstrainedRandomAttacker(
                seed=seed,
                reference_pool=reference_pool,
                m_max=m_max,
                attacker_id="a0",
                stdout=sys.stdout,
            ),
            MatchConfig(
                attacker_id="a0",
                anchor=starting,
                policy=governance_policy,
                budget=budget,
                feedback_policy=FeedbackPolicy(mode=args.feedback),
                defender=defender,
                seed=seed,
                enabled_action_keys=enabled_action_keys,
                logger=logger,
                reference_pool=reference_pool,
            ),
        )
        # Persist the uniform MatchResult alongside the episode artefacts.
        (logger.run_dir / "match_result.json").write_text(
            json.dumps(match.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Artefacts written under: {logger.run_dir}", file=sys.stderr)
        return match.episode

    if args.attacker == "a2":
        if seed is None:
            raise SystemExit(
                "ERROR: --seed / --experiment-seed is required for attacker a2."
            )
        pool_config = ReferencePoolConfig.load()
        reference_pool = ReferencePoolProvider.from_config(
            pool_config, raw_path=args.raw
        ).get_pool(starting.case_id)
        assert m_max is not None
        manifest.update(
            {
                "seed": seed,
                "experiment_seed": seed,
                "m_max": m_max,
                "q_max": q_max,
                "attack_budget": attack_budget.to_dict(),
                "a2_policy": (
                    "constrained_surrogate_guided_sequential_best_first_search_"
                    "with_failure_history_diversification"
                ),
                "a2_status": "mechanism_verification_pilot_not_dissertation_findings",
                "reference_pool": {
                    "K": reference_pool.K,
                    "generation_seed": reference_pool.generation_seed,
                    "pool_fingerprint": reference_pool.pool_fingerprint,
                    "config": pool_config.to_dict(),
                },
                "submission_path": (
                    "MatchOrchestrator -> AttackEnvironment -> "
                    "BudgetLedger / validator / D1"
                ),
            }
        )
        logger.write_manifest(manifest)
        attacker = SurrogateGuidedSearcher(
            budget=attack_budget,
            reference_pool=reference_pool,
            experiment_seed=seed,
            attacker_id="a2",
            stdout=sys.stdout,
        )
        match = MatchOrchestrator().run_episode(
            attacker,
            MatchConfig(
                attacker_id="a2",
                anchor=starting,
                policy=governance_policy,
                budget=budget,
                feedback_policy=FeedbackPolicy(mode=args.feedback),
                defender=defender,
                seed=seed,
                enabled_action_keys=enabled_action_keys,
                logger=logger,
                reference_pool=reference_pool,
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
        if attacker.governance_view is not None:
            (logger.run_dir / "a2_governance_view.json").write_text(
                json.dumps(
                    attacker.governance_view.to_public_dict(),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"Artefacts written under: {logger.run_dir}", file=sys.stderr)
        return match.episode

    raise SystemExit(f"ERROR: unsupported attacker {args.attacker!r}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "experiment_seed", None) is not None and args.seed is None:
        args.seed = int(args.experiment_seed)
    if args.seed is not None and getattr(args, "experiment_seed", None) is None:
        args.experiment_seed = int(args.seed)

    try:
        assert_month6_only(args.split)
    except AttackLabCaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.list_tp_cases:
        ids = discover_true_positive_case_ids(args.artefact_dir)
        for row_id in ids[:50]:
            print(row_id)
        if len(ids) > 50:
            print(f"... ({len(ids)} total)", file=sys.stderr)
        else:
            print(f"({len(ids)} total)", file=sys.stderr)
        return 0

    case_ids = _resolve_case_ids(args)
    if not case_ids:
        print(
            "ERROR: --case-id or --case-ids is required "
            "(or use --list-tp-cases to discover IDs).",
            file=sys.stderr,
        )
        return 2

    mutable_fields, max_attempts = _resolve_episode_controls(args)

    if args.defence != "d1":
        print("ERROR: only defence 'd1' is implemented.", file=sys.stderr)
        return 2
    if args.attacker not in {"human", "a0", "a2"}:
        print(f"ERROR: unsupported attacker {args.attacker!r}.", file=sys.stderr)
        return 2
    if args.attacker in {"a0", "a2"} and args.seed is None:
        print(
            "ERROR: --seed / --experiment-seed is required for attackers a0/a2.",
            file=sys.stderr,
        )
        return 2
    if (
        args.attacker in {"a0", "a2"}
        and _resolve_m_max(args) is None
        and getattr(args, "e_max", None) is None
    ):
        print(
            "ERROR: --m / --m-max is required for attackers a0/a2 "
            "(or supply m_max via --dev-config; deprecated --e-max alias accepted).",
            file=sys.stderr,
        )
        return 2
    if args.attacker == "human" and len(case_ids) > 1:
        print(
            "ERROR: human attacker supports a single --case-id only.",
            file=sys.stderr,
        )
        return 2

    try:
        governance_policy = CompiledGovernancePolicy.load(args.governance_policy)
    except (OSError, json.JSONDecodeError, GovernanceError) as exc:
        print(
            "ERROR: a valid train-months-0-5 compiled governance policy is "
            f"required before any attack can run: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        defender = FrozenXGBoostDefender.from_artefact_dir(args.artefact_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        if len(case_ids) == 1:
            _run_single_episode(
                case_id=case_ids[0],
                args=args,
                mutable_fields=mutable_fields,
                max_attempts=max_attempts,
                defender=defender,
                governance_policy=governance_policy,
                run_id=args.run_id,
                parent=None,
                seed=args.seed,
            )
            return 0

        # Batch mode: one parent directory, one child per case.
        batch_id = args.run_id or (
            f"a0_batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        batch_dir = new_run_directory(batch_id)
        summaries: list[dict[str, Any]] = []
        for index, case_id in enumerate(case_ids):
            # Distinct per-case seeds keep independence while remaining reproducible.
            case_seed = None if args.seed is None else int(args.seed) + index
            result = _run_single_episode(
                case_id=case_id,
                args=args,
                mutable_fields=mutable_fields,
                max_attempts=max_attempts,
                defender=defender,
                governance_policy=governance_policy,
                run_id=f"case_{case_id}",
                parent=batch_dir,
                seed=case_seed,
            )
            summaries.append(
                {
                    "case_id": result.case_id,
                    "success": result.success,
                    "attempts_used": result.attempts_used,
                    "max_attempts": result.max_attempts,
                    "stop_reason": result.stop_reason,
                    "seed": case_seed,
                }
            )

        batch_summary = {
            "label": "development_only_a0_batch_smoke",
            "not_dissertation_findings": True,
            "batch_id": batch_dir.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attacker": args.attacker,
            "defence": args.defence,
            "feedback_mode": args.feedback,
            "enabled_action_keys": (
                list(mutable_fields)
                if mutable_fields is not None
                else list(governance_policy.available_action_keys)
            ),
            "max_attempts": max_attempts,
            "base_seed": args.seed,
            "n_cases": len(case_ids),
            "n_success": sum(1 for row in summaries if row["success"]),
            "cases": summaries,
            "output_dir": str(batch_dir),
            "note": (
                "Development smoke batch only. Aggregate rates here are not "
                "dissertation findings."
            ),
        }
        summary_path = batch_dir / "batch_summary.json"
        summary_path.write_text(
            json.dumps(to_jsonable(batch_summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Batch summary: {summary_path}", file=sys.stderr)
        print(
            f"Batch successes: {batch_summary['n_success']}/{batch_summary['n_cases']} "
            "(development smoke only; not a finding)",
            file=sys.stderr,
        )
        return 0
    except AttackLabCaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
