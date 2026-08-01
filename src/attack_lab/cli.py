"""Command-line entry point for the attack laboratory."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from attack_lab.a0_random import ConstrainedRandomAttacker
from attack_lab.budget import BudgetSpec
from attack_lab.cases import (
    DEFAULT_RAW_PATH,
    AttackLabCaseError,
    assert_month6_only,
    discover_true_positive_case_ids,
    load_starting_case,
)
from attack_lab.defender import FrozenXGBoostDefender
from attack_lab.domains import (
    AttackLabDomainError,
    build_proposal_domains,
    load_numeric_domains_config,
)
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.governance import CompiledGovernancePolicy, GovernanceError
from attack_lab.human import HumanAttacker
from attack_lab.logger import TrajectoryLogger
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory
from attack_lab.types import EpisodeResult, to_jsonable
from attack_lab.validator import ConstraintValidator
from baf_data.config import FROZEN_CONFIG

DEFAULT_A0_DOMAINS = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "attack_lab_a0_dev_domains.json"
)
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
        choices=("human", "a0"),
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
        type=int,
        default=None,
        help=(
            "Development-only per-attacker Q_max. "
            "Not a final scientific freeze. Defaults to --max-attempts."
        ),
    )
    parser.add_argument(
        "--e-max",
        type=int,
        default=None,
        help=(
            "Development-only per-attacker E_max (cumulative field-edit budget). "
            "Not a final scientific freeze. Defaults to a large dummy value."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (required for attacker a0).",
    )
    parser.add_argument(
        "--a0-domains",
        type=Path,
        default=None,
        help=(
            "Development-only JSON with numeric_domains for A0 sampling. "
            f"Default for a0: {DEFAULT_A0_DOMAINS}"
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
        help="Optional run directory name under 05_outputs/attack_lab/.",
    )
    return parser


def _resolve_episode_controls(args: argparse.Namespace) -> tuple[tuple[str, ...], int]:
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
        raise SystemExit(
            "ERROR: --mutable-fields is required (or supply it via --dev-config). "
            "No final scientific mutable-field policy is assumed."
        )
    if max_attempts is None:
        raise SystemExit(
            "ERROR: --max-attempts is required (or supply it via --dev-config). "
            "No final scientific attempt budget is assumed."
        )
    if max_attempts < 1:
        raise SystemExit("ERROR: --max-attempts must be >= 1.")
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


def _load_a0_domains_payload(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    if args.a0_domains is not None:
        path = args.a0_domains
        return load_numeric_domains_config(path), str(path)
    if args.dev_config is not None:
        payload = json.loads(args.dev_config.read_text(encoding="utf-8"))
        if "numeric_domains" in payload:
            return payload, str(args.dev_config)
    if DEFAULT_A0_DOMAINS.is_file():
        return load_numeric_domains_config(DEFAULT_A0_DOMAINS), str(DEFAULT_A0_DOMAINS)
    raise SystemExit(
        "ERROR: A0 requires a development-only numeric domains configuration "
        "(--a0-domains or numeric_domains in --dev-config)."
    )


def _run_single_episode(
    *,
    case_id: str,
    args: argparse.Namespace,
    mutable_fields: tuple[str, ...],
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
    validator = ConstraintValidator.from_policy(
        governance_policy,
        enabled_action_keys=mutable_fields,
        data_config=FROZEN_CONFIG,
    )
    logger = TrajectoryLogger.create(run_id, parent=parent)

    q_max = int(args.q_max) if args.q_max is not None else int(max_attempts)
    e_max = int(args.e_max) if args.e_max is not None else 1_000_000
    budget = BudgetSpec.development_dummy(q_max=q_max, e_max=e_max)

    manifest: dict[str, Any] = {
        "case_id": starting.case_id,
        "data_split": starting.data_split,
        "defender_name": defender.name,
        "artefact_id": defender.artefact_id,
        "artefact_dir": str(defender.artefact_dir),
        "frozen_threshold": defender.threshold,
        "feedback_mode": args.feedback,
        "mutable_fields": list(mutable_fields),
        "max_attempts": max_attempts,
        "budget_spec": budget.to_dict(),
        "success_rule": (
            "development_smoke_test: initial BLOCK -> valid PASS "
            "under frozen threshold"
        ),
        "attacker": args.attacker,
        "defence": args.defence,
        "not_dissertation_findings": True,
    }

    env = AttackEnvironment(
        starting_case=starting,
        defender=defender,
        validator=validator,
        feedback_policy=FeedbackPolicy(mode=args.feedback),
        max_attempts=max_attempts,
        logger=logger,
        budget=budget,
    )

    print(f"Run output directory: {logger.run_dir}", file=sys.stderr)

    if args.attacker == "human":
        logger.write_manifest(manifest)
        HumanAttacker(env=env, stdin=sys.stdin, stdout=sys.stdout).run()
        # HumanAttacker finalises the episode and writes the summary.
        print(f"Artefacts written under: {logger.run_dir}", file=sys.stderr)
        return env.result()

    if args.attacker == "a0":
        if seed is None:
            raise SystemExit("ERROR: --seed is required for attacker a0.")
        domains_payload, domains_path = _load_a0_domains_payload(args)
        domains = build_proposal_domains(
            mutable_fields,
            categorical_vocabularies=defender.categorical_vocabularies(),
            numeric_domains_config=domains_payload,
            data_config=FROZEN_CONFIG,
            config_path=domains_path,
        )
        manifest.update(
            {
                "seed": seed,
                "a0_domain_label": domains.config_label,
                "a0_domains_path": domains.config_path,
                "a0_domain_sources": {
                    name: domain.source for name, domain in domains.domains.items()
                },
                "a0_independence_rule": (
                    "development_decision: each attempt redraws independently "
                    "from the original starting case; BLOCK feedback is not used "
                    "to improve later proposals"
                ),
            }
        )
        logger.write_manifest(manifest)
        result = ConstrainedRandomAttacker(
            env=env,
            domains=domains,
            seed=seed,
            stdout=sys.stdout,
        ).run()
        print(f"Artefacts written under: {logger.run_dir}", file=sys.stderr)
        return result

    raise SystemExit(f"ERROR: unsupported attacker {args.attacker!r}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
    if args.attacker not in {"human", "a0"}:
        print(f"ERROR: unsupported attacker {args.attacker!r}.", file=sys.stderr)
        return 2
    if args.attacker == "a0" and args.seed is None:
        print("ERROR: --seed is required for attacker a0.", file=sys.stderr)
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
            "mutable_fields": list(mutable_fields),
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
    except (AttackLabCaseError, AttackLabDomainError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
