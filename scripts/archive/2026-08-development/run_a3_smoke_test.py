#!/usr/bin/env python3
"""A3 smoke tests: 1-anchor and 5-anchor (development diagnostics only).

Uses frozen A3 model config, month-6 blocked-fraud anchors, K=10, Q=5, m=2.
Does not modify A0, A1, A2, D1, governance, budgets, or existing outputs.
Does not run 20- or 100-anchor evaluations.
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

from attack_lab.attackers.a3_agent import (  # noqa: E402
    FORMAL_A3_MODEL_CONFIG,
    PROMPT_VERSION,
    EpisodicLLMAgent,
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
FORBIDDEN_MARKERS = (
    "risk_score",
    "y_score",
    "feature_importance",
    "shap",
    "gradient",
    "fraud_bool",
)


def select_smoke_anchors(
    anchor_ids: Sequence[str],
    *,
    n: int,
    experiment_seed: int,
) -> list[str]:
    digest = hashlib.sha256(
        f"{int(experiment_seed)}:a3_smoke_anchor_selection".encode("utf-8")
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


def _field_pair_key(edited_fields: Sequence[str]) -> str:
    return "|".join(sorted(str(name) for name in edited_fields))


def _scan_prompt_leakage(run_dir: Path) -> list[str]:
    hits: list[str] = []
    for path in run_dir.rglob("a3_prompt_payload.json"):
        blob = path.read_text(encoding="utf-8").lower()
        for marker in FORBIDDEN_MARKERS:
            # Allow listing under explicitly_unavailable values, forbid as keys.
            if f'"{marker}":' in blob.replace(" ", ""):
                hits.append(f"{path}:{marker}")
    return hits


def _adaptive_stats(per_anchor: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strategy_switches = 0
    field_pair_switches = 0
    block_followups = 0
    repeated_candidates = 0
    repeated_strategies = 0
    note_examples: list[str] = []
    for row in per_anchor:
        traj = row.get("trajectory") or []
        seen_changes: set[str] = set()
        seen_strategies: set[str] = set()
        for index, step in enumerate(traj):
            changes_key = json.dumps(step.get("changes") or {}, sort_keys=True)
            strategy = step.get("strategy_label")
            pair = _field_pair_key(step.get("edited_fields") or [])
            if changes_key in seen_changes:
                repeated_candidates += 1
            else:
                seen_changes.add(changes_key)
            if strategy in seen_strategies:
                repeated_strategies += 1
            elif strategy:
                seen_strategies.add(strategy)
            note = step.get("adaptation_note")
            if isinstance(note, str) and note.strip() and len(note_examples) < 8:
                note_examples.append(note.strip()[:120])
            if index == 0:
                continue
            prev = traj[index - 1]
            if prev.get("public_label") != "BLOCK":
                continue
            block_followups += 1
            if strategy and strategy != prev.get("strategy_label"):
                strategy_switches += 1
            prev_pair = _field_pair_key(prev.get("edited_fields") or [])
            if pair and pair != prev_pair:
                field_pair_switches += 1
    return {
        "n_block_followups": block_followups,
        "strategy_switch_rate_after_block": (
            strategy_switches / block_followups if block_followups else None
        ),
        "field_pair_switch_rate_after_block": (
            field_pair_switches / block_followups if block_followups else None
        ),
        "repeated_candidate_events": repeated_candidates,
        "repeated_strategy_events": repeated_strategies,
        "adaptation_note_examples_paraphrase_safe": note_examples,
    }


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
    if not raw_path.exists():
        raise FileNotFoundError(
            f"BAF raw data not found at {raw_path}. Mount the external drive."
        )
    formal = FORMAL_A3_MODEL_CONFIG
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
        label="a3_smoke_reference_pool",
        source_path=pool_cfg.source_path,
    )
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=raw_path)
    budget_spec = budget.to_budget_spec(label="a3_smoke_budget_interface")

    per_anchor: list[dict[str, Any]] = []
    stop_counts: Counter[str] = Counter()
    parse_fail = 0
    gov_fail = 0
    total_llm_calls = 0
    total_retries = 0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0
    total_submissions = 0
    total_local_generation_attempts = 0
    total_local_rejections = 0
    total_local_regenerations = 0
    total_regeneration_exhaustions = 0
    total_env_steps = 0
    leakage_hits: list[str] = []
    config_hashes: set[str] = set()
    prompt_hashes: set[str] = set()

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
            run_id=f"a3_{anchor_id}",
        )
        logger.run_dir.mkdir(parents=True, exist_ok=True)
        attacker = EpisodicLLMAgent(
            experiment_seed=experiment_seed,
            reference_pool=pool,
            budget=budget,
            attacker_id="a3",
            model=formal.model,
            temperature=formal.temperature,
            top_p=formal.top_p,
            max_tokens=formal.max_tokens,
            max_parse_retries=formal.max_parse_retries,
            timeout_seconds=formal.timeout_seconds,
            thinking_disabled=formal.thinking_disabled,
            prompt_version=formal.prompt_version,
            max_local_generation_attempts_per_query=(
                formal.max_local_generation_attempts_per_query
            ),
            stdout=sys.stdout,
        )
        match = MatchOrchestrator().run_episode(
            attacker,
            MatchConfig(
                attacker_id="a3",
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
        leakage_hits.extend(_scan_prompt_leakage(logger.run_dir))

        counters = attacker.aggregate_counters()
        total_llm_calls += int(counters["llm_calls"])
        total_retries += int(counters["parse_retries"])
        total_local_generation_attempts += int(counters["local_generation_attempts"])
        total_local_rejections += int(counters["local_rejections"])
        total_local_regenerations += int(counters["local_regenerations"])
        total_regeneration_exhaustions += int(counters["regeneration_exhaustions"])
        total_env_steps += int(counters["env_step_calls"])
        parse_fail += int(counters["parse_failures"])
        gov_fail += int(counters["governance_failures"])

        trajectory: list[dict[str, Any]] = []
        for record in attacker.query_records:
            total_cost += float(record.estimated_cost_usd)
            total_prompt_tokens += int(record.prompt_tokens)
            total_completion_tokens += int(record.completion_tokens)
            total_latency_ms += float(record.latency_ms)
            config_hashes.add(record.config_hash)
            prompt_hashes.add(record.prompt_hash)
            if record.submitted:
                total_submissions += 1
            trajectory.append(
                {
                    "query_index": record.query_index,
                    "strategy_label": record.strategy_label,
                    "adaptation_note": record.adaptation_note,
                    "changes": dict(record.changes),
                    "edited_fields": list(
                        next(
                            (
                                step.edited_fields
                                for step in attacker.memory_steps
                                if step.query_index == record.query_index
                            ),
                            (),
                        )
                    ),
                    "governance_reject_reason": record.governance_reject_reason,
                    "submitted": record.submitted,
                    "env_step_called": record.env_step_called,
                    "public_label": record.public_label,
                    "llm_call_count": record.llm_call_count,
                    "retry_count": record.retry_count,
                    "parse_status": record.parse_status,
                    "local_generation_attempts": record.local_generation_attempts,
                    "local_rejections": record.local_rejections,
                    "local_regenerations": record.local_regenerations,
                    "regeneration_exhausted": record.regeneration_exhausted,
                }
            )

        assert int(counters["env_step_calls"]) == int(match.q_used)

        (logger.run_dir / "a3_episode_summary.json").write_text(
            json.dumps(
                {
                    "anchor_id": anchor_id,
                    "success": match.success,
                    "stop_reason": match.stop_reason,
                    "trajectory": trajectory,
                    "memory": [item.to_public_dict() for item in attacker.memory_steps],
                    "counters": counters,
                    "q_used": match.q_used,
                    "env_step_calls_equals_q_used": (
                        int(counters["env_step_calls"]) == int(match.q_used)
                    ),
                    "config_hash": attacker.config_hash,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        stop_counts[match.stop_reason] += 1
        per_anchor.append(
            {
                "anchor_id": anchor_id,
                "success": match.success,
                "stop_reason": match.stop_reason,
                "queries_used": match.q_used,
                "attempts_to_success": match.attempts_to_success,
                "first_success_query": match.attempts_to_success,
                "invalid_submissions": match.invalid_submissions,
                "llm_calls": attacker.total_llm_calls,
                "retries": attacker.total_retries,
                "env_steps": attacker.total_env_steps,
                "local_rejections": attacker.total_local_rejections,
                "local_regenerations": attacker.total_local_regenerations,
                "regeneration_exhaustions": attacker.total_regeneration_exhaustions,
                "trajectory": trajectory,
                "config_hash": attacker.config_hash,
            }
        )

    n = len(anchor_ids)
    return {
        "label": label,
        "status": "a3_smoke_development_diagnostics_only_not_dissertation_findings",
        "n_anchors": n,
        "attack_budget": budget.to_dict(),
        "experiment_seed": experiment_seed,
        "reference_pool_seed": REFERENCE_POOL_SEED,
        "K": 10,
        "governance_version": policy.policy_version,
        "policy_fingerprint": policy.policy_fingerprint,
        "formal_model_config": formal.to_dict(),
        "config_hash": formal.config_hash(),
        "config_hashes_observed": sorted(config_hashes),
        "prompt_hashes_observed": sorted(prompt_hashes),
        "prompt_version": PROMPT_VERSION,
        "asr_curve": _asr_curve(
            [row["attempts_to_success"] for row in per_anchor], budget.q_max, n
        ),
        "n_success": sum(1 for row in per_anchor if row["success"]),
        "stop_reason_counts": dict(stop_counts),
        "total_llm_calls": total_llm_calls,
        "llm_calls_mean_per_anchor": total_llm_calls / n if n else 0.0,
        "total_retries": total_retries,
        "parse_failure_events": parse_fail,
        "governance_failure_events": gov_fail,
        "total_submissions": total_submissions,
        "total_env_step_calls": total_env_steps,
        "local_generation_attempts": total_local_generation_attempts,
        "local_rejections": total_local_rejections,
        "local_regenerations": total_local_regenerations,
        "regeneration_exhaustions": total_regeneration_exhaustions,
        "forbidden_leakage_hits": leakage_hits,
        "adaptive_behaviour": _adaptive_stats(per_anchor),
        "token_usage": {
            "prompt_tokens_total": total_prompt_tokens,
            "completion_tokens_total": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "latency_ms_total": total_latency_ms,
        "latency_ms_mean_per_anchor": total_latency_ms / n if n else 0.0,
        "total_estimated_cost_usd": total_cost,
        "per_anchor": per_anchor,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-anchors", type=int, required=True, choices=(1, 5))
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--experiment-seed", type=int, default=DEFAULT_EXPERIMENT_SEED)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument("--frozen-anchors", type=Path, default=FROZEN_ANCHORS_SOURCE)
    args = parser.parse_args(argv)

    if int(args.m) != 2 or int(args.q) != 5:
        raise SystemExit("A3 smoke is locked to Q=5 and m=2.")

    budget = AttackBudget(q_max=int(args.q), m_max=int(args.m))
    frozen = json.loads(args.frozen_anchors.read_text(encoding="utf-8"))
    all_ids = [str(item) for item in frozen["anchor_ids"]]
    selected = select_smoke_anchors(
        all_ids, n=int(args.n_anchors), experiment_seed=int(args.experiment_seed)
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"a3_smoke_n{args.n_anchors}_m{budget.m_max}_q{budget.q_max}_"
        f"seed{args.experiment_seed}_{stamp}"
    )
    run_dir = new_run_directory(
        run_name,
        parent=EXPERIMENTS_ROOT / "a3" / "smoke",
        stage="experiments",
    )
    formal = FORMAL_A3_MODEL_CONFIG
    (run_dir / "smoke_anchors.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "stable_sha256(experiment_seed:a3_smoke_anchor_selection) "
                    "seeded shuffle of frozen 100 TP/BLOCK anchors"
                ),
                "source_frozen_anchors": str(args.frozen_anchors),
                "experiment_seed": int(args.experiment_seed),
                "n_anchors": int(args.n_anchors),
                "anchor_ids": selected,
                "prompt_version": PROMPT_VERSION,
                "model_config": formal.to_dict(),
                "config_hash": formal.config_hash(),
                "status": "a3_smoke_development_diagnostics_only_not_dissertation_findings",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "model_config.json").write_text(
        json.dumps(
            {**formal.to_dict(), "config_hash": formal.config_hash()},
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
        label=f"a3_smoke_n{args.n_anchors}",
    )
    summary["run_dir"] = str(run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"A3 smoke n={args.n_anchors}",
        "STATUS: DEVELOPMENT DIAGNOSTICS ONLY — NOT dissertation findings",
        f"run_dir: {run_dir}",
        f"config_hash: {summary['config_hash']}",
        f"anchors: {selected}",
        f"n_success: {summary['n_success']} / {summary['n_anchors']}",
        f"ASR@1..5: {summary['asr_curve']}",
        f"stop_reasons: {summary['stop_reason_counts']}",
        f"total_llm_calls: {summary['total_llm_calls']} "
        f"(mean/anchor={summary['llm_calls_mean_per_anchor']:.2f})",
        f"retries: {summary['total_retries']}",
        f"parse_failure_events: {summary['parse_failure_events']}",
        f"governance_failure_events: {summary['governance_failure_events']}",
        f"total_submissions/env_steps: {summary['total_submissions']} / "
        f"{summary['total_env_step_calls']}",
        f"local_generation_attempts: {summary['local_generation_attempts']}",
        f"local_rejections: {summary['local_rejections']}",
        f"local_regenerations: {summary['local_regenerations']}",
        f"regeneration_exhaustions: {summary['regeneration_exhaustions']}",
        f"forbidden_leakage_hits: {len(summary['forbidden_leakage_hits'])}",
        f"adaptive_behaviour: {summary['adaptive_behaviour']}",
        f"token_usage: {summary['token_usage']}",
        f"latency_ms_mean_per_anchor: {summary['latency_ms_mean_per_anchor']:.1f}",
        f"total_estimated_cost_usd: {summary['total_estimated_cost_usd']:.6f}",
        "status: a3_smoke_development_diagnostics_only_not_dissertation_findings",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "smoke_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
