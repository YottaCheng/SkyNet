#!/usr/bin/env python3
"""A1 frozen-config development evaluation on 100 month-6 blocked-fraud anchors.

Uses FORMAL_A1_MODEL_CONFIG (a1_oneshot_v2_diversified) without modifying A1,
A0, A2, D1, governance, anchors or budgets.

Status: development evidence only — NOT dissertation findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_IMPL / "src"))

from attack_lab.attackers.a1_planner import (  # noqa: E402
    FORMAL_A1_MODEL_CONFIG,
    PROMPT_VERSION_V2,
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
EXPECTED_CONFIG_HASH = (
    "22826044bf81f208eec8bf56fbdd3e648bb6c490581f5a7fa39a4bfa94cfff11"
)
N_DEV_ANCHORS = 100


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


def _load_parsed_plan(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "a1_parsed_plan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _query_efficiency(per_anchor: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [
        int(row["attempts_to_success"])
        for row in per_anchor
        if row.get("attempts_to_success") is not None
    ]
    queries = [int(row["queries_used"]) for row in per_anchor]
    n = len(per_anchor)
    n_success = len(successes)
    return {
        "n_success": n_success,
        "n_fail": n - n_success,
        "mean_queries_used_all": float(np.mean(queries)) if queries else None,
        "mean_first_success_query_among_successes": (
            float(np.mean(successes)) if successes else None
        ),
        "median_first_success_query_among_successes": (
            float(np.median(successes)) if successes else None
        ),
        "mean_queries_used_among_failures": (
            float(
                np.mean(
                    [
                        int(row["queries_used"])
                        for row in per_anchor
                        if row.get("attempts_to_success") is None
                    ]
                )
            )
            if any(row.get("attempts_to_success") is None for row in per_anchor)
            else None
        ),
        "note": (
            "attempts_to_success is censored/absent on failures; "
            "ASR@q is the primary query-efficiency curve."
        ),
    }


def run_batch(
    *,
    anchor_ids: Sequence[str],
    budget: AttackBudget,
    experiment_seed: int,
    run_dir: Path,
    raw_path: Path,
    artefact_dir: Path,
) -> dict[str, Any]:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"BAF raw data not found at {raw_path}. "
            "Mount the external drive before running the A1 development evaluation."
        )

    formal = FORMAL_A1_MODEL_CONFIG
    config_hash = formal.config_hash()
    if formal.prompt_version != PROMPT_VERSION_V2:
        raise RuntimeError("Formal A1 config must use a1_oneshot_v2_diversified.")
    if config_hash != EXPECTED_CONFIG_HASH:
        raise RuntimeError(
            f"Formal A1 config_hash mismatch: got {config_hash}, "
            f"expected {EXPECTED_CONFIG_HASH}. Refusing to run."
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
        label="a1_dev_eval_reference_pool",
        source_path=pool_cfg.source_path,
    )
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=raw_path)
    budget_spec = budget.to_budget_spec(label="a1_dev_eval_budget_interface")

    per_anchor: list[dict[str, Any]] = []
    stop_counts: Counter[str] = Counter()
    parse_statuses: Counter[str] = Counter()
    retry_reasons: Counter[str] = Counter()
    all_field_pairs: Counter[str] = Counter()
    all_strategy_labels: Counter[str] = Counter()

    total_estimated_cost = 0.0
    total_retries = 0
    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0
    total_raw_candidates = 0
    total_frozen_candidates = 0
    total_budget_exceeded = 0
    config_hashes: set[str] = set()
    prompt_hashes: set[str] = set()

    for idx, anchor_id in enumerate(anchor_ids, start=1):
        print(
            f"[A1-dev {idx}/{len(anchor_ids)}] anchor={anchor_id}",
            flush=True,
        )
        starting = load_starting_case(
            int(anchor_id),
            raw_path=raw_path,
            defender=defender,
            artefact_dir=artefact_dir,
        )
        pool = provider.get_pool(str(anchor_id), seed=REFERENCE_POOL_SEED)
        logger = TrajectoryLogger(
            run_dir=run_dir / f"anchor_{anchor_id}",
            run_id=f"a1_dev_{anchor_id}",
        )
        logger.run_dir.mkdir(parents=True, exist_ok=True)
        attacker = OneShotLLMPlanner(
            experiment_seed=experiment_seed,
            reference_pool=pool,
            budget=budget,
            attacker_id="a1",
            model=formal.model,
            temperature=formal.temperature,
            top_p=formal.top_p,
            max_tokens=formal.max_tokens,
            max_parse_retries=formal.max_parse_retries,
            timeout_seconds=formal.timeout_seconds,
            thinking_disabled=formal.thinking_disabled,
            prompt_version=formal.prompt_version,
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
        plan = _load_parsed_plan(logger.run_dir)
        if call is not None:
            parse_statuses[call.parse_status] += 1
            total_estimated_cost += float(call.estimated_cost_usd)
            total_retries += int(call.retry_count)
            total_llm_calls += int(call.llm_call_count)
            total_prompt_tokens += int(call.prompt_tokens)
            total_completion_tokens += int(call.completion_tokens)
            total_latency_ms += float(call.latency_ms)
            total_raw_candidates += int(call.n_raw_candidates)
            total_frozen_candidates += int(call.n_frozen_candidates)
            total_budget_exceeded += int(
                call.governance_reject_counts.get("budget_exceeded", 0)
            )
            config_hashes.add(str(call.config_hash))
            prompt_hashes.add(str(call.prompt_hash))
            for attempt in call.retry_ledger:
                if attempt.retry_reason:
                    retry_reasons[str(attempt.retry_reason)] += 1
            (logger.run_dir / "a1_call_record.json").write_text(
                json.dumps(call.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        field_pairs: list[str] = []
        strategy_labels: list[str] = []
        if plan is not None:
            for item in plan.get("frozen_candidates", []):
                meta = item.get("research_meta") or {}
                edited = meta.get("edited_fields") or list(
                    (item.get("changes") or {}).keys()
                )
                pair = _field_pair_key(edited)
                field_pairs.append(pair)
                all_field_pairs[pair] += 1
                label = meta.get("strategy_label")
                if isinstance(label, str) and label.strip():
                    strategy_labels.append(label.strip())
                    all_strategy_labels[label.strip()] += 1

        n_raw = int(call.n_raw_candidates) if call is not None else 0
        n_frozen = int(call.n_frozen_candidates) if call is not None else 0
        n_budget = (
            int(call.governance_reject_counts.get("budget_exceeded", 0))
            if call is not None
            else 0
        )

        stop_counts[match.stop_reason] += 1
        per_anchor.append(
            {
                "anchor_id": anchor_id,
                "success": match.success,
                "stop_reason": match.stop_reason,
                "queries_used": match.q_used,
                "first_success_query": match.attempts_to_success,
                "attempts_to_success": match.attempts_to_success,
                "invalid_submissions": match.invalid_submissions,
                "api_calls": call.llm_call_count if call is not None else None,
                "retry_count": call.retry_count if call is not None else None,
                "parse_status": call.parse_status if call is not None else None,
                "n_raw_candidates": n_raw,
                "n_frozen_candidates": n_frozen,
                "governance_valid_rate": (n_frozen / n_raw) if n_raw else None,
                "budget_exceeded_count": n_budget,
                "budget_exceeded_rate": (n_budget / n_raw) if n_raw else None,
                "n_distinct_field_pairs": len(set(field_pairs)),
                "field_pairs": field_pairs,
                "n_distinct_strategy_labels": len(set(strategy_labels)),
                "strategy_labels": strategy_labels,
                "governance_reject_counts": (
                    dict(call.governance_reject_counts) if call is not None else {}
                ),
                "prompt_tokens": call.prompt_tokens if call is not None else None,
                "completion_tokens": (
                    call.completion_tokens if call is not None else None
                ),
                "total_tokens": call.total_tokens if call is not None else None,
                "estimated_cost_usd": (
                    call.estimated_cost_usd if call is not None else None
                ),
                "latency_ms": call.latency_ms if call is not None else None,
                "config_hash": call.config_hash if call is not None else None,
                "prompt_hash": call.prompt_hash if call is not None else None,
                "prompt_version": (
                    call.prompt_version if call is not None else formal.prompt_version
                ),
            }
        )

        # Incremental checkpoint for long runs (development evidence only).
        checkpoint = {
            "status": "a1_dev_evaluation_in_progress_not_dissertation_findings",
            "completed_anchors": idx,
            "n_anchors_total": len(anchor_ids),
            "n_success_so_far": sum(1 for row in per_anchor if row["success"]),
            "total_estimated_cost_usd_so_far": total_estimated_cost,
        }
        (run_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    n = len(anchor_ids)
    asr = _asr_curve(
        [row["attempts_to_success"] for row in per_anchor], budget.q_max, n
    )
    parse_ok = int(parse_statuses.get("ok", 0))
    mean_field_pairs = (
        float(np.mean([row["n_distinct_field_pairs"] for row in per_anchor]))
        if per_anchor
        else 0.0
    )
    mean_strategy_labels = (
        float(np.mean([row["n_distinct_strategy_labels"] for row in per_anchor]))
        if per_anchor
        else 0.0
    )
    return {
        "label": "a1_dev_evaluation_n100",
        "status": "a1_development_evidence_only_not_dissertation_findings",
        "n_anchors": n,
        "attack_budget": budget.to_dict(),
        "experiment_seed": experiment_seed,
        "reference_pool_seed": REFERENCE_POOL_SEED,
        "K": 10,
        "governance_version": policy.policy_version,
        "policy_fingerprint": policy.policy_fingerprint,
        "formal_model_config": formal.to_dict(),
        "config_hash": config_hash,
        "expected_config_hash": EXPECTED_CONFIG_HASH,
        "config_hashes_observed": sorted(config_hashes),
        "prompt_hashes_observed": sorted(prompt_hashes),
        "asr_curve": asr,
        "query_efficiency": _query_efficiency(per_anchor),
        "n_success": sum(1 for row in per_anchor if row["success"]),
        "stop_reason_counts": dict(stop_counts),
        "parse_status_counts": dict(parse_statuses),
        "parse_success_rate": parse_ok / n if n else 0.0,
        "api_calls_total": total_llm_calls,
        "api_calls_mean_per_anchor": total_llm_calls / n if n else 0.0,
        "retries_total": total_retries,
        "retry_reason_counts": dict(retry_reasons),
        "governance_valid_rate_overall": (
            total_frozen_candidates / total_raw_candidates
            if total_raw_candidates
            else None
        ),
        "budget_exceeded_rate_overall": (
            total_budget_exceeded / total_raw_candidates
            if total_raw_candidates
            else None
        ),
        "field_pair_diversity": {
            "mean_distinct_field_pairs_per_anchor": mean_field_pairs,
            "global_field_pair_counts": dict(all_field_pairs),
            "n_global_distinct_field_pairs": len(all_field_pairs),
        },
        "strategy_label_diversity": {
            "mean_distinct_strategy_labels_per_anchor": mean_strategy_labels,
            "global_strategy_label_counts": dict(all_strategy_labels),
            "n_global_distinct_strategy_labels": len(all_strategy_labels),
        },
        "token_usage": {
            "prompt_tokens_total": total_prompt_tokens,
            "completion_tokens_total": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "prompt_tokens_mean": total_prompt_tokens / n if n else 0.0,
            "completion_tokens_mean": total_completion_tokens / n if n else 0.0,
        },
        "latency_ms_total": total_latency_ms,
        "latency_ms_mean": total_latency_ms / n if n else 0.0,
        "total_estimated_cost_usd": total_estimated_cost,
        "per_anchor": per_anchor,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--experiment-seed", type=int, default=DEFAULT_EXPERIMENT_SEED)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument("--frozen-anchors", type=Path, default=FROZEN_ANCHORS_SOURCE)
    args = parser.parse_args(argv)

    if int(args.m) != 2 or int(args.q) != 5:
        raise SystemExit("A1 development evaluation is locked to Q=5 and m=2.")

    budget = AttackBudget(q_max=int(args.q), m_max=int(args.m))
    frozen = json.loads(args.frozen_anchors.read_text(encoding="utf-8"))
    anchor_ids = [str(item) for item in frozen["anchor_ids"]]
    if len(anchor_ids) != N_DEV_ANCHORS:
        raise SystemExit(
            f"Expected {N_DEV_ANCHORS} frozen anchors; got {len(anchor_ids)}."
        )

    formal = FORMAL_A1_MODEL_CONFIG
    if formal.config_hash() != EXPECTED_CONFIG_HASH:
        raise SystemExit(
            f"config_hash mismatch before run: {formal.config_hash()} "
            f"!= {EXPECTED_CONFIG_HASH}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"a1_dev_eval_v2_n{len(anchor_ids)}_m{budget.m_max}_q{budget.q_max}_"
        f"seed{args.experiment_seed}_{stamp}"
    )
    run_dir = new_run_directory(
        run_name,
        parent=EXPERIMENTS_ROOT / "a1" / "development",
        stage="experiments",
    )

    (run_dir / "frozen_anchors.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "exact ordered list from A0/A2 frozen month-6 TP/BLOCK "
                    "anchors; no re-sampling"
                ),
                "source_frozen_anchors": str(args.frozen_anchors),
                "experiment_seed": int(args.experiment_seed),
                "n_anchors": len(anchor_ids),
                "anchor_ids": anchor_ids,
                "prompt_version": formal.prompt_version,
                "model_config": formal.to_dict(),
                "config_hash": formal.config_hash(),
                "status": "a1_development_evidence_only_not_dissertation_findings",
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
        anchor_ids=anchor_ids,
        budget=budget,
        experiment_seed=int(args.experiment_seed),
        run_dir=run_dir,
        raw_path=args.raw,
        artefact_dir=args.artefact_dir,
    )
    summary["run_dir"] = str(run_dir)
    summary["frozen_anchors_path"] = str(run_dir / "frozen_anchors.json")

    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "A1 development evaluation n=100",
        "STATUS: development evidence only — NOT dissertation findings",
        f"run_dir: {run_dir}",
        f"budget: Q={budget.q_max}, m={budget.m_max}, K=10",
        f"formal_model_config: {formal.to_dict()}",
        f"config_hash: {summary['config_hash']}",
        f"n_success: {summary['n_success']} / {summary['n_anchors']}",
        f"ASR@1..5: {summary['asr_curve']}",
        f"query_efficiency: {summary['query_efficiency']}",
        f"governance_valid_rate_overall: {summary['governance_valid_rate_overall']}",
        f"budget_exceeded_rate_overall: {summary['budget_exceeded_rate_overall']}",
        f"field_pair_diversity_mean: "
        f"{summary['field_pair_diversity']['mean_distinct_field_pairs_per_anchor']:.3f}",
        f"strategy_label_diversity_mean: "
        f"{summary['strategy_label_diversity']['mean_distinct_strategy_labels_per_anchor']:.3f}",
        f"api_calls_total: {summary['api_calls_total']} "
        f"(mean/anchor={summary['api_calls_mean_per_anchor']:.2f})",
        f"retries_total: {summary['retries_total']} "
        f"reasons={summary['retry_reason_counts']}",
        f"parse_success_rate: {summary['parse_success_rate']:.3f}",
        f"token_usage: {summary['token_usage']}",
        f"latency_ms_mean: {summary['latency_ms_mean']:.1f}",
        f"total_estimated_cost_usd: {summary['total_estimated_cost_usd']:.6f}",
        f"n_prompt_hashes: {len(summary['prompt_hashes_observed'])}",
        "status: a1_development_evidence_only_not_dissertation_findings",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "development_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
