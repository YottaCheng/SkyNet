#!/usr/bin/env python3
"""Controlled A3 prompt-development ablation (P0/P1/P2) on a fixed month-6 set.

Development / prompt-selection only. Not dissertation findings.
Does not tune D1, governance, Q/m/K, sampling, or month 7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "04_implementation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attack_lab.attackers.a1_planner import LLMCompletionClient  # noqa: E402
from attack_lab.attackers.a3_agent import (  # noqa: E402
    FORMAL_A3_MODEL_CONFIG,
    PROMPT_VARIANT_LABELS,
    PROMPT_VERSION,
    PROMPT_VERSION_P1_COMPACT,
    PROMPT_VERSION_P1_RANKED_PORTFOLIO,
    PROMPT_VERSION_P2_NOVELTY,
    RANKED_PORTFOLIO_CAP,
    A3ModelConfig,
    EpisodicLLMAgent,
    build_a3_prompt_payload,
)
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.candidate_identity import (  # noqa: E402
    canonical_candidate_fingerprint,
)
from attack_lab.cases import (  # noqa: E402
    discover_true_positive_case_ids,
    load_starting_case,
)
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.environment import AttackEnvironment  # noqa: E402
from attack_lab.feedback import FeedbackPolicy  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.logger import TrajectoryLogger  # noqa: E402
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator  # noqa: E402
from attack_lab.outbound_payload import (  # noqa: E402
    OUTBOUND_POLICY_VERSION,
    audit_outbound_payload,
    temporary_episode_id,
)
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory  # noqa: E402
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402
from attack_lab.validator import ConstraintValidator  # noqa: E402
from baf_data.config import FROZEN_CONFIG  # noqa: E402

DEFAULT_RAW_PATH = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")
GOVERNANCE = (
    ROOT / "04_implementation" / "config" / "attacker_compiled_governance.json"
)
FROZEN_COMPARISON_ANCHORS = (
    ROOT
    / "05_outputs"
    / "experiments"
    / "a0"
    / "calibration"
    / "governance_v2"
    / "formal_multiseed_m123_20260804T161724Z"
    / "frozen_anchors.json"
)
EXPERIMENTS_ROOT = ROOT / "05_outputs" / "experiments"
DEFAULT_EXPERIMENT_SEED = 20260804
REFERENCE_POOL_SEED = 20260804
SELECTION_DIGEST_LABEL = "a3_prompt_dev_anchor_selection"
DEFAULT_N_ANCHORS = 25
ABLATION_MAX_LOCAL_GENERATION_ATTEMPTS = 1

PREDEFINED_ENGINEERING_CRITERIA = {
    "projected_duplicate_submissions": 0,
    "local_generation_exhaustion_rate_max": 0.0,
    "valid_submission_rate_min": 0.98,
    "parse_failure_events": 0,
    "information_leakage_hits": 0,
    "governance_invalid_submissions": 0,
    "post_block_projected_novelty_rate_min": 1.0,
    "prompt_efficiency_rule": (
        "Token cost is a final tie-break after attack performance and query efficiency."
    ),
    "selection_rule": (
        "Reject every condition failing an integrity gate; among eligible "
        "conditions maximise ASR@5, then ASR@4..1, then query efficiency, "
        "valid submission rate and lower token cost."
    ),
}

VARIANT_PROMPT_VERSIONS = {
    "P0": PROMPT_VERSION,
    "P1": PROMPT_VERSION_P1_COMPACT,
    "P2": PROMPT_VERSION_P2_NOVELTY,
    "RP1": PROMPT_VERSION_P1_RANKED_PORTFOLIO,
}

FORBIDDEN_MARKERS = (
    "risk_score",
    "y_score",
    "feature_importance",
    "shap",
    "gradient",
    "fraud_bool",
)


def select_prompt_dev_anchors(
    *,
    eligible_ids: Sequence[int],
    frozen_comparison_ids: Sequence[str],
    n: int,
    experiment_seed: int,
) -> list[str]:
    """Select N month-6 TP/BLOCK anchors disjoint from the frozen comparison 100."""
    frozen = {str(x) for x in frozen_comparison_ids}
    remainder = [str(x) for x in eligible_ids if str(x) not in frozen]
    if len(remainder) < n:
        raise RuntimeError(
            f"Need {n} disjoint prompt-dev anchors; only {len(remainder)} remain "
            f"outside the frozen comparison set of {len(frozen)}."
        )
    digest = hashlib.sha256(
        f"{int(experiment_seed)}:{SELECTION_DIGEST_LABEL}".encode("utf-8")
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    ordered = list(remainder)
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


def _scan_prompt_leakage(run_dir: Path) -> list[str]:
    hits: list[str] = []
    for path in run_dir.rglob("a3_prompt_payload.json"):
        blob = path.read_text(encoding="utf-8").lower()
        for marker in FORBIDDEN_MARKERS:
            if f'"{marker}":' in blob.replace(" ", ""):
                hits.append(f"{path}:{marker}")
    return hits


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_provenance(artefact_dir: Path) -> dict[str, Any]:
    implementation_root = ROOT / "04_implementation"
    snapshot_files = sorted((SRC / "attack_lab").glob("*.py")) + [
        Path(__file__).resolve(),
        GOVERNANCE,
        implementation_root / "config" / "reference_pool_config.json",
        implementation_root
        / "config"
        / "constraint_profiles"
        / "identity_composition_proxy_v1.json",
        implementation_root / "requirements.txt",
    ]
    snapshot_digest = hashlib.sha256()
    snapshot_entries: list[dict[str, str]] = []
    for path in snapshot_files:
        file_hash = _sha256_file(path)
        relative = str(path.relative_to(ROOT))
        snapshot_entries.append({"path": relative, "sha256": file_hash})
        snapshot_digest.update(relative.encode("utf-8"))
        snapshot_digest.update(b"\0")
        snapshot_digest.update(file_hash.encode("ascii"))
        snapshot_digest.update(b"\n")

    def git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(implementation_root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    artefact_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(artefact_dir.iterdir())
        if path.is_file()
    }
    git_status = git_output("status", "--short")
    return {
        "git_head": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status_short": git_status.splitlines() if git_status else [],
        "implementation_snapshot_sha256": snapshot_digest.hexdigest(),
        "implementation_snapshot_files": snapshot_entries,
        "raw_dataset_expected_sha256": FROZEN_CONFIG.expected_sha256,
        "frozen_d1_artefact_dir": str(artefact_dir),
        "frozen_d1_artefact_file_sha256": artefact_hashes,
    }


def _temperature_slug(temperature: float) -> str:
    return f"{float(temperature):g}".replace("-", "m").replace(".", "p")


def _parse_temperatures(raw: str) -> list[float]:
    temperatures: list[float] = []
    for item in raw.split(","):
        value = float(item.strip())
        if not math.isfinite(value) or not (0.0 <= value <= 2.0):
            raise ValueError("Temperatures must be finite values in [0, 2].")
        if value not in temperatures:
            temperatures.append(value)
    if not temperatures:
        raise ValueError("At least one temperature is required.")
    return temperatures


def _field_pair_key(edited_fields: Sequence[str]) -> str:
    return "|".join(sorted(str(name) for name in edited_fields))


def _adaptive_stats(per_anchor: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strategy_switches = 0
    field_pair_switches = 0
    block_followups = 0
    for row in per_anchor:
        traj = row.get("trajectory") or []
        for index, step in enumerate(traj):
            if index == 0:
                continue
            prev = traj[index - 1]
            if prev.get("public_label") != "BLOCK":
                continue
            block_followups += 1
            strategy = step.get("strategy_label")
            if strategy and strategy != prev.get("strategy_label"):
                strategy_switches += 1
            pair = _field_pair_key(step.get("edited_fields") or [])
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
    }


def _local_rejection_breakdown(per_anchor: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in per_anchor:
        for step in row.get("trajectory") or []:
            for rec in step.get("local_generation_records") or []:
                reason = rec.get("local_rejection_reason")
                if reason:
                    counts[str(reason)] += 1
    return dict(counts)


def prepare_anchor_set(
    *,
    n_anchors: int,
    experiment_seed: int,
    frozen_anchors_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    frozen = json.loads(frozen_anchors_path.read_text(encoding="utf-8"))
    eligible = discover_true_positive_case_ids()
    selected = select_prompt_dev_anchors(
        eligible_ids=eligible,
        frozen_comparison_ids=frozen["anchor_ids"],
        n=n_anchors,
        experiment_seed=experiment_seed,
    )
    overlap = sorted(set(selected) & {str(x) for x in frozen["anchor_ids"]})
    if overlap:
        raise RuntimeError(f"Prompt-dev set overlaps frozen comparison anchors: {overlap}")
    payload = {
        "selection_rule": (
            f"stable_sha256(experiment_seed:{SELECTION_DIGEST_LABEL}) seeded "
            "shuffle of month-6 TP/BLOCK anchors EXCLUDING the frozen A0–A3 "
            "comparison set of 100"
        ),
        "experiment_seed": int(experiment_seed),
        "n_anchors": int(n_anchors),
        "n_eligible_month6_tp_block": len(eligible),
        "n_frozen_comparison_reserved": len(frozen["anchor_ids"]),
        "n_remainder_pool": len(eligible) - len(frozen["anchor_ids"]),
        "source_frozen_comparison_anchors": str(frozen_anchors_path),
        "anchor_ids": selected,
        "disjoint_from_frozen_comparison": True,
        "status": "a3_prompt_development_set_only_not_dissertation_findings",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing prompt-development anchor set: {out_path}"
        )
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def preflight_outbound_payloads(
    *,
    anchor_ids: Sequence[str],
    budget: AttackBudget,
    experiment_seed: int,
    raw_path: Path,
    artefact_dir: Path,
    temperatures: Sequence[float],
    prompt_version: str = PROMPT_VERSION_P1_COMPACT,
    max_local_generation_attempts_per_query: int = (
        ABLATION_MAX_LOCAL_GENERATION_ATTEMPTS
    ),
) -> dict[str, Any]:
    """Build and audit all fixed-anchor first-call payloads before any API call."""

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
        label="a3_prompt_dev_reference_pool",
        source_path=pool_cfg.source_path,
    )
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=raw_path)
    budget_spec = budget.to_budget_spec(label="a3_prompt_dev_budget")
    payload_manifests: list[dict[str, Any]] = []
    external_fields: set[str] = set()
    external_reference_fields: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="a3_outbound_preflight_") as tmp:
        tmp_root = Path(tmp)
        for index, anchor_id in enumerate(anchor_ids, start=1):
            starting = load_starting_case(
                int(anchor_id),
                raw_path=raw_path,
                defender=defender,
                artefact_dir=artefact_dir,
            )
            pool = provider.get_pool(str(anchor_id), seed=REFERENCE_POOL_SEED)
            if int(pool.K) != 10:
                raise RuntimeError(f"Outbound preflight requires K=10, got {pool.K}.")
            logger = TrajectoryLogger(
                run_dir=tmp_root / f"anchor_{index:02d}",
                run_id=f"a3_preflight_{index:02d}",
            )
            logger.run_dir.mkdir(parents=True, exist_ok=False)
            validator = ConstraintValidator.from_policy(
                policy, enabled_action_keys=None
            )
            env = AttackEnvironment(
                starting_case=starting,
                defender=defender,
                validator=validator,
                feedback_policy=FeedbackPolicy(mode="label_only"),
                logger=logger,
                budget=budget_spec,
                read_only_context_fields=tuple(pool.read_only_context_fields),
            )
            public_id = temporary_episode_id(
                f"a3:{experiment_seed}:{index:02d}"
            )
            payload = build_a3_prompt_payload(
                env=env,
                reference_pool=pool,
                budget=budget,
                memory_steps=(),
                locked_static_values={},
                query_index=1,
                prompt_version=prompt_version,
                max_local_generation_attempts=(
                    max_local_generation_attempts_per_query
                ),
                outbound_episode_id=public_id,
            )
            manifest = audit_outbound_payload(
                payload,
                allowed_top_level_keys=(
                    "task",
                    "prompt_version",
                    "query_index",
                    "budget",
                    "original_anchor",
                    "current_application",
                    "reference_pool",
                    "action_catalogue",
                    "field_roles",
                    "locked_episode_static_choices",
                    "edit_slot_accounting",
                    "episode_memory",
                    "local_generation_attempt",
                    "max_local_generation_attempts_per_query",
                    "portfolio_cap",
                    "output_schema",
                    "explicitly_unavailable",
                    "local_proposal_repair",
                    "neutral_affordance_view",
                ),
                allowed_feature_fields=(
                    set(payload["original_anchor"]["visible_fields"])
                    | set(validator.enabled_action_keys)
                ),
            )
            external_fields.update(manifest["external_feature_fields"])
            for profile in payload["reference_pool"]["profiles"]:
                external_reference_fields.update(profile["fields"])
            payload_manifests.append(
                {
                    "temporary_anchor_id": public_id,
                    "payload_sha256": manifest["payload_sha256"],
                    "preflight": manifest["preflight"],
                }
            )

    model_configs = []
    for temperature in temperatures:
        ranked = prompt_version == PROMPT_VERSION_P1_RANKED_PORTFOLIO
        config = A3ModelConfig(
            **{
                **FORMAL_A3_MODEL_CONFIG.to_dict(),
                "prompt_version": prompt_version,
                "temperature": float(temperature),
                "max_parse_retries": (
                    0 if ranked else FORMAL_A3_MODEL_CONFIG.max_parse_retries
                ),
                "max_local_generation_attempts_per_query": (
                    max_local_generation_attempts_per_query
                ),
                "portfolio_cap": RANKED_PORTFOLIO_CAP,
            }
        )
        model_configs.append(
            {**config.to_dict(), "config_hash": config.config_hash()}
        )
    return {
        "status": "PASS",
        "evidence_scope": "month6_development_only_not_dissertation_findings",
        "outbound_policy_version": OUTBOUND_POLICY_VERSION,
        "n_fixed_anchors": len(anchor_ids),
        "all_payloads_preflighted_before_api": True,
        "external_feature_fields": sorted(external_fields),
        "external_reference_fields": sorted(external_reference_fields),
        "external_feedback_labels": ["PASS", "BLOCK", "INVALID"],
        "temporary_identifiers_only": True,
        "contains_month7": False,
        "contains_local_paths": False,
        "contains_credentials": False,
        "contains_researcher_only_diagnostics": False,
        "payloads": payload_manifests,
        "model_configs": model_configs,
    }


def run_variant(
    *,
    variant: str,
    anchor_ids: Sequence[str],
    budget: AttackBudget,
    experiment_seed: int,
    run_dir: Path,
    raw_path: Path,
    artefact_dir: Path,
    temperature: float,
    max_local_generation_attempts_per_query: int = (
        ABLATION_MAX_LOCAL_GENERATION_ATTEMPTS
    ),
    prompt_version: str | None = None,
    llm_client: LLMCompletionClient | None = None,
) -> dict[str, Any]:
    if prompt_version is None:
        prompt_version = VARIANT_PROMPT_VERSIONS[variant]
    ranked = prompt_version == PROMPT_VERSION_P1_RANKED_PORTFOLIO
    formal = A3ModelConfig(
        **{
            **FORMAL_A3_MODEL_CONFIG.to_dict(),
            "prompt_version": prompt_version,
            "temperature": float(temperature),
            "max_parse_retries": (
                0 if ranked else FORMAL_A3_MODEL_CONFIG.max_parse_retries
            ),
            "max_local_generation_attempts_per_query": (
                max_local_generation_attempts_per_query
            ),
            "portfolio_cap": RANKED_PORTFOLIO_CAP,
        }
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
        label="a3_prompt_dev_reference_pool",
        source_path=pool_cfg.source_path,
    )
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=raw_path)
    budget_spec = budget.to_budget_spec(label="a3_prompt_dev_budget")

    per_anchor: list[dict[str, Any]] = []
    stop_counts: Counter[str] = Counter()
    parse_fail = 0
    gov_fail = 0
    total_llm_calls = 0
    total_retries = 0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_cached_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0
    total_submissions = 0
    total_local_generation_attempts = 0
    total_local_rejections = 0
    total_local_regenerations = 0
    total_regeneration_exhaustions = 0
    total_env_steps = 0
    total_query_records = 0
    total_invalid_submissions = 0
    projected_duplicate_submissions = 0
    block_followups = 0
    novel_projected_after_block = 0
    duplicate_local_rejections = 0
    leakage_hits: list[str] = []
    config_hashes: set[str] = set()
    successes_at: list[int | None] = []

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
            run_id=f"a3_{variant.lower()}_{anchor_id}",
        )
        logger.run_dir.mkdir(parents=True, exist_ok=False)
        logger.write_manifest(
            {
                "status": (
                    "a3_prompt_temperature_development_only_not_dissertation_findings"
                ),
                "attacker_id": "a3",
                "anchor_id": str(anchor_id),
                "data_split": "dev_month6",
                "experiment_seed": int(experiment_seed),
                "reference_pool_seed": REFERENCE_POOL_SEED,
                "reference_pool_fingerprint": pool.pool_fingerprint,
                "reference_pool_K": pool.K,
                "budget": budget_spec.to_dict(),
                "feedback_mode": "label_only",
                "governance_version": policy.policy_version,
                "governance_fingerprint": policy.policy_fingerprint,
                "defender_name": defender.name,
                "defender_artefact_id": defender.artefact_id,
                "model_config": formal.to_dict(),
                "model_config_hash": formal.config_hash(),
            }
        )
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
            portfolio_cap=formal.portfolio_cap,
            llm_client=llm_client,
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
        total_query_records += len(attacker.query_records)
        total_invalid_submissions += int(match.invalid_submissions)
        parse_fail += int(counters["parse_failures"])
        gov_fail += int(counters["governance_failures"])

        trajectory: list[dict[str, Any]] = []
        for record in attacker.query_records:
            total_cost += float(record.estimated_cost_usd)
            total_prompt_tokens += int(record.prompt_tokens)
            total_cached_tokens += int(record.cached_tokens)
            total_completion_tokens += int(record.completion_tokens)
            total_latency_ms += float(record.latency_ms)
            config_hashes.add(record.config_hash)
            if record.submitted:
                total_submissions += 1
            for local_rec in record.local_generation_records:
                if local_rec.local_rejection_reason == "duplicate_candidate":
                    duplicate_local_rejections += 1
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
                    "selected_portfolio_rank": record.selected_portfolio_rank,
                    "local_generation_records": [
                        item.to_dict() for item in record.local_generation_records
                    ],
                }
            )

        assert int(counters["env_step_calls"]) == int(match.q_used)

        seen_projected: set[str] = set()
        previous_label: str | None = None
        for step in match.trajectory:
            candidate = step.validity.candidate_features
            if candidate is None:
                previous_label = step.public_feedback.label
                continue
            projected_hash = canonical_candidate_fingerprint(
                anchor_id=str(anchor_id),
                projected_candidate=candidate,
                action_fields=pool.action_fields,
            )
            if projected_hash in seen_projected:
                projected_duplicate_submissions += 1
            if previous_label == "BLOCK":
                block_followups += 1
                if projected_hash not in seen_projected:
                    novel_projected_after_block += 1
            seen_projected.add(projected_hash)
            previous_label = step.public_feedback.label

        (logger.run_dir / "a3_episode_summary.json").write_text(
            json.dumps(
                {
                    "anchor_id": anchor_id,
                    "variant": variant,
                    "prompt_version": prompt_version,
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
        successes_at.append(match.attempts_to_success)
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
    intended_q_slots = n * int(budget.q_max)
    rejection_breakdown = _local_rejection_breakdown(per_anchor)
    adaptive = _adaptive_stats(per_anchor)
    asr_curve = _asr_curve(successes_at, int(budget.q_max), n)
    summary = {
        "status": "a3_prompt_development_ablation_only_not_dissertation_findings",
        "variant": variant,
        "condition_id": f"{variant}_t{_temperature_slug(temperature)}",
        "variant_label": PROMPT_VARIANT_LABELS[prompt_version],
        "prompt_version": prompt_version,
        "config_hash": formal.config_hash(),
        "config_hashes_observed": sorted(config_hashes),
        "n_anchors": n,
        "q_max": int(budget.q_max),
        "m_max": int(budget.m_max),
        "n_success": sum(1 for row in per_anchor if row["success"]),
        "asr_curve": asr_curve,
        "asr_curve_auc_mean": (
            sum(float(asr_curve[f"ASR@{q}"]) for q in range(1, int(budget.q_max) + 1))
            / int(budget.q_max)
        ),
        "mean_queries_to_success": (
            sum(int(x) for x in successes_at if x is not None)
            / max(1, sum(1 for x in successes_at if x is not None))
        ),
        "stop_reason_counts": dict(stop_counts),
        "primary_criteria": {
            "predefined_thresholds": PREDEFINED_ENGINEERING_CRITERIA,
            "A_projected_duplicate_submission_count": (
                projected_duplicate_submissions
            ),
            "A_duplicate_candidate_local_rejection_rate": (
                duplicate_local_rejections / total_local_rejections
                if total_local_rejections
                else 0.0
            ),
            "A_duplicate_candidate_count": duplicate_local_rejections,
            "A_total_local_rejections": total_local_rejections,
            "B_regeneration_exhaustion_rate": (
                total_regeneration_exhaustions / n if n else 0.0
            ),
            "B_regeneration_exhaustions": total_regeneration_exhaustions,
            "C_valid_submission_rate_over_attempted_query_records": (
                (total_env_steps - total_invalid_submissions) / total_query_records
                if total_query_records
                else 0.0
            ),
            "C_attempted_query_records": total_query_records,
            "C_invalid_submissions": total_invalid_submissions,
            "C_total_env_steps": total_env_steps,
            "C_intended_q_slots": intended_q_slots,
            "D_env_step_equals_q_used_all_anchors": all(
                int(row["env_steps"]) == int(row["queries_used"]) for row in per_anchor
            ),
            "D_forbidden_leakage_hits": len(leakage_hits),
            "D_single_config_hash": len(config_hashes) == 1,
            "E_block_conditioned_adaptation": adaptive,
            "E_post_block_projected_novelty_rate": (
                novel_projected_after_block / block_followups
                if block_followups
                else None
            ),
            "E_post_block_followups": block_followups,
            "F_parse_failure_events": parse_fail,
            "F_governance_failure_events": gov_fail,
            "G_prompt_tokens_total": total_prompt_tokens,
            "G_prompt_tokens_mean_per_query_record": (
                total_prompt_tokens
                / max(
                    1,
                    sum(len(row.get("trajectory") or []) for row in per_anchor),
                )
            ),
            "G_prompt_tokens_mean_per_valid_submission": (
                total_prompt_tokens / max(1, total_env_steps - total_invalid_submissions)
            ),
            "G_completion_tokens_total": total_completion_tokens,
            "G_prompt_cache_hit_tokens_total": total_cached_tokens,
            "G_prompt_cache_miss_tokens_total_derived": max(
                0, total_prompt_tokens - total_cached_tokens
            ),
            "G_prompt_cache_hit_ratio": (
                total_cached_tokens / total_prompt_tokens
                if total_prompt_tokens
                else 0.0
            ),
            "G_cache_telemetry_note": (
                "cache hits use prompt_tokens_details.cached_tokens when exposed; "
                "cache misses are derived as prompt_tokens - cache hits because "
                "the frozen shared client does not expose a separate miss field"
            ),
        },
        "local_rejection_reason_counts": rejection_breakdown,
        "total_llm_calls": total_llm_calls,
        "mean_llm_calls_per_episode": total_llm_calls / n if n else 0.0,
        "total_retries": total_retries,
        "total_submissions": total_submissions,
        "total_env_step_calls": total_env_steps,
        "total_query_records": total_query_records,
        "total_invalid_submissions": total_invalid_submissions,
        "projected_duplicate_submissions": projected_duplicate_submissions,
        "local_generation_attempts": total_local_generation_attempts,
        "local_rejections": total_local_rejections,
        "local_regenerations": total_local_regenerations,
        "regeneration_exhaustions": total_regeneration_exhaustions,
        "forbidden_leakage_hits": leakage_hits,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "prompt_cache_hit_tokens": total_cached_tokens,
            "prompt_cache_miss_tokens_derived": max(
                0, total_prompt_tokens - total_cached_tokens
            ),
            "prompt_cache_hit_ratio": (
                total_cached_tokens / total_prompt_tokens
                if total_prompt_tokens
                else 0.0
            ),
            "completion_tokens": total_completion_tokens,
        },
        "latency_ms_mean_per_anchor": total_latency_ms / n if n else 0.0,
        "total_estimated_cost_usd": total_cost,
        "estimated_cost_usd_per_successful_episode": (
            total_cost / sum(1 for x in successes_at if x is not None)
            if any(x is not None for x in successes_at)
            else None
        ),
        "per_anchor": per_anchor,
        "model_config": formal.to_dict(),
    }
    return summary


def _write_comparison(
    *,
    parent_dir: Path,
    condition_summaries: Mapping[str, Mapping[str, Any]],
    anchors_meta: Mapping[str, Any],
) -> None:
    rows = []
    for condition_id, s in condition_summaries.items():
        primary = s["primary_criteria"]
        novelty_rate = primary["E_post_block_projected_novelty_rate"]
        criteria_met = (
            primary["A_projected_duplicate_submission_count"] == 0
            and primary["B_regeneration_exhaustion_rate"]
            <= PREDEFINED_ENGINEERING_CRITERIA[
                "local_generation_exhaustion_rate_max"
            ]
            and primary["C_valid_submission_rate_over_attempted_query_records"]
            >= PREDEFINED_ENGINEERING_CRITERIA["valid_submission_rate_min"]
            and primary["D_forbidden_leakage_hits"] == 0
            and primary["D_env_step_equals_q_used_all_anchors"]
            and primary["F_parse_failure_events"] == 0
            and primary["C_invalid_submissions"] == 0
            and (
                novelty_rate is None
                or novelty_rate
                >= PREDEFINED_ENGINEERING_CRITERIA[
                    "post_block_projected_novelty_rate_min"
                ]
            )
        )
        rows.append(
            {
                "condition_id": condition_id,
                "variant": s["variant"],
                "temperature": s["model_config"]["temperature"],
                "prompt_version": s["prompt_version"],
                "config_hash": s["config_hash"],
                "meets_all_primary_engineering_criteria": criteria_met,
                "A_projected_duplicate_submissions": primary[
                    "A_projected_duplicate_submission_count"
                ],
                "A_dup_rate": primary["A_duplicate_candidate_local_rejection_rate"],
                "A_dup_count": primary["A_duplicate_candidate_count"],
                "B_exhaust_rate": primary["B_regeneration_exhaustion_rate"],
                "B_exhaust_count": primary["B_regeneration_exhaustions"],
                "C_valid_submission_rate": primary[
                    "C_valid_submission_rate_over_attempted_query_records"
                ],
                "C_invalid_submissions": primary["C_invalid_submissions"],
                "D_leakage_hits": primary["D_forbidden_leakage_hits"],
                "D_env_q_ok": primary["D_env_step_equals_q_used_all_anchors"],
                "E_post_block_projected_novelty_rate": novelty_rate,
                "E_strategy_switch_after_block": (
                    primary["E_block_conditioned_adaptation"][
                        "strategy_switch_rate_after_block"
                    ]
                ),
                "F_parse_failures": primary["F_parse_failure_events"],
                "G_prompt_tokens_total": primary["G_prompt_tokens_total"],
                "G_prompt_tokens_mean_per_valid_submission": primary[
                    "G_prompt_tokens_mean_per_valid_submission"
                ],
                "G_prompt_cache_hit_ratio": primary[
                    "G_prompt_cache_hit_ratio"
                ],
                "total_llm_calls": s["total_llm_calls"],
                "mean_llm_calls_per_episode": s["mean_llm_calls_per_episode"],
                "total_estimated_cost_usd": s["total_estimated_cost_usd"],
                "estimated_cost_usd_per_successful_episode": s[
                    "estimated_cost_usd_per_successful_episode"
                ],
                "local_rejection_reason_counts": s[
                    "local_rejection_reason_counts"
                ],
                "asr_curve": s["asr_curve"],
                "asr_curve_auc_mean": s["asr_curve_auc_mean"],
                "mean_queries_to_success": s["mean_queries_to_success"],
                "n_success": s["n_success"],
                "stop_reason_counts": s["stop_reason_counts"],
            }
        )
    eligible = [row for row in rows if row["meets_all_primary_engineering_criteria"]]
    eligible.sort(
        key=lambda row: (
            tuple(row["asr_curve"][f"ASR@{q}"] for q in (5, 4, 3, 2, 1)),
            -float(row["mean_queries_to_success"]),
            float(row["C_valid_submission_rate"]),
            -float(row["G_prompt_tokens_mean_per_valid_submission"]),
        ),
        reverse=True,
    )
    selected = eligible[0]["condition_id"] if eligible else None
    comparison = {
        "status": "a3_prompt_development_ablation_only_not_dissertation_findings",
        "selection": anchors_meta,
        "primary_criteria_note": (
            "Select only among conditions meeting every predefined engineering "
            "integrity threshold; then maximise ASR@5 with the pre-specified "
            "secondary tie-breaks."
        ),
        "predefined_engineering_criteria": PREDEFINED_ENGINEERING_CRITERIA,
        "conditions": rows,
        "selection_recommendation_rule": PREDEFINED_ENGINEERING_CRITERIA[
            "selection_rule"
        ],
        "selected_condition": selected,
    }
    (parent_dir / "ablation_comparison.json").write_text(
        json.dumps(to_jsonable(comparison), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "A3 prompt / temperature development ablation",
        "STATUS: DEVELOPMENT / PROMPT SELECTION ONLY — NOT dissertation findings",
        f"anchors: n={anchors_meta['n_anchors']} disjoint_from_frozen_100="
        f"{anchors_meta['disjoint_from_frozen_comparison']}",
        "",
        "Integrity gates first; performance selection second:",
    ]
    for row in rows:
        lines.extend(
            [
                f"--- {row['condition_id']} ({row['prompt_version']}, "
                f"temperature={row['temperature']}) ---",
                f"  criteria_met={row['meets_all_primary_engineering_criteria']}",
                f"  A projected_duplicates="
                f"{row['A_projected_duplicate_submissions']}; "
                f"local_dup_rate={row['A_dup_rate']:.3f} "
                f"(count={row['A_dup_count']})",
                f"  B exhaust_rate={row['B_exhaust_rate']:.3f} "
                f"(count={row['B_exhaust_count']})",
                f"  C valid_submission_rate="
                f"{row['C_valid_submission_rate']:.3f}; "
                f"invalid={row['C_invalid_submissions']}",
                f"  D leakage={row['D_leakage_hits']} env_q_ok={row['D_env_q_ok']}",
                f"  E projected_novelty_after_BLOCK="
                f"{row['E_post_block_projected_novelty_rate']}",
                f"  E strategy_switch_after_block="
                f"{row['E_strategy_switch_after_block']}",
                f"  F parse_failures={row['F_parse_failures']}",
                f"  G prompt_tokens_total={row['G_prompt_tokens_total']}; "
                f"mean/valid_submission="
                f"{row['G_prompt_tokens_mean_per_valid_submission']:.1f}",
                f"  local_rejections={row['local_rejection_reason_counts']}",
                f"  secondary ASR@1..5={row['asr_curve']} successes={row['n_success']}",
                f"  mean_queries_to_success={row['mean_queries_to_success']}",
                f"  stop_reasons={row['stop_reason_counts']}",
            ]
        )
    lines.append(f"Pre-specified selection winner: {selected}")
    report = "\n".join(lines) + "\n"
    (parent_dir / "ablation_comparison.txt").write_text(report, encoding="utf-8")
    print(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-anchors-only",
        action="store_true",
        help="Write the fixed disjoint prompt-dev anchor set and exit.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Audit all 25 outbound payloads, write manifests, and make no API calls.",
    )
    parser.add_argument(
        "--variants",
        default="P1",
        help="Comma-separated subset of P0,P1,P2 (final pass default: P1 only).",
    )
    parser.add_argument(
        "--temperatures",
        default="0.0,0.2,0.5",
        help=(
            "Comma-separated temperature grid. Prompt variants run at the first "
            "value; the temperature prompt variant also runs at each remaining value."
        ),
    )
    parser.add_argument(
        "--temperature-prompt-variant",
        default="P1",
        choices=("P0", "P1", "P2"),
        help="Prompt held fixed for the temperature dimension (default: P1).",
    )
    parser.add_argument("--n-anchors", type=int, default=DEFAULT_N_ANCHORS)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--experiment-seed", type=int, default=DEFAULT_EXPERIMENT_SEED)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument(
        "--frozen-comparison-anchors",
        type=Path,
        default=FROZEN_COMPARISON_ANCHORS,
    )
    parser.add_argument(
        "--anchors-file",
        type=Path,
        default=None,
        help="Existing prompt-dev anchors JSON; created if missing.",
    )
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=None,
        help=(
            "Optional NEW, non-existing ablation parent directory. Existing paths "
            "are refused to prevent artefact overwrite."
        ),
    )
    args = parser.parse_args(argv)

    if int(args.m) != 2 or int(args.q) != 5:
        raise SystemExit("Prompt ablation is locked to Q=5 and m=2.")

    variants = [v.strip().upper() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_PROMPT_VERSIONS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    temperatures = _parse_temperatures(str(args.temperatures))
    base_temperature = temperatures[0]
    temperature_variant = str(args.temperature_prompt_variant).upper()
    if (
        variants != ["P1"]
        or temperatures != [0.0, 0.2, 0.5]
        or temperature_variant != "P1"
        or int(args.n_anchors) != 25
    ):
        raise SystemExit(
            "Final A3 selection is frozen to P1 @ 0.0/0.2/0.5 on exactly 25 anchors."
        )
    conditions: list[tuple[str, float]] = [
        (variant, base_temperature) for variant in variants
    ]
    for temperature in temperatures[1:]:
        condition = (temperature_variant, temperature)
        if condition not in conditions:
            conditions.append(condition)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_anchors = (
        EXPERIMENTS_ROOT
        / "a3"
        / "prompt_development"
        / f"a3_prompt_dev_anchors_n{args.n_anchors}_seed{args.experiment_seed}.json"
    )
    anchors_path = args.anchors_file or default_anchors

    if args.prepare_anchors_only or not anchors_path.exists():
        anchors_meta = prepare_anchor_set(
            n_anchors=int(args.n_anchors),
            experiment_seed=int(args.experiment_seed),
            frozen_anchors_path=args.frozen_comparison_anchors,
            out_path=anchors_path,
        )
        print(f"Wrote prompt-dev anchors: {anchors_path}")
        print(
            f"n={anchors_meta['n_anchors']} remainder_pool="
            f"{anchors_meta['n_remainder_pool']} overlap=0"
        )
        if args.prepare_anchors_only:
            return 0
    else:
        anchors_meta = json.loads(anchors_path.read_text(encoding="utf-8"))

    if not args.raw.exists():
        raise FileNotFoundError(
            f"BAF raw data not found at {args.raw}. Mount the external drive."
        )

    if args.parent_dir is not None:
        parent_dir = args.parent_dir
        if parent_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse existing ablation directory: {parent_dir}"
            )
        parent_dir.mkdir(parents=True, exist_ok=False)
    else:
        parent_dir = new_run_directory(
            f"a3_prompt_temperature_ablation_n{args.n_anchors}_m{args.m}_q{args.q}_"
            f"seed{args.experiment_seed}_{stamp}",
            parent=EXPERIMENTS_ROOT / "a3" / "prompt_development",
            stage="experiments",
        )

    (parent_dir / "prompt_dev_anchors.json").write_text(
        json.dumps(anchors_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (parent_dir / "ablation_protocol.json").write_text(
        json.dumps(
            {
                "status": "a3_prompt_development_ablation_only_not_dissertation_findings",
                "variants": {
                    v: {
                        "prompt_version": VARIANT_PROMPT_VERSIONS[v],
                        "label": PROMPT_VARIANT_LABELS[VARIANT_PROMPT_VERSIONS[v]],
                    }
                    for v in ("P0", "P1", "P2")
                },
                "conditions": [
                    {
                        "condition_id": f"{variant}_t{_temperature_slug(temperature)}",
                        "prompt_variant": variant,
                        "prompt_version": VARIANT_PROMPT_VERSIONS[variant],
                        "temperature": temperature,
                    }
                    for variant, temperature in conditions
                ],
                "frozen": {
                    "Q": 5,
                    "m": 2,
                    "K": 10,
                    "max_local_generation_attempts_per_query": (
                        ABLATION_MAX_LOCAL_GENERATION_ATTEMPTS
                    ),
                    "temperature_is_the_only_decoding_variable": True,
                    "top_p": FORMAL_A3_MODEL_CONFIG.top_p,
                    "model": FORMAL_A3_MODEL_CONFIG.model,
                    "month": 6,
                },
                "predefined_engineering_criteria": PREDEFINED_ENGINEERING_CRITERIA,
                    "performance_selection": [
                        "ASR@5",
                        "ASR@4..1",
                        "query_efficiency",
                        "valid_submission_rate",
                        "token_cost",
                    ],
                "anchors_file": str(anchors_path),
                "selection_rule": (
                    PREDEFINED_ENGINEERING_CRITERIA["selection_rule"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    budget = AttackBudget(q_max=int(args.q), m_max=int(args.m))
    anchor_ids = [str(x) for x in anchors_meta["anchor_ids"]]
    outbound_preflight = preflight_outbound_payloads(
        anchor_ids=anchor_ids,
        budget=budget,
        experiment_seed=int(args.experiment_seed),
        raw_path=args.raw,
        artefact_dir=args.artefact_dir,
        temperatures=temperatures,
    )
    (parent_dir / "outbound_preflight_manifest.json").write_text(
        json.dumps(outbound_preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    condition_summaries: dict[str, dict[str, Any]] = {}
    provenance = _implementation_provenance(args.artefact_dir)
    (parent_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": (
                    "a3_prompt_temperature_development_only_not_dissertation_findings"
                ),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command_argv": sys.argv,
                "data_split": "dev_month6",
                "month7_opened": False,
                "anchor_selection": anchors_meta,
                "conditions": [
                    {
                        "prompt_variant": variant,
                        "prompt_version": VARIANT_PROMPT_VERSIONS[variant],
                        "temperature": temperature,
                    }
                    for variant, temperature in conditions
                ],
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.preflight_only:
        print(f"Outbound preflight PASS; no API calls made. parent_dir: {parent_dir}")
        return 0

    for variant, temperature in conditions:
        prompt_version = VARIANT_PROMPT_VERSIONS[variant]
        formal = A3ModelConfig(
            **{
                **FORMAL_A3_MODEL_CONFIG.to_dict(),
                "prompt_version": prompt_version,
                "temperature": float(temperature),
                "max_local_generation_attempts_per_query": (
                    ABLATION_MAX_LOCAL_GENERATION_ATTEMPTS
                ),
            }
        )
        condition_id = f"{variant}_t{_temperature_slug(temperature)}"
        run_dir = parent_dir / f"condition_{condition_id}_{prompt_version}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "model_config.json").write_text(
            json.dumps(
                {**formal.to_dict(), "config_hash": formal.config_hash()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"\n=== Running {condition_id} ({prompt_version}, "
            f"temperature={temperature}) ==="
        )
        summary = run_variant(
            variant=variant,
            anchor_ids=anchor_ids,
            budget=budget,
            experiment_seed=int(args.experiment_seed),
            run_dir=run_dir,
            raw_path=args.raw,
            artefact_dir=args.artefact_dir,
            temperature=temperature,
        )
        summary["run_dir"] = str(run_dir)
        (run_dir / "summary.json").write_text(
            json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        condition_summaries[condition_id] = summary

    _write_comparison(
        parent_dir=parent_dir,
        condition_summaries=condition_summaries,
        anchors_meta=anchors_meta,
    )
    print(f"parent_dir: {parent_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
