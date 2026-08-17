#!/usr/bin/env python3
"""Offline-only gates and nine-cell preflight for A3 B0/B1/B2.

This script reads archived Month-6 development payload/trajectory artefacts,
never loads raw data or D1, never imports an API client, and emits only local
manifests.  It cannot execute a DeepSeek cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from attack_lab.attackers.a3_agent import (
    FORMAL_A3_MODEL_CONFIG,
    PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
    PROMPT_VERSION_B2_GROUNDED_REFLECTION,
    PROMPT_VERSION_P1_COMPACT,
    A3MemoryStep,
    A3ModelConfig,
    build_a3_prompt_payload,
    render_a3_messages,
)
from attack_lab.attacker_interface import AttackerEpisode
from attack_lab.budget import AttackBudget
from attack_lab.cases import StartingCase
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.governance import CompiledGovernancePolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.outbound_payload import audit_outbound_payload
from attack_lab.reference_pool import ReferencePool, ReferenceProfile
from attack_lab.types import AttackProposal, to_jsonable
from attack_lab.validator import (
    OPAQUE_CANDIDATE_ASSESSMENT_VERSION,
    ConstraintValidator,
)

REPO = Path(__file__).resolve().parents[3]
GOVERNANCE_PATH = REPO / "config" / "attacker_compiled_governance.json"
ARCHIVE = Path(
    "/Users/ziyaoch/ucl/dissertation/05_outputs/experiments/a3/"
    "prompt_development/a3_p1_local_repair_validation_n25_m2_q5_t0_"
    "maxlocal3_seed20260804_20260809T215203Z/"
    "condition_P1_t0_maxlocal3_a3_episodic_p1_compact_v2_stable_prefix"
)
CONDITIONS = {
    "B0": PROMPT_VERSION_P1_COMPACT,
    "B1": PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
    "B2": PROMPT_VERSION_B2_GROUNDED_REFLECTION,
}
CELL_ORDER = (
    (1, "B0"),
    (1, "B1"),
    (1, "B2"),
    (2, "B1"),
    (2, "B2"),
    (2, "B0"),
    (3, "B2"),
    (3, "B0"),
    (3, "B1"),
)
Q_MAX = 5
M_MAX = 2
K = 10
MAX_LOCAL = 3
EXPERIMENT_SEED = 20260804
EXPECTED_OUTBOUND_FIELDS = (
    "bank_months_count",
    "current_address_months_count",
    "customer_age",
    "device_os",
    "email_is_free",
    "employment_status",
    "foreign_request",
    "has_other_cards",
    "housing_status",
    "income",
    "intended_balcon_amount",
    "keep_alive_session",
    "payment_type",
    "prev_address_months_count",
    "proposed_credit_limit",
    "session_length_in_minutes",
    "source",
)


class NoCallDefender:
    name = "offline_preflight_no_defender"
    artefact_id = "not_loaded"
    threshold = 0.0

    def score_application(self, _features: Mapping[str, Any]):
        raise AssertionError("Offline preflight must never call D1.")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        to_jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _full_anchor(anchor_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(anchor_dir / "trajectory.jsonl")
    candidate = next(
        dict(row["validity"]["candidate_features"])
        for row in rows
        if row.get("validity", {}).get("candidate_features") is not None
    )
    candidate.update(dict(payload["original_anchor"]["visible_fields"]))
    return candidate


def _reference_pool(payload: Mapping[str, Any], anchor_id: str) -> ReferencePool:
    raw = dict(payload["reference_pool"])
    profiles = tuple(
        ReferenceProfile(
            profile_id=str(item["profile_id"]),
            fields=dict(item["fields"]),
            generation_seed=EXPERIMENT_SEED,
        )
        for item in raw["profiles"]
    )
    return ReferencePool(
        anchor_id=anchor_id,
        K=int(raw["K"]),
        generation_seed=EXPERIMENT_SEED,
        pool_fingerprint=_sha(raw),
        context_fields=tuple(raw["context_fields"]),
        action_fields=tuple(raw["action_fields"]),
        read_only_context_fields=tuple(raw["read_only_context_fields"]),
        profiles=profiles,
        source_row_ids=tuple(range(int(raw["K"]))),
    )


def _fixture(anchor_dir: Path, policy: CompiledGovernancePolicy, scratch: Path):
    anchor_id = anchor_dir.name.removeprefix("anchor_")
    archived_payload = _read_json(
        anchor_dir / "query_01" / "local_gen_01" / "a3_prompt_payload.json"
    )
    full_anchor = _full_anchor(anchor_dir, archived_payload)
    starting_case = StartingCase(
        case_id=anchor_id,
        source_row_id=-1,
        label=1,
        features=full_anchor,
        initial_score=0.0,
        initial_decision="BLOCK",
        data_split="dev_month6",
    )
    pool = _reference_pool(archived_payload, anchor_id)
    validator = ConstraintValidator.from_policy(policy)
    logger = TrajectoryLogger(
        run_dir=scratch / f"anchor_{anchor_id}", run_id="offline_preflight"
    )
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=NoCallDefender(),
        validator=validator,
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=AttackBudget(q_max=Q_MAX, m_max=M_MAX).to_budget_spec(
            label="a3_stage_b_offline_preflight"
        ),
        read_only_context_fields=pool.read_only_context_fields,
    )
    return env, AttackerEpisode(env), pool, _read_jsonl(anchor_dir / "trajectory.jsonl")


def _model_config(prompt_version: str) -> A3ModelConfig:
    return A3ModelConfig(
        model=FORMAL_A3_MODEL_CONFIG.model,
        thinking_disabled=FORMAL_A3_MODEL_CONFIG.thinking_disabled,
        temperature=0.0,
        top_p=FORMAL_A3_MODEL_CONFIG.top_p,
        max_tokens=FORMAL_A3_MODEL_CONFIG.max_tokens,
        max_parse_retries=FORMAL_A3_MODEL_CONFIG.max_parse_retries,
        timeout_seconds=FORMAL_A3_MODEL_CONFIG.timeout_seconds,
        prompt_version=prompt_version,
        max_local_generation_attempts_per_query=MAX_LOCAL,
    )


def _action_semantics(policy: CompiledGovernancePolicy) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for action_key in policy.available_action_keys:
        rule = policy.field_for_action(action_key)
        assert rule is not None
        mapping_hash = None
        if rule.agent_action_mode == "proxy_action":
            mapping_hash = _sha(dict(rule.resolved_proxy_actions))
        fields.append(
            {
                "action_key": action_key,
                "underlying_feature": rule.feature,
                "attacker_visibility": rule.attacker_visible,
                "agent_mutability": rule.agent_mutability,
                "governance_status": rule.governance_status,
                "category": (
                    "episode_static" if rule.is_episode_locked else "per_attempt"
                ),
                "data_type": rule.data_type,
                "domain_mode": rule.domain_mode,
                "sampling_kind": rule.sampling_kind,
                "lower_bound": rule.lower_bound,
                "upper_bound": rule.upper_bound,
                "allowed_values": to_jsonable(list(rule.allowed_values)),
                "sentinel_spec": to_jsonable(dict(rule.sentinel_spec)),
                "sentinel_policy": rule.sentinel_policy,
                "hard_constraints_sha256": _sha(rule.hard_constraints),
                "action_mode": rule.agent_action_mode,
                "proxy_abstract_actions": list(rule.resolved_proxy_actions),
                "proxy_mapping_sha256": mapping_hash,
            }
        )
    return {
        "q_max": Q_MAX,
        "m_max": M_MAX,
        "max_local_generation_attempts_per_query": MAX_LOCAL,
        "candidate_assessment_version": OPAQUE_CANDIDATE_ASSESSMENT_VERSION,
        "action_fields": fields,
    }


def _action_keys_from_payload(payload: Mapping[str, Any], condition: str) -> list[str]:
    if condition == "B0":
        return [str(item["action_key"]) for item in payload["action_catalogue"]]
    return [
        str(item["action_key"])
        for item in payload["neutral_affordance_view"]["actions"]
    ]


def _build_payloads(
    *,
    anchor_dir: Path,
    condition: str,
    policy: CompiledGovernancePolicy,
    scratch: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_env, facade, pool, rows = _fixture(anchor_dir, policy, scratch)
    prompt_version = CONDITIONS[condition]
    payloads: list[dict[str, Any]] = []
    q1 = build_a3_prompt_payload(
        env=facade,
        reference_pool=pool,
        budget=AttackBudget(q_max=Q_MAX, m_max=M_MAX),
        memory_steps=(),
        locked_static_values={},
        query_index=1,
        prompt_version=prompt_version,
        local_generation_attempt=1,
        max_local_generation_attempts=MAX_LOCAL,
        outbound_episode_id=f"dev-anchor-{_sha(anchor_dir.name)[:16]}",
    )
    payloads.append(q1)

    first = rows[0]
    proposal = AttackProposal(changes=dict(first["proposed_changes"]))
    prep = raw_env.validator.prepare_episode_locks(
        raw_env.starting_case.features, proposal
    )
    assessment = raw_env.validator.assess_candidate(
        raw_env.starting_case.features,
        proposal,
        locked_values=prep.locked_values,
        pre_feedback_errors=prep.errors,
        anchor_id=raw_env.starting_case.case_id,
        m_max=M_MAX,
    )
    memory = (
        A3MemoryStep(
            query_index=1,
            strategy_label=first["research_meta"].get("strategy_label"),
            changes=dict(first["proposed_changes"]),
            edited_fields=assessment.edited_action_dimensions,
            public_label=str(first["public_feedback"]["label"]),
            adaptation_note=first["research_meta"].get("adaptation_note"),
            governance_reject_reason=None,
            q_remaining_after=4,
            submitted=True,
        ),
    )
    q2 = build_a3_prompt_payload(
        env=facade,
        reference_pool=pool,
        budget=AttackBudget(q_max=Q_MAX, m_max=M_MAX),
        memory_steps=memory,
        locked_static_values=prep.locked_values,
        query_index=2,
        prompt_version=prompt_version,
        local_generation_attempt=2,
        max_local_generation_attempts=MAX_LOCAL,
        outbound_episode_id=f"dev-anchor-{_sha(anchor_dir.name)[:16]}",
        local_proposal_repair=(
            {
                "local_generation_attempt": 1,
                "local_rejection_reason": "same_as_anchor",
                "changes": dict(first["proposed_changes"]),
            },
        ),
    )
    payloads.append(q2)
    return payloads, {
        "action_keys": _action_keys_from_payload(q1, condition),
        "q1_top_level_keys": sorted(q1),
        "q2_top_level_keys": sorted(q2),
        "changes_schema": q1["output_schema"]["properties"]["changes"],
    }


def _gate2_rows(
    *,
    policy: CompiledGovernancePolicy,
    outbound_fields: set[str],
    action_keys_by_condition: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_action_keys = {
        condition: set(keys) for condition, keys in action_keys_by_condition.items()
    }
    for field, rule in policy.fields.items():
        proxy = rule.agent_action_mode == "proxy_action"
        action_key = (
            rule.proxy_action_key
            if proxy
            else field if rule.is_mutable else None
        )
        raw_present = field in outbound_fields
        action_present_all = (
            action_key is not None
            and all(action_key in keys for keys in all_action_keys.values())
        )
        if rule.attacker_visible == "yes":
            visibility_basis = (
                "attacker_visible_read_only_context"
                if not rule.is_mutable
                else "attacker_visible_mutable_raw_action"
            )
            passed = raw_present and (
                action_present_all if rule.is_mutable else not action_present_all
            )
        elif rule.attacker_visible == "proxy_only":
            visibility_basis = "proxy_only_abstract_action"
            passed = not raw_present and proxy and action_present_all
        else:
            visibility_basis = "hidden_or_forbidden"
            passed = not raw_present and not action_present_all
        rows.append(
            {
                "field": field,
                "outbound_present": raw_present,
                "visibility_basis": visibility_basis,
                "mutable": bool(rule.is_mutable),
                "action_key": action_key,
                "action_key_present_all_conditions": action_present_all,
                "proxy": proxy,
                "raw_value_exposed": raw_present,
                "governance_status": rule.governance_status,
                "agent_mutability": rule.agent_mutability,
                "attacker_visibility": rule.attacker_visible,
                "reference_pool_role": (
                    "read_only_context"
                    if field in {"bank_months_count", "has_other_cards"}
                    else "mutable_action_context"
                    if field in EXPECTED_OUTBOUND_FIELDS
                    else "not_raw_outbound"
                ),
                "governance_source": (
                    "config/attacker_compiled_governance.json#"
                    + policy.policy_fingerprint
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / "_local_scratch_no_api"
    scratch.mkdir()

    policy = CompiledGovernancePolicy.load(GOVERNANCE_PATH)
    anchors = sorted(ARCHIVE.glob("anchor_*"))
    if len(anchors) != 25:
        raise RuntimeError(f"Expected 25 fixed anchors; found {len(anchors)}")

    canonical = _action_semantics(policy)
    canonical_hash = _sha(canonical)
    gate1 = {
        "gate": "ACTION-SPACE EQUIVALENCE",
        "governance_policy_fingerprint": policy.policy_fingerprint,
        "conditions": {
            condition: {"semantic_manifest": canonical, "sha256": canonical_hash}
            for condition in CONDITIONS
        },
        "representation_checks": {},
        "status": "PENDING",
    }
    cells: list[dict[str, Any]] = []
    observed_outbound_fields: set[str] = set()
    action_keys_by_condition: dict[str, list[str]] = {}
    schema_by_condition: dict[str, Any] = {}
    preflight_failures: list[dict[str, Any]] = []
    max_payload_utf8_bytes_observed = 0

    for order_index, (replicate, condition) in enumerate(CELL_ORDER, start=1):
        prompt_version = CONDITIONS[condition]
        config = _model_config(prompt_version)
        cell_id = f"r{replicate}_{order_index:02d}_{condition.lower()}"
        cell_dir = output / "planned_cells" / cell_id
        anchor_records: list[dict[str, Any]] = []
        cell_calls = 0
        for anchor_dir in anchors:
            payloads, representation = _build_payloads(
                anchor_dir=anchor_dir,
                condition=condition,
                policy=policy,
                scratch=scratch / cell_id,
            )
            if condition not in action_keys_by_condition:
                action_keys_by_condition[condition] = representation["action_keys"]
                schema_by_condition[condition] = representation
            payload_records = []
            for payload in payloads:
                max_payload_utf8_bytes_observed = max(
                    max_payload_utf8_bytes_observed,
                    len(
                        json.dumps(
                            to_jsonable(payload),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ),
                )
                audit = audit_outbound_payload(
                    payload,
                    allowed_top_level_keys=tuple(payload.keys()),
                    allowed_feature_fields=(
                        set(EXPECTED_OUTBOUND_FIELDS)
                        | set(policy.available_action_keys)
                    ),
                )
                observed_outbound_fields.update(audit["external_feature_fields"])
                if audit["preflight"] != "PASS":
                    preflight_failures.append(
                        {
                            "cell_id": cell_id,
                            "anchor": anchor_dir.name,
                            "query_shape": len(payload_records) + 1,
                        }
                    )
                payload_records.append(
                    {
                        "query_shape": len(payload_records) + 1,
                        "payload_sha256": audit["payload_sha256"],
                        "rendered_messages_sha256": _sha(
                            render_a3_messages(payload)
                        ),
                        "top_level_keys": audit["top_level_keys"],
                        "external_feature_fields": audit["external_feature_fields"],
                        "public_feedback_labels_present": audit[
                            "public_feedback_labels_present"
                        ],
                        "preflight": audit["preflight"],
                    }
                )
            cell_calls += len(payloads)
            anchor_records.append(
                {
                    "temporary_anchor_id": payloads[0]["original_anchor"]["case_id"],
                    "K": payloads[0]["reference_pool"]["K"],
                    "profile_ids": [
                        item["profile_id"]
                        for item in payloads[0]["reference_pool"]["profiles"]
                    ],
                    "payload_shapes": payload_records,
                }
            )
        cell_manifest = {
            "cell_id": cell_id,
            "execution_order": order_index,
            "replicate": replicate,
            "condition": condition,
            "prompt_version": prompt_version,
            "model_config": config.to_dict(),
            "config_hash": config.config_hash(),
            "planned_unique_output_directory": str(cell_dir),
            "n_fixed_anchors": len(anchors),
            "offline_payload_shapes_built": cell_calls,
            "anchors": anchor_records,
        }
        (output / f"{cell_id}_preflight.json").write_text(
            json.dumps(cell_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cells.append(
            {key: value for key, value in cell_manifest.items() if key != "anchors"}
        )

    expected_action_keys = list(policy.available_action_keys)
    for condition in CONDITIONS:
        actual = action_keys_by_condition.get(condition, [])
        gate1["representation_checks"][condition] = {
            "runtime_action_keys": expected_action_keys,
            "prompt_representation_action_keys": actual,
            "exact_order_and_set_match": actual == expected_action_keys,
            "q_max": Q_MAX,
            "m_max": M_MAX,
            "static_lock_semantics": "ConstraintValidator.prepare_episode_locks",
            "candidate_assessment": OPAQUE_CANDIDATE_ASSESSMENT_VERSION,
        }
    gate1["status"] = (
        "PASS"
        if len({item["sha256"] for item in gate1["conditions"].values()}) == 1
        and all(
            item["exact_order_and_set_match"]
            for item in gate1["representation_checks"].values()
        )
        else "FAIL"
    )
    (output / "gate1_action_space_equivalence.json").write_text(
        json.dumps(gate1, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    gate2_rows = _gate2_rows(
        policy=policy,
        outbound_fields=observed_outbound_fields,
        action_keys_by_condition=action_keys_by_condition,
    )
    gate2 = {
        "gate": "OUTBOUND-FIELD GOVERNANCE CROSSCHECK",
        "governance_policy_fingerprint": policy.policy_fingerprint,
        "expected_exact_17_fields": list(EXPECTED_OUTBOUND_FIELDS),
        "observed_external_feature_fields": sorted(observed_outbound_fields),
        "exact_17_match": observed_outbound_fields == set(EXPECTED_OUTBOUND_FIELDS),
        "forbidden_leakage_count": sum(
            1 for item in gate2_rows if item["status"] == "FAIL"
        ),
        "rows": gate2_rows,
    }
    gate2["status"] = (
        "PASS"
        if gate2["exact_17_match"]
        and gate2["forbidden_leakage_count"] == 0
        else "FAIL"
    )
    (output / "gate2_outbound_governance_crosscheck.json").write_text(
        json.dumps(gate2, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "gate2_outbound_governance_crosscheck.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate2_rows[0]))
        writer.writeheader()
        writer.writerows(gate2_rows)

    schema_diff = {
        "B0": schema_by_condition["B0"],
        "B1": schema_by_condition["B1"],
        "B2": schema_by_condition["B2"],
        "B1_B2_changes_schema_identical": (
            schema_by_condition["B1"]["changes_schema"]
            == schema_by_condition["B2"]["changes_schema"]
        ),
        "B0_legacy_representation_preserved": (
            "neutral_affordance_view"
            not in schema_by_condition["B0"]["q1_top_level_keys"]
        ),
    }
    (output / "payload_schema_diff.json").write_text(
        json.dumps(schema_diff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    max_local_generations = len(CELL_ORDER) * len(anchors) * Q_MAX * MAX_LOCAL
    max_transport_calls = max_local_generations * (
        1 + FORMAL_A3_MODEL_CONFIG.max_parse_retries
    )
    estimated_prompt_tokens_upper_proxy = (
        math.ceil((2 * max_payload_utf8_bytes_observed) / 3)
        * max_transport_calls
    )
    estimated_max_completion_tokens = (
        max_transport_calls * FORMAL_A3_MODEL_CONFIG.max_tokens
    )
    protocol = {
        "status": (
            "PASS"
            if gate1["status"] == "PASS"
            and gate2["status"] == "PASS"
            and not preflight_failures
            and schema_diff["B1_B2_changes_schema_identical"]
            else "FAIL"
        ),
        "offline_only": True,
        "external_api_calls": 0,
        "month7_read": False,
        "raw_dataset_read": False,
        "d1_called": False,
        "fixed_anchor_count": len(anchors),
        "cells": cells,
        "cell_order": [f"R{replicate}:{condition}" for replicate, condition in CELL_ORDER],
        "same_conditions": {
            "model_alias": FORMAL_A3_MODEL_CONFIG.model,
            "temperature": 0.0,
            "top_p": FORMAL_A3_MODEL_CONFIG.top_p,
            "max_tokens": FORMAL_A3_MODEL_CONFIG.max_tokens,
            "q_max": Q_MAX,
            "m_max": M_MAX,
            "K": K,
            "max_local_generation_attempts_per_query": MAX_LOCAL,
            "feedback": "label_only",
            "governance_policy_fingerprint": policy.policy_fingerprint,
        },
        "replicate_semantics": (
            "Provider-side generation repeats; no controllable API seed is assumed."
        ),
        "estimated_max_local_planner_generations": max_local_generations,
        "estimated_max_transport_calls_including_parse_retries": max_transport_calls,
        "max_offline_payload_utf8_bytes_observed_q1_q2": (
            max_payload_utf8_bytes_observed
        ),
        "estimated_prompt_tokens_upper_proxy": estimated_prompt_tokens_upper_proxy,
        "estimated_max_completion_tokens": estimated_max_completion_tokens,
        "token_estimate_note": (
            "Prompt estimate is an engineering proxy, not provider tokenizer output: "
            "twice the largest observed q1/q2 payload bytes divided by 3, multiplied "
            "by the worst-case transport-call count. Actual usage must be logged."
        ),
        "payload_shapes_audited": len(CELL_ORDER) * len(anchors) * 2,
        "preflight_failures": preflight_failures,
        "selection_rule": [
            "integrity/leakage/legal/Q/m/memory gates",
            "mean ASR@5 over 3 repeats and ASR curve",
            "queries-to-success and post-BLOCK strategy change",
            "exhaustion",
            "token cost",
        ],
        "no_best_run_selection": True,
        "no_additional_condition_after_grid": True,
    }
    (output / "nine_cell_preflight_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "gate1": gate1["status"],
                "gate2": gate2["status"],
                "protocol": protocol["status"],
                "cells": len(cells),
                "anchors_per_cell": len(anchors),
                "payload_shapes_audited": protocol["payload_shapes_audited"],
                "external_api_calls": 0,
                "month7_read": False,
                "output_dir": str(output),
            },
            indent=2,
        )
    )
    return 0 if protocol["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
