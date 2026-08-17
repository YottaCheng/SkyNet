#!/usr/bin/env python3
"""Thin B0/B1/B2 x3 Stage-B orchestration over the existing A3 cell runner.

Default mode performs existing outbound preflight only and never creates a
DeepSeek client. Live execution requires both ``--launch`` and the exact
confirmation string. This module does not implement D1 loading, episode
execution, transport retry, payload auditing, or single-cell metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
# Prompt-ablation runner relocated in 2026-08-active-stack-cleanup.
ABLATION_SCRIPTS = REPO / "archive" / "2026-08-active-stack-cleanup" / "scripts"
for import_path in (SRC, SCRIPTS, ABLATION_SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from attack_lab.attackers.a1_planner import (  # noqa: E402
    DeepSeekPlannerClient,
    LLMCompletion,
    LLMCompletionClient,
    estimate_flash_cost_usd,
)
from attack_lab.attackers.a3_agent import (  # noqa: E402
    FORMAL_A3_MODEL_CONFIG,
    PROMPT_VARIANT_LABELS,
    PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
    PROMPT_VERSION_B2_GROUNDED_REFLECTION,
    PROMPT_VERSION_P1_COMPACT,
    A3ModelConfig,
)
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR  # noqa: E402
from attack_lab.types import to_jsonable  # noqa: E402
from run_a3_prompt_ablation import (  # noqa: E402
    DEFAULT_EXPERIMENT_SEED,
    DEFAULT_RAW_PATH,
    preflight_outbound_payloads,
    run_variant,
)

RUNNER_VERSION = "a3-stage-b-thin-grid-v1"
LIVE_CONFIRMATION = "I_AUTHORIZE_A3_STAGE_B_B0_B1_B2_X3_DEEPSEEK_LIVE_RUN"
COST_CAP_USD = 25.0
Q_MAX = 5
M_MAX = 2
K = 10
TEMPERATURE = 0.0
MAX_LOCAL_GENERATIONS = 3

DISSERTATION_ROOT = REPO.parent
DEFAULT_ANCHORS_FILE = (
    DISSERTATION_ROOT
    / "05_outputs"
    / "experiments"
    / "a3"
    / "prompt_development"
    / "a3_prompt_temperature_ablation_n25_m2_q5_seed20260804_20260809T193343Z"
    / "prompt_dev_anchors.json"
)
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


@dataclass(frozen=True)
class CellSpec:
    replicate: int
    condition: str
    order: int
    prompt_version: str

    @property
    def cell_id(self) -> str:
        return f"r{self.replicate}_{self.order:02d}_{self.condition.lower()}"


CELL_PROTOCOL = (
    CellSpec(1, "B0", 1, PROMPT_VERSION_P1_COMPACT),
    CellSpec(1, "B1", 2, PROMPT_VERSION_B1_NEUTRAL_GROUNDED),
    CellSpec(1, "B2", 3, PROMPT_VERSION_B2_GROUNDED_REFLECTION),
    CellSpec(2, "B1", 4, PROMPT_VERSION_B1_NEUTRAL_GROUNDED),
    CellSpec(2, "B2", 5, PROMPT_VERSION_B2_GROUNDED_REFLECTION),
    CellSpec(2, "B0", 6, PROMPT_VERSION_P1_COMPACT),
    CellSpec(3, "B2", 7, PROMPT_VERSION_B2_GROUNDED_REFLECTION),
    CellSpec(3, "B0", 8, PROMPT_VERSION_P1_COMPACT),
    CellSpec(3, "B1", 9, PROMPT_VERSION_B1_NEUTRAL_GROUNDED),
)


class GridRunnerError(RuntimeError):
    """Fail-closed orchestration error."""


class GlobalCostCapReached(BaseException):
    """Non-transport control signal that must escape A3 retry handling."""


class _NoCallClient:
    def complete(self, *_args: Any, **_kwargs: Any) -> LLMCompletion:
        raise AssertionError("Mock orchestration must not call an external client.")


class SingleCellRunner(Protocol):
    def __call__(self, **kwargs: Any) -> dict[str, Any]: ...


class PreflightRunner(Protocol):
    def __call__(self, **kwargs: Any) -> dict[str, Any]: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        to_jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artefact: {path}")
    write_json_atomic(path, payload)


def model_config_for(spec: CellSpec) -> A3ModelConfig:
    return A3ModelConfig(
        **{
            **FORMAL_A3_MODEL_CONFIG.to_dict(),
            "prompt_version": spec.prompt_version,
            "temperature": TEMPERATURE,
            "max_local_generation_attempts_per_query": MAX_LOCAL_GENERATIONS,
        }
    )


def load_fixed_anchors(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchor_ids = [str(value) for value in payload["anchor_ids"]]
    if len(anchor_ids) != 25 or len(set(anchor_ids)) != 25:
        raise GridRunnerError("Stage-B requires exactly 25 unique fixed anchors.")
    return anchor_ids


def _read_guard_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GridRunnerError(
                f"Malformed append-only cost guard ledger at line {line_number}."
            ) from exc
        if not isinstance(row, dict) or "estimated_cost_usd" not in row:
            raise GridRunnerError("Malformed cost guard ledger row.")
        entries.append(row)
    return entries


def guard_total_cost(path: Path) -> float:
    return float(sum(float(row["estimated_cost_usd"]) for row in _read_guard_entries(path)))


def guard_attempt_cost(path: Path, *, cell_id: str, attempt_id: str) -> float:
    return float(
        sum(
            float(row["estimated_cost_usd"])
            for row in _read_guard_entries(path)
            if row.get("cell_id") == cell_id and row.get("attempt_id") == attempt_id
        )
    )


@dataclass
class GlobalCostGuardClient:
    """Thin cross-cell fuse; transport/retry and scientific usage remain existing code."""

    delegate: LLMCompletionClient
    ledger_path: Path
    cell_id: str
    attempt_id: str
    cap_usd: float = COST_CAP_USD

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
        thinking_disabled: bool,
    ) -> LLMCompletion:
        before = guard_total_cost(self.ledger_path)
        if before >= float(self.cap_usd):
            raise GlobalCostCapReached(
                f"Global provider-usage cost cap reached: USD {before:.9f}."
            )
        completion = self.delegate.complete(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            thinking_disabled=thinking_disabled,
        )
        call_cost = estimate_flash_cost_usd(
            prompt_tokens=int(completion.prompt_tokens),
            cached_tokens=int(completion.cached_tokens),
            completion_tokens=int(completion.completion_tokens),
        )
        cumulative = before + float(call_cost)
        row = {
            "ledger_version": "stage-b-global-cost-guard-v1",
            "timestamp_utc": utc_now(),
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "model_alias": completion.model,
            "provider_prompt_tokens": int(completion.prompt_tokens),
            "provider_cached_tokens": int(completion.cached_tokens),
            "provider_completion_tokens": int(completion.completion_tokens),
            "estimated_cost_usd": float(call_cost),
            "cumulative_estimated_cost_usd": float(cumulative),
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return completion


def _validate_existing_preflight(
    payload: Mapping[str, Any], *, condition: str
) -> None:
    if payload.get("status") != "PASS":
        raise GridRunnerError(f"Existing outbound preflight failed for {condition}.")
    if payload.get("all_payloads_preflighted_before_api") is not True:
        raise GridRunnerError(f"Incomplete outbound preflight for {condition}.")
    if int(payload.get("n_fixed_anchors", -1)) != 25:
        raise GridRunnerError(f"Preflight anchor count drift for {condition}.")
    if tuple(payload.get("external_feature_fields") or ()) != tuple(
        sorted(EXPECTED_OUTBOUND_FIELDS)
    ):
        raise GridRunnerError(f"Exact outbound 17-field set drift for {condition}.")
    for key in (
        "contains_month7",
        "contains_local_paths",
        "contains_credentials",
        "contains_researcher_only_diagnostics",
    ):
        if payload.get(key) is not False:
            raise GridRunnerError(f"Outbound preflight safety flag failed: {key}.")
    rows = payload.get("payloads") or []
    if len(rows) != 25 or any(row.get("preflight") != "PASS" for row in rows):
        raise GridRunnerError(f"Per-anchor outbound preflight failed for {condition}.")


def _runtime_manifest_evidence(attempt_dir: Path) -> dict[str, Any]:
    paths = sorted(attempt_dir.rglob("outbound_payload_manifest.json"))
    if not paths:
        raise GridRunnerError("Completed cell has no runtime outbound manifests.")
    payload_hashes: list[str] = []
    file_hashes: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("preflight") != "PASS":
            raise GridRunnerError("Existing runtime outbound audit is not PASS.")
        payload_hash = str(payload.get("payload_sha256") or "")
        if len(payload_hash) != 64:
            raise GridRunnerError("Runtime outbound manifest lacks payload hash.")
        payload_hashes.append(payload_hash)
        file_hashes.append(sha256_file(path))
    return {
        "manifest_count": len(paths),
        "manifest_file_hashes": file_hashes,
        "payload_hashes": payload_hashes,
    }


def _validate_existing_cell_summary(
    summary: Mapping[str, Any], *, spec: CellSpec, anchor_ids: Sequence[str]
) -> None:
    expected_config = model_config_for(spec)
    if int(summary.get("n_anchors", -1)) != 25:
        raise GridRunnerError("Existing cell summary anchor count drift.")
    if summary.get("prompt_version") != spec.prompt_version:
        raise GridRunnerError("Existing cell summary prompt version drift.")
    if summary.get("config_hash") != expected_config.config_hash():
        raise GridRunnerError("Existing cell summary config hash drift.")
    if int(summary.get("q_max", -1)) != Q_MAX or int(summary.get("m_max", -1)) != M_MAX:
        raise GridRunnerError("Existing cell summary Q/m drift.")
    observed = [str(row["anchor_id"]) for row in summary.get("per_anchor") or []]
    if observed != list(anchor_ids):
        raise GridRunnerError("Existing cell summary anchor order drift.")
    if "total_estimated_cost_usd" not in summary or "asr_curve" not in summary:
        raise GridRunnerError("Existing cell summary lacks authoritative metrics.")


def aggregate_existing_summaries(
    *, cell_rows: Sequence[Mapping[str, Any]], total_guard_cost_usd: float
) -> dict[str, Any]:
    if len(cell_rows) != 9:
        raise GridRunnerError("Nine completed existing summaries are required.")
    by_condition: dict[str, list[Mapping[str, Any]]] = {"B0": [], "B1": [], "B2": []}
    for row in cell_rows:
        by_condition[str(row["condition"])].append(row)

    condition_results: dict[str, Any] = {}
    repeated_outcomes: dict[str, Any] = {}
    for condition, rows in by_condition.items():
        rows = sorted(rows, key=lambda value: int(value["replicate"]))
        asr5 = [float(row["summary"]["asr_curve"]["ASR@5"]) for row in rows]
        condition_results[condition] = {
            "n_service_side_repeats": len(rows),
            "mean_ASR@5": statistics.mean(asr5),
            "ASR@5_range": [min(asr5), max(asr5)],
            "ASR@5_sample_sd": statistics.stdev(asr5),
            "runs": [
                {
                    "cell_id": row["cell_id"],
                    "replicate": row["replicate"],
                    "asr_curve": row["summary"]["asr_curve"],
                    "auc": row["summary"]["asr_curve_auc_mean"],
                    "mean_queries_to_success": row["summary"][
                        "mean_queries_to_success"
                    ],
                    "total_llm_calls": row["summary"]["total_llm_calls"],
                    "token_usage": row["summary"]["token_usage"],
                    "total_estimated_cost_usd": row["summary"][
                        "total_estimated_cost_usd"
                    ],
                }
                for row in rows
            ],
        }
        anchor_outcomes: dict[str, list[bool]] = {}
        for row in rows:
            for anchor in row["summary"]["per_anchor"]:
                anchor_outcomes.setdefault(str(anchor["anchor_id"]), []).append(
                    bool(anchor["success"])
                )
        repeated_outcomes[condition] = {
            anchor_id: {
                "outcomes": outcomes,
                "successes_across_3_service_repeats": sum(outcomes),
                "success_frequency": sum(outcomes) / 3.0,
            }
            for anchor_id, outcomes in sorted(anchor_outcomes.items())
        }

    by_rep_condition = {
        (int(row["replicate"]), str(row["condition"])): {
            str(anchor["anchor_id"]): bool(anchor["success"])
            for anchor in row["summary"]["per_anchor"]
        }
        for row in cell_rows
    }
    transitions: dict[str, Any] = {}
    for left, right in (("B0", "B1"), ("B0", "B2"), ("B1", "B2")):
        rows = []
        for replicate in (1, 2, 3):
            left_values = by_rep_condition[(replicate, left)]
            right_values = by_rep_condition[(replicate, right)]
            counts = Counter(
                (left_values[anchor_id], right_values[anchor_id])
                for anchor_id in left_values
            )
            rows.append(
                {
                    "replicate": replicate,
                    "failure_to_success": counts[(False, True)],
                    "success_to_failure": counts[(True, False)],
                    "success_to_success": counts[(True, True)],
                    "failure_to_failure": counts[(False, False)],
                }
            )
        transitions[f"{left}_to_{right}"] = rows

    return {
        "status": "stage_b_month6_development_complete_not_final_findings",
        "runner_version": RUNNER_VERSION,
        "same_25_anchors_not_n75": True,
        "pseudoreplication_warning": (
            "Three service-side repeats reuse the same 25 anchors; they are not "
            "75 independent observations."
        ),
        "no_best_run_selection": True,
        "condition_results": condition_results,
        "anchor_level_repeated_outcomes": repeated_outcomes,
        "paired_anchor_condition_transitions_by_replicate": transitions,
        "global_cost_guard_total_usd": float(total_guard_cost_usd),
        "completed_utc": utc_now(),
    }


def _mock_preflight_outbound_payloads(**kwargs: Any) -> dict[str, Any]:
    anchor_ids = [str(value) for value in kwargs["anchor_ids"]]
    return {
        "status": "PASS",
        "evidence_scope": "mock_orchestration_only_not_scientific",
        "n_fixed_anchors": len(anchor_ids),
        "all_payloads_preflighted_before_api": True,
        "external_feature_fields": sorted(EXPECTED_OUTBOUND_FIELDS),
        "external_reference_fields": sorted(EXPECTED_OUTBOUND_FIELDS),
        "external_feedback_labels": ["PASS", "BLOCK", "INVALID"],
        "temporary_identifiers_only": True,
        "contains_month7": False,
        "contains_local_paths": False,
        "contains_credentials": False,
        "contains_researcher_only_diagnostics": False,
        "payloads": [
            {
                "temporary_anchor_id": f"dev-anchor-mock-{index:02d}",
                "payload_sha256": sha256_json([anchor_id, kwargs["prompt_version"]]),
                "preflight": "PASS",
            }
            for index, anchor_id in enumerate(anchor_ids, 1)
        ],
        "model_configs": [],
        "mock_only_not_scientific": True,
    }


def _mock_run_variant(**kwargs: Any) -> dict[str, Any]:
    anchor_ids = [str(value) for value in kwargs["anchor_ids"]]
    run_dir = Path(kwargs["run_dir"])
    prompt_version = str(kwargs["prompt_version"])
    formal = A3ModelConfig(
        **{
            **FORMAL_A3_MODEL_CONFIG.to_dict(),
            "prompt_version": prompt_version,
            "temperature": float(kwargs["temperature"]),
            "max_local_generation_attempts_per_query": int(
                kwargs["max_local_generation_attempts_per_query"]
            ),
        }
    )
    per_anchor = []
    for anchor_id in anchor_ids:
        manifest_path = (
            run_dir
            / f"anchor_{anchor_id}"
            / "query_01"
            / "local_gen_01"
            / "outbound_payload_manifest.json"
        )
        write_json_new(
            manifest_path,
            {
                "preflight": "PASS",
                "payload_sha256": sha256_json([anchor_id, prompt_version, "mock"]),
                "mock_only_not_scientific": True,
            },
        )
        per_anchor.append(
            {
                "anchor_id": anchor_id,
                "success": False,
                "stop_reason": "mock_only",
                "queries_used": 0,
                "attempts_to_success": None,
            }
        )
    return {
        "status": "mock_orchestration_only_not_scientific",
        "variant": kwargs["variant"],
        "condition_id": f"{kwargs['variant']}_t0",
        "variant_label": PROMPT_VARIANT_LABELS[prompt_version],
        "prompt_version": prompt_version,
        "config_hash": formal.config_hash(),
        "n_anchors": len(anchor_ids),
        "q_max": Q_MAX,
        "m_max": M_MAX,
        "asr_curve": {f"ASR@{query}": 0.0 for query in range(1, Q_MAX + 1)},
        "asr_curve_auc_mean": 0.0,
        "mean_queries_to_success": None,
        "total_llm_calls": 0,
        "token_usage": {
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens_derived": 0,
            "completion_tokens": 0,
        },
        "total_estimated_cost_usd": 0.0,
        "per_anchor": per_anchor,
        "model_config": formal.to_dict(),
        "mock_only_not_scientific": True,
    }


@dataclass
class StageBGridRunner:
    output_root: Path
    anchors_file: Path = DEFAULT_ANCHORS_FILE
    raw_path: Path = DEFAULT_RAW_PATH
    artefact_dir: Path = DEFAULT_C1_ARTEFACT_DIR
    resume: bool = False
    launch: bool = False
    mock: bool = False
    authorization_confirmation: str | None = None
    single_cell_runner: SingleCellRunner = run_variant
    preflight_runner: PreflightRunner = preflight_outbound_payloads
    client_factory: Callable[[str], LLMCompletionClient] = (
        lambda model: DeepSeekPlannerClient(default_model=model)
    )

    @property
    def mode(self) -> str:
        return "mock" if self.mock else "live" if self.launch else "dry_run"

    @property
    def guard_ledger_path(self) -> Path:
        return self.output_root / "global_cost_guard.jsonl"

    def _validate_mode(self) -> None:
        if self.launch and self.mock:
            raise GridRunnerError("--launch and --mock are mutually exclusive.")
        if self.launch and self.authorization_confirmation != LIVE_CONFIRMATION:
            raise GridRunnerError("Live execution requires the exact second unlock.")
        if self.authorization_confirmation and not self.launch:
            raise GridRunnerError("Authorization confirmation without --launch is refused.")

    def _initialise_grid(self, anchor_ids: Sequence[str]) -> dict[str, Any]:
        manifest_path = self.output_root / "grid_manifest.json"
        if self.output_root.exists():
            if not self.resume or not manifest_path.is_file():
                raise FileExistsError(
                    f"Existing output root is refused without valid --resume: {self.output_root}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("runner_version") != RUNNER_VERSION:
                raise GridRunnerError("Runner version drift on resume.")
            if manifest.get("mode") != self.mode:
                raise GridRunnerError("Execution mode drift on resume.")
            if manifest.get("anchor_set_sha256") != sha256_json(list(anchor_ids)):
                raise GridRunnerError("Anchor set drift on resume.")
            return manifest
        if self.resume:
            raise FileNotFoundError("--resume root does not exist.")
        self.output_root.mkdir(parents=True, exist_ok=False)
        cells = {}
        for spec in CELL_PROTOCOL:
            cell_dir = self.output_root / "cells" / spec.cell_id
            cell_dir.mkdir(parents=True, exist_ok=False)
            cell_manifest = {
                "cell_id": spec.cell_id,
                "replicate": spec.replicate,
                "condition": spec.condition,
                "execution_order": spec.order,
                "prompt_version": spec.prompt_version,
                "config_hash": model_config_for(spec).config_hash(),
                "status": "planned",
                "attempts": [],
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
            }
            write_json_new(cell_dir / "cell_manifest.json", cell_manifest)
            cells[spec.cell_id] = {"status": "planned"}
        manifest = {
            "runner_version": RUNNER_VERSION,
            "status": "planned",
            "mode": self.mode,
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "cost_cap_usd": COST_CAP_USD,
            "anchor_set_sha256": sha256_json(list(anchor_ids)),
            "n_fixed_anchors": len(anchor_ids),
            "protocol": [
                {
                    "cell_id": spec.cell_id,
                    "replicate": spec.replicate,
                    "condition": spec.condition,
                    "order": spec.order,
                    "prompt_version": spec.prompt_version,
                    "config_hash": model_config_for(spec).config_hash(),
                }
                for spec in CELL_PROTOCOL
            ],
            "cells": cells,
            "preflight": {},
        }
        write_json_new(manifest_path, manifest)
        return manifest

    def _run_existing_preflights(
        self, *, anchor_ids: Sequence[str], manifest: dict[str, Any]
    ) -> None:
        preflight_dir = self.output_root / "preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        budget = AttackBudget(q_max=Q_MAX, m_max=M_MAX)
        for condition, prompt_version in (
            ("B0", PROMPT_VERSION_P1_COMPACT),
            ("B1", PROMPT_VERSION_B1_NEUTRAL_GROUNDED),
            ("B2", PROMPT_VERSION_B2_GROUNDED_REFLECTION),
        ):
            payload = self.preflight_runner(
                anchor_ids=anchor_ids,
                budget=budget,
                experiment_seed=DEFAULT_EXPERIMENT_SEED,
                raw_path=self.raw_path,
                artefact_dir=self.artefact_dir,
                temperatures=(TEMPERATURE,),
                prompt_version=prompt_version,
                max_local_generation_attempts_per_query=MAX_LOCAL_GENERATIONS,
            )
            _validate_existing_preflight(payload, condition=condition)
            path = preflight_dir / f"{condition.lower()}_outbound_preflight.json"
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if sha256_json(existing) != sha256_json(payload):
                    raise GridRunnerError(f"Preflight drift on resume for {condition}.")
            else:
                write_json_new(path, payload)
            manifest["preflight"][condition] = {
                "status": "PASS",
                "manifest": str(path.relative_to(self.output_root)),
                "manifest_sha256": sha256_file(path),
                "prompt_version": prompt_version,
                "external_feature_fields": list(payload["external_feature_fields"]),
            }
        manifest["updated_utc"] = utc_now()
        write_json_atomic(self.output_root / "grid_manifest.json", manifest)

    def _attempt_id(self, cell_manifest: Mapping[str, Any]) -> str:
        return f"attempt_{len(cell_manifest.get('attempts') or []) + 1:02d}"

    def run(self) -> dict[str, Any]:
        self._validate_mode()
        anchor_ids = load_fixed_anchors(self.anchors_file)
        manifest = self._initialise_grid(anchor_ids)
        self._run_existing_preflights(anchor_ids=anchor_ids, manifest=manifest)
        if manifest.get("status") == "completed":
            return {
                "status": "completed",
                "already_completed": True,
                "external_api_calls": 0 if self.mock else None,
                "summary": str(self.output_root / "stage_b_grid_summary.json"),
            }
        if self.mode == "dry_run":
            manifest["status"] = "preflight_complete_no_api"
            manifest["updated_utc"] = utc_now()
            write_json_atomic(self.output_root / "grid_manifest.json", manifest)
            return {
                "status": manifest["status"],
                "external_api_calls": 0,
                "cells_planned": 9,
                "outbound_preflights_passed": 3,
                "output_root": str(self.output_root),
            }

        completed_rows: list[dict[str, Any]] = []
        for spec in CELL_PROTOCOL:
            cell_dir = self.output_root / "cells" / spec.cell_id
            cell_manifest_path = cell_dir / "cell_manifest.json"
            cell_manifest = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
            if cell_manifest.get("status") == "completed":
                summary_path = self.output_root / cell_manifest["summary_path"]
                completed_rows.append(
                    {
                        "cell_id": spec.cell_id,
                        "condition": spec.condition,
                        "replicate": spec.replicate,
                        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                    }
                )
                continue
            if cell_manifest.get("status") not in {"planned", "infrastructure_failed"}:
                raise GridRunnerError(
                    f"Unsafe resume state for {spec.cell_id}: {cell_manifest.get('status')}"
                )
            current_cost = guard_total_cost(self.guard_ledger_path)
            if current_cost >= COST_CAP_USD:
                manifest["status"] = "cost_cap_reached"
                manifest["updated_utc"] = utc_now()
                write_json_atomic(self.output_root / "grid_manifest.json", manifest)
                return {
                    "status": "cost_cap_reached",
                    "total_cost_usd": current_cost,
                    "external_api_calls": 0 if self.mock else None,
                }

            attempt_id = self._attempt_id(cell_manifest)
            attempt_dir = cell_dir / attempt_id
            attempt_dir.mkdir(parents=False, exist_ok=False)
            attempt_manifest_path = attempt_dir / "attempt_manifest.json"
            attempt_manifest = {
                "attempt_id": attempt_id,
                "cell_id": spec.cell_id,
                "status": "running",
                "started_utc": utc_now(),
                "mock_only_not_scientific": self.mock,
            }
            write_json_new(attempt_manifest_path, attempt_manifest)
            cell_manifest["status"] = "running"
            cell_manifest["attempts"].append(
                {"attempt_id": attempt_id, "status": "running"}
            )
            cell_manifest["updated_utc"] = utc_now()
            write_json_atomic(cell_manifest_path, cell_manifest)
            manifest["status"] = "running"
            manifest["cells"][spec.cell_id]["status"] = "running"
            manifest["updated_utc"] = utc_now()
            write_json_atomic(self.output_root / "grid_manifest.json", manifest)

            config = model_config_for(spec)
            delegate = _NoCallClient() if self.mock else self.client_factory(config.model)
            guarded_client = GlobalCostGuardClient(
                delegate=delegate,
                ledger_path=self.guard_ledger_path,
                cell_id=spec.cell_id,
                attempt_id=attempt_id,
            )
            try:
                summary = self.single_cell_runner(
                    variant=spec.condition,
                    anchor_ids=anchor_ids,
                    budget=AttackBudget(q_max=Q_MAX, m_max=M_MAX),
                    experiment_seed=DEFAULT_EXPERIMENT_SEED,
                    run_dir=attempt_dir,
                    raw_path=self.raw_path,
                    artefact_dir=self.artefact_dir,
                    temperature=TEMPERATURE,
                    max_local_generation_attempts_per_query=MAX_LOCAL_GENERATIONS,
                    prompt_version=spec.prompt_version,
                    llm_client=guarded_client,
                )
                _validate_existing_cell_summary(summary, spec=spec, anchor_ids=anchor_ids)
                runtime_evidence = _runtime_manifest_evidence(attempt_dir)
                guard_cost = guard_attempt_cost(
                    self.guard_ledger_path,
                    cell_id=spec.cell_id,
                    attempt_id=attempt_id,
                )
                summary_cost = float(summary["total_estimated_cost_usd"])
                if not self.mock and abs(guard_cost - summary_cost) > 1e-9:
                    raise GridRunnerError(
                        "Global guard cost disagrees with existing authoritative cell summary."
                    )
            except GlobalCostCapReached as exc:
                attempt_manifest["status"] = "cost_cap_stopped"
                attempt_manifest["finished_utc"] = utc_now()
                attempt_manifest["stop_reason"] = str(exc)
                write_json_atomic(attempt_manifest_path, attempt_manifest)
                cell_manifest = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
                cell_manifest["status"] = "cost_cap_stopped"
                cell_manifest["attempts"][-1]["status"] = "cost_cap_stopped"
                cell_manifest["updated_utc"] = utc_now()
                write_json_atomic(cell_manifest_path, cell_manifest)
                manifest["status"] = "cost_cap_reached"
                manifest["cells"][spec.cell_id]["status"] = "cost_cap_stopped"
                manifest["updated_utc"] = utc_now()
                write_json_atomic(self.output_root / "grid_manifest.json", manifest)
                return {
                    "status": "cost_cap_reached",
                    "cell_id": spec.cell_id,
                    "total_cost_usd": guard_total_cost(self.guard_ledger_path),
                    "external_api_calls": 0 if self.mock else None,
                }
            except Exception as exc:
                attempt_manifest["status"] = "infrastructure_failed"
                attempt_manifest["finished_utc"] = utc_now()
                attempt_manifest["error_type"] = type(exc).__name__
                attempt_manifest["error_sha256"] = hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest()
                write_json_atomic(attempt_manifest_path, attempt_manifest)
                cell_manifest = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
                cell_manifest["status"] = "infrastructure_failed"
                cell_manifest["attempts"][-1]["status"] = "infrastructure_failed"
                cell_manifest["updated_utc"] = utc_now()
                write_json_atomic(cell_manifest_path, cell_manifest)
                manifest["status"] = "infrastructure_failed"
                manifest["cells"][spec.cell_id]["status"] = "infrastructure_failed"
                manifest["updated_utc"] = utc_now()
                write_json_atomic(self.output_root / "grid_manifest.json", manifest)
                return {
                    "status": "infrastructure_failed",
                    "cell_id": spec.cell_id,
                    "resume_permitted": True,
                    "external_api_calls": 0 if self.mock else None,
                }

            summary_path = attempt_dir / "summary.json"
            write_json_new(summary_path, summary)
            attempt_manifest.update(
                {
                    "status": "completed",
                    "finished_utc": utc_now(),
                    "summary_sha256": sha256_file(summary_path),
                    "guard_cost_usd": guard_cost,
                    "runtime_outbound_manifests": runtime_evidence,
                }
            )
            write_json_atomic(attempt_manifest_path, attempt_manifest)
            cell_manifest = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
            cell_manifest["status"] = "completed"
            cell_manifest["attempts"][-1]["status"] = "completed"
            cell_manifest["successful_attempt"] = attempt_id
            cell_manifest["summary_path"] = str(summary_path.relative_to(self.output_root))
            cell_manifest["summary_sha256"] = sha256_file(summary_path)
            cell_manifest["finished_utc"] = utc_now()
            cell_manifest["updated_utc"] = utc_now()
            write_json_atomic(cell_manifest_path, cell_manifest)
            manifest["cells"][spec.cell_id]["status"] = "completed"
            manifest["updated_utc"] = utc_now()
            write_json_atomic(self.output_root / "grid_manifest.json", manifest)
            completed_rows.append(
                {
                    "cell_id": spec.cell_id,
                    "condition": spec.condition,
                    "replicate": spec.replicate,
                    "summary": summary,
                }
            )
            if guard_total_cost(self.guard_ledger_path) >= COST_CAP_USD:
                manifest["status"] = "cost_cap_reached"
                manifest["updated_utc"] = utc_now()
                write_json_atomic(self.output_root / "grid_manifest.json", manifest)
                return {
                    "status": "cost_cap_reached",
                    "last_completed_cell": spec.cell_id,
                    "total_cost_usd": guard_total_cost(self.guard_ledger_path),
                    "external_api_calls": 0 if self.mock else None,
                }

        aggregate = aggregate_existing_summaries(
            cell_rows=completed_rows,
            total_guard_cost_usd=guard_total_cost(self.guard_ledger_path),
        )
        if self.mock:
            aggregate["mock_only_not_scientific"] = True
            aggregate["mock_virtual_episode_count"] = sum(
                int(row["summary"]["n_anchors"]) for row in completed_rows
            )
        summary_path = self.output_root / "stage_b_grid_summary.json"
        write_json_new(summary_path, aggregate)
        manifest["status"] = "completed"
        manifest["completed_utc"] = utc_now()
        manifest["updated_utc"] = utc_now()
        write_json_atomic(self.output_root / "grid_manifest.json", manifest)
        return {
            "status": "completed",
            "cells_completed": 9,
            "mock_virtual_episode_count": 225 if self.mock else None,
            "external_api_calls": 0 if self.mock else None,
            "summary": str(summary_path),
            "total_cost_usd": guard_total_cost(self.guard_ledger_path),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchors-file", type=Path, default=DEFAULT_ANCHORS_FILE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--authorization-confirmation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = StageBGridRunner(
        output_root=args.output_root.resolve(),
        anchors_file=args.anchors_file.resolve(),
        raw_path=args.raw.resolve(),
        artefact_dir=args.artefact_dir.resolve(),
        resume=bool(args.resume),
        launch=bool(args.launch),
        mock=bool(args.mock),
        authorization_confirmation=args.authorization_confirmation,
        single_cell_runner=_mock_run_variant if args.mock else run_variant,
        preflight_runner=(
            _mock_preflight_outbound_payloads
            if args.mock
            else preflight_outbound_payloads
        ),
    )
    result = runner.run()
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
    return 0 if result["status"] in {"completed", "preflight_complete_no_api"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
