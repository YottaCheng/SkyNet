#!/usr/bin/env python3
"""Frozen Month-6 A0–A3 confirmatory comparison.

Locked configuration:
  A0 random
  A1 V4.3 + deepseek-v4-pro + thinking disabled
  A2 search (existing public-reference Gower v2)
  A3 V2.3 + deepseek-v4-pro + thinking disabled

Same-arena Month-6 N=50, K=10, Q=5, m=2. Month 7 remains closed.
No prompt/model/ablation/parameter changes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SCRIPTS = IMPL / "scripts"
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

import run_dev_model_selection as base  # noqa: E402
from attack_lab.benchmark_pins import (  # noqa: E402
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
    PINNED_GOVERNANCE_FINGERPRINT,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
    BenchmarkPinError,
    pinned_attacker_summary,
)
from attack_lab.cases import DEFAULT_RAW_PATH  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.experiment_config import sha256_file  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory  # noqa: E402
from attack_lab.reference_pool import ReferencePoolProvider  # noqa: E402
from attack_lab.types import to_jsonable  # noqa: E402
from baf_data.config import FROZEN_CONFIG  # noqa: E402

N_FROZEN = 50
OUTPUT_PARENT = (
    ROOT / "05_outputs" / "experiments" / "comparisons" / "a0_a3" / "month6_frozen"
)
EXPECTED_A1 = "a1_oneshot_v4_3_public_reference_view"
EXPECTED_A3 = "a3_episodic_reflective_v2_3_public_reference_view"
FORBIDDEN_PROMPT_SUBSTRINGS = (
    "v4_4",
    "v2_4",
    "adversarial_objective",
)
CONDITIONS: tuple[tuple[str, str, str | None], ...] = (
    ("A0", "a0", None),
    ("A1-Pro", "a1", MODEL_PRO),
    ("A2", "a2", None),
    ("A3-Pro", "a3", MODEL_PRO),
)
INFRA_STOPS = frozenset(
    {
        "runner_exception",
        "timeout",
        "transport_error",
        "rate_limit",
        "raw_source_integrity",
    }
)
INTEGRITY_STOPS = frozenset(
    {
        "q_violation",
        "m_violation",
        "provenance_violation",
        "hidden_exposure",
    }
)


def wilson_interval(k: int, n: int, *, z: float = 1.96) -> dict[str, float | None]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return {"k": k, "n": n, "p": None, "low": None, "high": None, "method": "wilson_95"}
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    margin = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / den
    return {
        "k": k,
        "n": n,
        "p": p,
        "low": max(0.0, centre - margin),
        "high": min(1.0, centre + margin),
        "method": "wilson_95",
    }


def assert_no_deprecated_contracts() -> None:
    for mod in (
        "attack_lab.attackers.a1_v4_4_contract",
        "attack_lab.attackers.a3_v2_4_contract",
    ):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            continue
        raise BenchmarkPinError(f"Deprecated contract still importable: {mod}")


def hard_preflight(
    *,
    raw_path: Path,
    policy: CompiledGovernancePolicy,
    defender: FrozenXGBoostDefender,
    provider: ReferencePoolProvider,
    anchors: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if str(PINNED_A1_PROMPT_VERSION) != EXPECTED_A1:
        errors.append(f"A1 pin mismatch: {PINNED_A1_PROMPT_VERSION!r}")
    if str(PINNED_A3_PROMPT_VERSION) != EXPECTED_A3:
        errors.append(f"A3 pin mismatch: {PINNED_A3_PROMPT_VERSION!r}")
    if str(PINNED_A2_GOWER_POLICY) != "a2_public_reference_gower_v2":
        errors.append(f"A2 Gower pin mismatch: {PINNED_A2_GOWER_POLICY!r}")
    if policy.policy_fingerprint != PINNED_GOVERNANCE_FINGERPRINT:
        errors.append("governance fingerprint mismatch")
    if defender.artefact_id != PINNED_D1_ARTEFACT_ID:
        errors.append("D1 artefact id mismatch")
    if "month7" in str(DEFAULT_C1_ARTEFACT_DIR).lower():
        errors.append("Month-7 path in D1 artefact dir")
    try:
        assert_no_deprecated_contracts()
    except BenchmarkPinError as exc:
        errors.append(str(exc))

    raw_sha = sha256_file(raw_path)
    if raw_sha != FROZEN_CONFIG.expected_sha256:
        errors.append("raw dataset SHA-256 mismatch")

    first_pool = provider.get_pool(str(anchors[0]), seed=base.REFERENCE_POOL_SEED)
    probes: list[dict[str, Any]] = []
    for cid, kind, model in CONDITIONS:
        attacker = base.build_attacker(
            condition_id=cid,
            attacker_kind=kind,
            llm_model=model,
            pool=first_pool,
            recorder=None,
            thinking_disabled=True,
            reasoning_effort=None,
        )
        prompt = getattr(attacker, "prompt_version", None)
        gower = getattr(attacker, "gower_policy", None)
        thinking = getattr(attacker, "thinking_disabled", None)
        effort = getattr(attacker, "reasoning_effort", None)
        live_model = getattr(attacker, "model", None)
        if kind == "a1":
            if str(prompt) != EXPECTED_A1:
                errors.append(f"{cid} prompt {prompt!r}")
            if str(live_model) != MODEL_PRO:
                errors.append(f"{cid} model {live_model!r}")
            if thinking is not True:
                errors.append(f"{cid} thinking not disabled")
            if effort not in {None, ""}:
                errors.append(f"{cid} reasoning_effort={effort!r}")
        if kind == "a3":
            if str(prompt) != EXPECTED_A3:
                errors.append(f"{cid} prompt {prompt!r}")
            if str(live_model) != MODEL_PRO:
                errors.append(f"{cid} model {live_model!r}")
            if thinking is not True:
                errors.append(f"{cid} thinking not disabled")
            if effort not in {None, ""}:
                errors.append(f"{cid} reasoning_effort={effort!r}")
        if kind == "a2" and str(gower) != PINNED_A2_GOWER_POLICY:
            errors.append(f"{cid} gower {gower!r}")
        blob = json.dumps(
            {"prompt": prompt, "gower": gower, "model": live_model},
            sort_keys=True,
        )
        if any(s in blob.lower() for s in FORBIDDEN_PROMPT_SUBSTRINGS):
            errors.append(f"{cid} loaded forbidden prompt token")
        probes.append(
            {
                "condition_id": cid,
                "attacker_kind": kind,
                "prompt_version": prompt,
                "gower_policy": gower,
                "model": live_model,
                "thinking_disabled": thinking,
                "reasoning_effort": effort,
            }
        )

    pool_fps = {
        str(aid): provider.get_pool(str(aid), seed=base.REFERENCE_POOL_SEED).pool_fingerprint
        for aid in anchors
    }
    if len(pool_fps) != len(anchors):
        errors.append("reference-pool fingerprint map incomplete")
    if errors:
        raise SystemExit("PRE-RUN HARD CHECK FAILED:\n- " + "\n- ".join(errors))
    return {
        "status": "PRE-RUN HARD CHECK PASSED",
        "anchors": list(anchors),
        "n": len(anchors),
        "raw_sha256": raw_sha,
        "d1_artefact_id": defender.artefact_id,
        "governance_fingerprint": policy.policy_fingerprint,
        "reference_pool_fingerprints": pool_fps,
        "probes": probes,
        "pins": pinned_attacker_summary(),
        "month7_opened": False,
        "deepseek_calls": 0,
    }


def latency_stats(cond_dir: Path) -> dict[str, Any]:
    latencies: list[float] = []
    for path in cond_dir.rglob("llm_transport_identity.json"):
        ident = json.loads(path.read_text(encoding="utf-8"))
        for call in ident.get("calls") or []:
            latencies.append(float(call.get("latency_ms") or 0.0))
    return {
        "llm_calls_with_latency": len(latencies),
        "latency_ms_sum": sum(latencies) if latencies else 0.0,
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else None,
    }


def classify_row(row: Mapping[str, Any]) -> str:
    if row.get("runner_exception"):
        return "infrastructure_failure"
    stop = str(row.get("stop_reason") or "")
    if stop in INFRA_STOPS or "integrity" in stop.lower() and "raw" in stop.lower():
        return "infrastructure_failure"
    if any(
        int(row.get(k) or 0) > 0
        for k in (
            "Q_violations",
            "m_violations",
            "hidden_exposure",
            "raw_proxy_exposure",
            "non_reference_backed",
            "post_freeze_adaptation",
        )
    ):
        return "integrity_failure"
    if stop in INTEGRITY_STOPS:
        return "integrity_failure"
    if row.get("success"):
        return "success"
    return "attack_failure"


def condition_report(rows: Sequence[Mapping[str, Any]], cond_dir: Path) -> dict[str, Any]:
    completed = [r for r in rows if not r.get("runner_exception")]
    n = len(rows)
    successes = sum(1 for r in completed if r.get("success"))
    classes = Counter(classify_row(r) for r in rows)
    integ = base.condition_integrity(rows)
    curve = base.asr_curve(rows)
    asr_ci = {
        key: wilson_interval(
            sum(
                1
                for r in completed
                if r.get("attempts_to_success") is not None
                and int(r["attempts_to_success"]) <= int(key.split("@")[1])
            ),
            n,
        )
        for key in curve
    }
    return {
        "n_anchors": n,
        "completed": len(completed),
        "successes": successes,
        "success_rate": (successes / n) if n else None,
        "success_rate_wilson_95": wilson_interval(successes, n),
        "asr_curve": curve,
        "asr_curve_wilson_95": asr_ci,
        "stop_reasons": dict(Counter(str(r.get("stop_reason")) for r in rows)),
        "failure_classes": dict(classes),
        "integrity": integ,
        "cost_usd": integ.get("estimated_cost_usd"),
        "prompt_tokens": integ.get("prompt_tokens"),
        "completion_tokens": integ.get("completion_tokens"),
        "llm_calls": integ.get("llm_calls"),
        **latency_stats(cond_dir),
        "requested_models": integ.get("requested_models"),
        "returned_models": integ.get("returned_models"),
        "system_fingerprints": integ.get("system_fingerprints"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        raise SystemExit(f"Raw dataset missing: {raw_path}")
    if "month7" in str(raw_path).lower():
        raise SystemExit("Refusing Month-7 raw path.")

    policy = CompiledGovernancePolicy.load(base.GOVERNANCE_PATH)
    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    base.preflight_defence_and_pins(
        policy=policy,
        defender=defender,
        artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
    )
    provider = ReferencePoolProvider.from_config(
        base.build_pool_config(), raw_path=raw_path
    )
    anchors = base.resolve_anchors(n=N_FROZEN)
    if len(anchors) != N_FROZEN:
        raise SystemExit(f"Expected {N_FROZEN} anchors, got {len(anchors)}.")
    base.verify_same_arena(anchors, provider, defender, raw_path)

    preflight = hard_preflight(
        raw_path=raw_path,
        policy=policy,
        defender=defender,
        provider=provider,
        anchors=anchors,
    )
    if args.preflight_only:
        print(json.dumps(to_jsonable(preflight), indent=2, sort_keys=True))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"month6_frozen_a0_a3_n{N_FROZEN}_m{base.M_MAX}_Q{base.Q_MAX}_"
        f"seed{base.EXPERIMENT_SEED}_{stamp}",
        parent=OUTPUT_PARENT,
        stage="experiments",
    )
    if "month7" in str(run_dir).lower():
        raise SystemExit("Refusing Month-7 output path.")

    frozen_manifest = {
        "status": "month6_frozen_a0_a3_confirmatory",
        "created_utc": base.utc_now(),
        "month7_opened": False,
        "n_anchors": N_FROZEN,
        "anchors": list(anchors),
        "K": base.K,
        "Q": base.Q_MAX,
        "m": base.M_MAX,
        "experiment_seed": base.EXPERIMENT_SEED,
        "reference_pool_seed": base.REFERENCE_POOL_SEED,
        "feedback_mode": base.FEEDBACK_MODE,
        "dataset": {
            "name": "BAF Base",
            "split": "dev_month6",
            "raw_path": str(raw_path),
            "sha256": preflight["raw_sha256"],
        },
        "d1_artefact_id": defender.artefact_id,
        "governance_fingerprint": policy.policy_fingerprint,
        "require_reference_provenance": PINNED_REQUIRE_REFERENCE_PROVENANCE,
        "pins": pinned_attacker_summary(),
        "conditions": [
            {
                "condition_id": cid,
                "attacker_kind": kind,
                "model": model,
                "thinking": "disabled" if kind in {"a1", "a3"} else None,
                "reasoning_effort": None,
                "prompt_version": (
                    PINNED_A1_PROMPT_VERSION
                    if kind == "a1"
                    else PINNED_A3_PROMPT_VERSION
                    if kind == "a3"
                    else None
                ),
                "gower_policy": PINNED_A2_GOWER_POLICY if kind == "a2" else None,
            }
            for cid, kind, model in CONDITIONS
        ],
        "reference_pool_fingerprints": preflight["reference_pool_fingerprints"],
        "note": (
            "Frozen Month-6 confirmatory A0–A3 comparison. "
            "Not Month-7. D2 is not included."
        ),
    }
    base.write_json(run_dir / "FROZEN_MANIFEST.json", frozen_manifest)
    base.write_json(run_dir / "PREFLIGHT.json", preflight)

    expected_pools = dict(preflight["reference_pool_fingerprints"])
    by_condition: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    for cid, kind, model in CONDITIONS:
        print(f"[{base.utc_now()}] START {cid}", flush=True)
        rows = base.run_condition(
            run_dir=run_dir,
            condition_id=cid,
            attacker_kind=kind,
            llm_model=model,
            anchors=anchors,
            policy=policy,
            defender=defender,
            provider=provider,
            raw_path=raw_path,
            thinking_disabled=True,
            reasoning_effort=None,
            expected_pool_fingerprints=expected_pools,
        )
        by_condition[cid] = rows
        reports[cid] = condition_report(rows, run_dir / cid)
        base.write_json(run_dir / cid / "condition_report.json", reports[cid])

    failure_summary = {
        "attack_failure": {},
        "infrastructure_failure": {},
        "integrity_failure": {},
        "success": {},
    }
    for cid, rows in by_condition.items():
        for cls in failure_summary:
            items = [r for r in rows if classify_row(r) == cls]
            failure_summary[cls][cid] = {
                "n": len(items),
                "stop_reasons": dict(Counter(str(r.get("stop_reason")) for r in items)),
                "anchor_ids": [r.get("anchor_id") for r in items],
            }

    experiment_report = {
        "status": "month6_frozen_a0_a3_complete",
        "month7_opened": False,
        "n_anchors": N_FROZEN,
        "anchors": list(anchors),
        "K": base.K,
        "Q": base.Q_MAX,
        "m": base.M_MAX,
        "conditions": reports,
        "per_attacker_asr5": {
            cid: reports[cid]["asr_curve"].get("ASR@5") for cid in reports
        },
        "total_estimated_cost_usd": sum(
            float(reports[cid].get("cost_usd") or 0.0) for cid in reports
        ),
        "integrity_all_ok": all(
            int(reports[cid]["integrity"].get("Q_violations") or 0) == 0
            and int(reports[cid]["integrity"].get("m_violations") or 0) == 0
            and int(reports[cid]["integrity"].get("hidden_exposure") or 0) == 0
            and int(reports[cid]["integrity"].get("raw_proxy_exposure") or 0) == 0
            and int(reports[cid]["integrity"].get("non_reference_backed") or 0) == 0
            and int(reports[cid]["integrity"].get("runner_exceptions") or 0) == 0
            for cid in reports
        ),
        "automatic_winner": None,
        "note": (
            "Frozen Month-6 confirmatory comparison under locked pins. "
            "Not Month-7. Not a hybrid D1+D2 result."
        ),
    }
    base.write_json(run_dir / "EXPERIMENT_REPORT.json", experiment_report)
    base.write_json(run_dir / "FAILURE_SUMMARY.json", failure_summary)
    print(json.dumps(to_jsonable(experiment_report), indent=2, sort_keys=True), flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
