#!/usr/bin/env python3
"""Ablation sanity: A3 V2.3 + DeepSeek Pro + Thinking ON (NOT findings).

Isolates thinking-mode / API compatibility from the V2.4 prompt revision.
Uses the frozen previous A3 prompt version only for this run:

  a3_episodic_reflective_v2_3_public_reference_view

Does not modify V2.4 files, prompts, schemas, governance, K/Q/m, D1, or Month 7.
Thinking-enabled requests use max_tokens=2000 (ThinkOff default 800 unchanged).
N=5 Month-6 same-arena anchors only. Not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SCRIPTS = IMPL / "scripts"
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

import run_dev_model_selection as base  # noqa: E402
from attack_lab.attackers.a1_planner import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    THINKING_ENABLED_MAX_TOKENS,
    resolve_max_tokens,
)
from attack_lab.attackers.a3_agent import EpisodicLLMAgent  # noqa: E402
from attack_lab.attackers.a3_v2_3_contract import (  # noqa: E402
    PROMPT_VERSION_A3_V2_3,
)
from attack_lab.benchmark_pins import (  # noqa: E402
    MODEL_PRO,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
    REASONING_EFFORT_MAX,
    BenchmarkPinError,
    assert_thinking_cell_config,
    pinned_attacker_summary,
)
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.cases import DEFAULT_RAW_PATH  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory  # noqa: E402
from attack_lab.reference_pool import ReferencePoolProvider  # noqa: E402
from attack_lab.types import to_jsonable  # noqa: E402

CONDITION_ID = "A3-Pro-V2_3-ThinkOn"
N_SANITY = 5
SANITY_PROMPT_VERSION = PROMPT_VERSION_A3_V2_3  # frozen previous version


def build_a3_v2_3_thinkon_attacker(*, pool, recorder):
    """Construct A3 with experiment-local V2.3 pin (does not touch global V2.4 pin)."""
    assert_thinking_cell_config(
        thinking_disabled=False,
        reasoning_effort=REASONING_EFFORT_MAX,
        expect_thinking_disabled=False,
    )
    max_tokens = resolve_max_tokens(
        thinking_disabled=False,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    if max_tokens != THINKING_ENABLED_MAX_TOKENS:
        raise BenchmarkPinError(
            f"Thinking-on max_tokens must be {THINKING_ENABLED_MAX_TOKENS}; "
            f"got {max_tokens}."
        )
    attacker = EpisodicLLMAgent(
        experiment_seed=base.EXPERIMENT_SEED,
        reference_pool=pool,
        budget=AttackBudget(q_max=base.Q_MAX, m_max=base.M_MAX),
        attacker_id="a3",
        prompt_version=SANITY_PROMPT_VERSION,
        model=MODEL_PRO,
        thinking_disabled=False,
        reasoning_effort=REASONING_EFFORT_MAX,
        max_tokens=DEFAULT_MAX_TOKENS,  # resolved to 2000 inside agent/client
        llm_client=recorder,
        stdout=None,
    )
    if str(attacker.prompt_version) != SANITY_PROMPT_VERSION:
        raise BenchmarkPinError(
            f"A3 prompt_version drift: expected {SANITY_PROMPT_VERSION!r}, "
            f"got {attacker.prompt_version!r}."
        )
    if str(attacker.model) != MODEL_PRO:
        raise BenchmarkPinError(
            f"model mismatch: expected {MODEL_PRO!r}, got {attacker.model!r}."
        )
    if bool(attacker.thinking_disabled):
        raise BenchmarkPinError("thinking_disabled must be False for this sanity.")
    if str(attacker.reasoning_effort or "") != REASONING_EFFORT_MAX:
        raise BenchmarkPinError(
            f"reasoning_effort must be {REASONING_EFFORT_MAX!r}."
        )
    if int(attacker.max_tokens) != THINKING_ENABLED_MAX_TOKENS:
        raise BenchmarkPinError(
            f"resolved max_tokens must be {THINKING_ENABLED_MAX_TOKENS}; "
            f"got {attacker.max_tokens}."
        )
    return attacker


def preflight(
    *,
    policy: CompiledGovernancePolicy,
    defender: FrozenXGBoostDefender,
    provider: ReferencePoolProvider,
    anchors: Sequence[str],
    raw_path: Path,
) -> dict[str, Any]:
    if SANITY_PROMPT_VERSION != "a3_episodic_reflective_v2_3_public_reference_view":
        raise BenchmarkPinError("Sanity pin is not frozen A3 V2.3.")
    base.preflight_defence_and_pins(
        policy=policy,
        defender=defender,
        artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
    )
    if "month7" in str(DEFAULT_C1_ARTEFACT_DIR).lower():
        raise BenchmarkPinError("Month-7 artefact path detected.")
    base.verify_same_arena(anchors, provider, defender, raw_path)
    pool = provider.get_pool(str(anchors[0]), seed=base.REFERENCE_POOL_SEED)
    attacker = build_a3_v2_3_thinkon_attacker(pool=pool, recorder=None)
    manifest = base.attacker_condition_manifest(
        condition_id=CONDITION_ID,
        attacker_kind="a3",
        llm_model=MODEL_PRO,
        attacker=attacker,
        thinking_disabled=False,
        reasoning_effort=REASONING_EFFORT_MAX,
    )
    manifest["max_tokens"] = int(attacker.max_tokens)
    manifest["experiment_prompt_pin"] = SANITY_PROMPT_VERSION
    manifest["global_benchmark_pins_unchanged"] = pinned_attacker_summary()
    manifest["note"] = (
        "Experiment-local A3 V2.3 pin for thinking ablation only; "
        "global PINNED_A3 remains V2.4."
    )
    return manifest


def summarise_sanity(rows: Sequence[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    completed = [r for r in rows if not r.get("runner_exception")]
    parse_ok = 0
    parse_total = 0
    empty_raw = 0
    raw_total = 0
    env_steps = 0
    submitted = 0
    for p in (run_dir / CONDITION_ID).rglob("a3_retry_ledger.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        for att in data.get("attempts") or []:
            parse_total += 1
            if att.get("parse_status") == "ok":
                parse_ok += 1
            rp = att.get("raw_response_path")
            if rp:
                raw_total += 1
                path = Path(rp)
                if path.exists() and path.stat().st_size == 0:
                    empty_raw += 1
    for p in (run_dir / CONDITION_ID).rglob("a3_query_record.json"):
        q = json.loads(p.read_text(encoding="utf-8"))
        if q.get("env_step_called"):
            env_steps += 1
        if q.get("submitted"):
            submitted += 1
    integ = base.condition_integrity(rows)
    return {
        "status": "development_a3_v2_3_thinkon_sanity_not_findings",
        "condition_id": CONDITION_ID,
        "prompt_version": SANITY_PROMPT_VERSION,
        "model": MODEL_PRO,
        "thinking_disabled": False,
        "reasoning_effort": REASONING_EFFORT_MAX,
        "max_tokens": THINKING_ENABLED_MAX_TOKENS,
        "n": len(rows),
        "completed": len(completed),
        "runner_exceptions": sum(1 for r in rows if r.get("runner_exception")),
        "success": f"{sum(1 for r in completed if r.get('success'))}/{len(rows)}",
        "parse_success_rate": (parse_ok / parse_total) if parse_total else None,
        "parse_ok": parse_ok,
        "parse_total": parse_total,
        "empty_content_count": empty_raw,
        "raw_response_files": raw_total,
        "env_step_called_count": env_steps,
        "submitted_candidate_count": submitted,
        "total_defender_queries": sum(
            int(r.get("scored_defender_queries") or r.get("q_used") or 0)
            for r in completed
        ),
        "q_used_sum": sum(int(r.get("q_used") or 0) for r in completed),
        "m_violations": integ.get("m_violations"),
        "hidden_exposure": integ.get("hidden_exposure"),
        "non_reference_backed": integ.get("non_reference_backed"),
        "integrity": integ,
        "estimated_cost_usd": integ.get("estimated_cost_usd"),
        "prompt_tokens": integ.get("prompt_tokens"),
        "completion_tokens": integ.get("completion_tokens"),
        "stop_reasons": {
            str(r.get("stop_reason")): sum(
                1 for x in rows if str(x.get("stop_reason")) == str(r.get("stop_reason"))
            )
            for r in rows
        },
        "month7_opened": False,
        "automatic_winner": None,
        "note": "Ablation sanity only. Not a benchmark and not dissertation findings.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate pins/config and exit without DeepSeek calls.",
    )
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        raise SystemExit(f"Raw dataset missing: {raw_path}")

    policy = CompiledGovernancePolicy.load(base.GOVERNANCE_PATH)
    missing_proxies = {
        "name_email_alignment",
        "home_phone_configuration",
        "mobile_phone_configuration",
    } - set(policy.available_action_keys)
    if missing_proxies:
        raise SystemExit(
            f"Governance missing abstract proxies: {sorted(missing_proxies)}"
        )
    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    provider = ReferencePoolProvider.from_config(
        base.build_pool_config(), raw_path=raw_path
    )
    # resolve_anchors(n=5) cannot validate the frozen leading-15 prefix; resolve 15 then slice.
    anchors15 = base.resolve_anchors(n=15)
    anchors = anchors15[:N_SANITY]
    if anchors != base.ANCHORS_15[:N_SANITY]:
        raise SystemExit("N=5 sanity anchors drifted from leading same-arena set.")

    manifest = preflight(
        policy=policy,
        defender=defender,
        provider=provider,
        anchors=anchors,
        raw_path=raw_path,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_OK",
                    "manifest": manifest,
                    "anchors": list(anchors),
                    "deepseek_calls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"dev_a3_v2_3_thinkon_sanity_n{N_SANITY}_m{base.M_MAX}_Q{base.Q_MAX}_"
        f"seed{base.EXPERIMENT_SEED}_{stamp}",
        parent=ROOT / "05_outputs" / "scratch" / "smoke",
        stage="scratch",
    )
    base.write_json(
        run_dir / "run_config.json",
        {
            "status": "development_a3_v2_3_thinkon_sanity_not_findings",
            "created_utc": base.utc_now(),
            "N": N_SANITY,
            "K": base.K,
            "Q": base.Q_MAX,
            "m": base.M_MAX,
            "experiment_seed": base.EXPERIMENT_SEED,
            "reference_pool_seed": base.REFERENCE_POOL_SEED,
            "condition_manifest": manifest,
            "require_reference_provenance": PINNED_REQUIRE_REFERENCE_PROVENANCE,
            "month7_opened": False,
            "d1_artefact_dir": str(DEFAULT_C1_ARTEFACT_DIR),
            "d1_artefact_id": defender.artefact_id,
            "raw_path": str(raw_path),
            "governance_fingerprint": policy.policy_fingerprint,
            "note": (
                "Ablation: A3 V2.3 + Pro + Thinking ON + max_tokens=2000. "
                "Not a Month-7 result and not a dissertation finding."
            ),
        },
    )
    base.write_json(
        run_dir / "sanity_anchors.json",
        {"anchor_ids": list(anchors), "N": N_SANITY},
    )

    # Monkey-patch build_attacker for this process only via local run_condition wrapper.
    expected_pools = {
        str(aid): provider.get_pool(str(aid), seed=base.REFERENCE_POOL_SEED).pool_fingerprint
        for aid in anchors
    }

    def _build_attacker(**kwargs):
        return build_a3_v2_3_thinkon_attacker(
            pool=kwargs["pool"],
            recorder=kwargs.get("recorder"),
        )

    original_build = base.build_attacker
    original_pin = base.PINNED_A3_PROMPT_VERSION
    base.build_attacker = _build_attacker  # type: ignore[assignment]
    # Local pin for prompt-mechanics audit only; restore global V2.4 pin after.
    base.PINNED_A3_PROMPT_VERSION = SANITY_PROMPT_VERSION
    try:
        rows = base.run_condition(
            run_dir=run_dir,
            condition_id=CONDITION_ID,
            attacker_kind="a3",
            llm_model=MODEL_PRO,
            anchors=anchors,
            policy=policy,
            defender=defender,
            provider=provider,
            raw_path=raw_path,
            thinking_disabled=False,
            reasoning_effort=REASONING_EFFORT_MAX,
            expected_pool_fingerprints=expected_pools,
        )
    finally:
        base.build_attacker = original_build  # type: ignore[assignment]
        base.PINNED_A3_PROMPT_VERSION = original_pin

    # Rewrite condition manifest with experiment-local pin (run_condition wrote V2.4 pin).
    base.write_json(run_dir / CONDITION_ID / "condition_manifest.json", manifest)
    report = summarise_sanity(rows, run_dir)
    report["manifest"] = manifest
    base.write_json(run_dir / "SANITY_REPORT.json", report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
