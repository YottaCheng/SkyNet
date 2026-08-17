#!/usr/bin/env python3
"""Runner-only ablation: A3 V2.3 + Pro reasoning-effort sanity (NOT findings).

Cells (N=5 same-arena Month-6 anchors):
  A3-Pro-ThinkOff      thinking disabled
  A3-Pro-ThinkHigh     thinking enabled, reasoning_effort=high
  A3-Pro-ThinkMinimal  thinking enabled, reasoning_effort=low

Uses frozen A3 V2.3 only for this run. Does not modify V2.4, prompts, schemas,
attacker logic, governance, K/Q/m, D1, or Month 7.

Official Chat Completions efforts only: low | high | max (fail-closed).
Early-stops a thinking cell if empty-content rate saturates on the first anchors.
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
from attack_lab.attackers.a3_v2_3_contract import PROMPT_VERSION_A3_V2_3  # noqa: E402
from attack_lab.benchmark_pins import (  # noqa: E402
    MODEL_PRO,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
    BenchmarkPinError,
    pinned_attacker_summary,
)
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.cases import DEFAULT_RAW_PATH  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory  # noqa: E402
from attack_lab.reference_pool import ReferencePoolProvider  # noqa: E402
from attack_lab.types import to_jsonable  # noqa: E402

# Official OpenAI-format Chat Completions values only (DeepSeek Thinking Mode docs).
OFFICIAL_REASONING_EFFORTS = frozenset({"low", "high", "max"})
ABLATION_PROMPT_VERSION = PROMPT_VERSION_A3_V2_3
N_SANITY = 5

# (condition_id, thinking_disabled, reasoning_effort)
CONDITIONS: tuple[tuple[str, bool, str | None], ...] = (
    ("A3-Pro-ThinkOff", True, None),
    ("A3-Pro-ThinkHigh", False, "high"),
    ("A3-Pro-ThinkMinimal", False, "low"),
)


def require_official_reasoning_effort(
    effort: str | None, *, thinking_disabled: bool
) -> str | None:
    if thinking_disabled:
        if effort not in {None, ""}:
            raise BenchmarkPinError(
                "ThinkOff must not set reasoning_effort; "
                f"got {effort!r}."
            )
        return None
    name = str(effort or "").strip()
    if name not in OFFICIAL_REASONING_EFFORTS:
        raise BenchmarkPinError(
            f"Unsupported reasoning_effort={effort!r}; "
            f"official Chat Completions values={sorted(OFFICIAL_REASONING_EFFORTS)}."
        )
    return name


def build_a3_v2_3_attacker(
    *,
    pool,
    recorder,
    thinking_disabled: bool,
    reasoning_effort: str | None,
):
    effort = require_official_reasoning_effort(
        reasoning_effort, thinking_disabled=thinking_disabled
    )
    if thinking_disabled:
        max_tokens = DEFAULT_MAX_TOKENS
    else:
        max_tokens = resolve_max_tokens(
            thinking_disabled=False, max_tokens=DEFAULT_MAX_TOKENS
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
        prompt_version=ABLATION_PROMPT_VERSION,
        model=MODEL_PRO,
        thinking_disabled=bool(thinking_disabled),
        reasoning_effort=effort,
        max_tokens=DEFAULT_MAX_TOKENS if thinking_disabled else DEFAULT_MAX_TOKENS,
        llm_client=recorder,
        stdout=None,
    )
    if str(attacker.prompt_version) != ABLATION_PROMPT_VERSION:
        raise BenchmarkPinError(
            f"prompt_version drift: expected {ABLATION_PROMPT_VERSION!r}, "
            f"got {attacker.prompt_version!r}."
        )
    if str(attacker.model) != MODEL_PRO:
        raise BenchmarkPinError(
            f"model mismatch: expected {MODEL_PRO!r}, got {attacker.model!r}."
        )
    if bool(attacker.thinking_disabled) != bool(thinking_disabled):
        raise BenchmarkPinError("thinking_disabled drifted after construction.")
    if thinking_disabled:
        if attacker.reasoning_effort not in {None, ""}:
            raise BenchmarkPinError("ThinkOff attacker still has reasoning_effort.")
        if int(attacker.max_tokens) != DEFAULT_MAX_TOKENS:
            raise BenchmarkPinError(
                f"ThinkOff max_tokens must remain {DEFAULT_MAX_TOKENS}."
            )
    else:
        if str(attacker.reasoning_effort) != str(effort):
            raise BenchmarkPinError(
                f"reasoning_effort drift: expected {effort!r}, "
                f"got {attacker.reasoning_effort!r}."
            )
        if int(attacker.max_tokens) != THINKING_ENABLED_MAX_TOKENS:
            raise BenchmarkPinError(
                f"resolved max_tokens must be {THINKING_ENABLED_MAX_TOKENS}."
            )
    return attacker


def cell_manifest(
    *,
    condition_id: str,
    thinking_disabled: bool,
    reasoning_effort: str | None,
    attacker: Any,
) -> dict[str, Any]:
    man = base.attacker_condition_manifest(
        condition_id=condition_id,
        attacker_kind="a3",
        llm_model=MODEL_PRO,
        attacker=attacker,
        thinking_disabled=thinking_disabled,
        reasoning_effort=reasoning_effort,
    )
    man["max_tokens"] = int(attacker.max_tokens)
    man["experiment_prompt_pin"] = ABLATION_PROMPT_VERSION
    man["official_reasoning_efforts"] = sorted(OFFICIAL_REASONING_EFFORTS)
    man["global_benchmark_pins_unchanged"] = pinned_attacker_summary()
    return man


def empty_content_stats(cond_dir: Path) -> dict[str, Any]:
    raw_total = 0
    raw_empty = 0
    parse_ok = 0
    parse_total = 0
    for p in cond_dir.rglob("a3_retry_ledger.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        for att in data.get("attempts") or []:
            parse_total += 1
            if att.get("parse_status") == "ok":
                parse_ok += 1
            rp = att.get("raw_response_path")
            if not rp:
                continue
            path = Path(rp)
            if path.exists():
                raw_total += 1
                if path.stat().st_size == 0:
                    raw_empty += 1
    env_steps = 0
    submitted = 0
    for p in cond_dir.rglob("a3_query_record.json"):
        q = json.loads(p.read_text(encoding="utf-8"))
        if q.get("env_step_called"):
            env_steps += 1
        if q.get("submitted"):
            submitted += 1
    return {
        "raw_total": raw_total,
        "raw_empty": raw_empty,
        "empty_content_rate": (raw_empty / raw_total) if raw_total else None,
        "parse_ok": parse_ok,
        "parse_total": parse_total,
        "parse_success_rate": (parse_ok / parse_total) if parse_total else None,
        "env_step_called": env_steps,
        "submitted": submitted,
    }


def summarise_condition(rows: Sequence[dict[str, Any]], cond_dir: Path) -> dict[str, Any]:
    completed = [r for r in rows if not r.get("runner_exception")]
    integ = base.condition_integrity(rows)
    stats = empty_content_stats(cond_dir)
    return {
        "n": len(rows),
        "completed": len(completed),
        "runner_exceptions": sum(1 for r in rows if r.get("runner_exception")),
        "success": f"{sum(1 for r in completed if r.get('success'))}/{len(rows)}",
        "asr_curve": base.asr_curve(rows),
        "stop_reasons": {
            str(r.get("stop_reason")): sum(
                1 for x in rows if str(x.get("stop_reason")) == str(r.get("stop_reason"))
            )
            for r in rows
        },
        "total_defender_queries": sum(
            int(r.get("scored_defender_queries") or r.get("q_used") or 0)
            for r in completed
        ),
        "integrity": integ,
        **stats,
        "estimated_cost_usd": integ.get("estimated_cost_usd"),
        "prompt_tokens": integ.get("prompt_tokens"),
        "completion_tokens": integ.get("completion_tokens"),
        "latency_related_llm_calls": integ.get("llm_calls"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        raise SystemExit(f"Raw dataset missing: {raw_path}")
    if ABLATION_PROMPT_VERSION != "a3_episodic_reflective_v2_3_public_reference_view":
        raise SystemExit("Ablation pin is not frozen A3 V2.3.")

    # Fail-closed registration of efforts before any construction.
    for cid, think_off, effort in CONDITIONS:
        require_official_reasoning_effort(effort, thinking_disabled=think_off)

    policy = CompiledGovernancePolicy.load(base.GOVERNANCE_PATH)
    missing_proxies = {
        "name_email_alignment",
        "home_phone_configuration",
        "mobile_phone_configuration",
    } - set(policy.available_action_keys)
    if missing_proxies:
        raise SystemExit(f"Governance missing abstract proxies: {sorted(missing_proxies)}")
    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    base.preflight_defence_and_pins(
        policy=policy,
        defender=defender,
        artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
    )
    provider = ReferencePoolProvider.from_config(
        base.build_pool_config(), raw_path=raw_path
    )
    anchors15 = base.resolve_anchors(n=15)
    anchors = anchors15[:N_SANITY]
    if anchors != base.ANCHORS_15[:N_SANITY]:
        raise SystemExit("N=5 anchors drifted from leading same-arena set.")
    base.verify_same_arena(anchors, provider, defender, raw_path)

    first_pool = provider.get_pool(str(anchors[0]), seed=base.REFERENCE_POOL_SEED)
    manifests = []
    for cid, think_off, effort in CONDITIONS:
        attacker = build_a3_v2_3_attacker(
            pool=first_pool,
            recorder=None,
            thinking_disabled=think_off,
            reasoning_effort=effort,
        )
        manifests.append(
            cell_manifest(
                condition_id=cid,
                thinking_disabled=think_off,
                reasoning_effort=effort,
                attacker=attacker,
            )
        )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_OK",
                    "prompt_version": ABLATION_PROMPT_VERSION,
                    "official_reasoning_efforts": sorted(OFFICIAL_REASONING_EFFORTS),
                    "conditions": manifests,
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
        f"dev_a3_v2_3_reasoning_effort_sanity_n{N_SANITY}_m{base.M_MAX}_"
        f"Q{base.Q_MAX}_seed{base.EXPERIMENT_SEED}_{stamp}",
        parent=ROOT / "05_outputs" / "scratch" / "smoke",
        stage="scratch",
    )
    base.write_json(
        run_dir / "run_config.json",
        {
            "status": "development_a3_v2_3_reasoning_effort_sanity_not_findings",
            "created_utc": base.utc_now(),
            "N": N_SANITY,
            "K": base.K,
            "Q": base.Q_MAX,
            "m": base.M_MAX,
            "prompt_version": ABLATION_PROMPT_VERSION,
            "official_reasoning_efforts": sorted(OFFICIAL_REASONING_EFFORTS),
            "conditions": manifests,
            "require_reference_provenance": PINNED_REQUIRE_REFERENCE_PROVENANCE,
            "month7_opened": False,
            "d1_artefact_id": defender.artefact_id,
            "governance_fingerprint": policy.policy_fingerprint,
            "automatic_winner": None,
            "note": (
                "Runner-only reasoning-effort ablation on A3 V2.3. "
                "Not a dissertation finding."
            ),
        },
    )
    base.write_json(
        run_dir / "sanity_anchors.json",
        {"anchor_ids": list(anchors), "N": N_SANITY},
    )

    expected_pools = {
        str(aid): provider.get_pool(str(aid), seed=base.REFERENCE_POOL_SEED).pool_fingerprint
        for aid in anchors
    }

    original_build = base.build_attacker
    original_pin = base.PINNED_A3_PROMPT_VERSION
    original_assert_thinking = base.assert_thinking_cell_config

    def _assert_thinking_cell_config(
        *,
        thinking_disabled: bool,
        reasoning_effort: str | None,
        expect_thinking_disabled: bool,
    ) -> None:
        """Runner-local fail-closed: allow official low/high/max when thinking on."""
        if bool(thinking_disabled) != bool(expect_thinking_disabled):
            raise BenchmarkPinError(
                f"thinking_disabled mismatch: expected={expect_thinking_disabled!r}, "
                f"got={thinking_disabled!r}."
            )
        require_official_reasoning_effort(
            reasoning_effort, thinking_disabled=thinking_disabled
        )

    base.PINNED_A3_PROMPT_VERSION = ABLATION_PROMPT_VERSION
    base.assert_thinking_cell_config = _assert_thinking_cell_config  # type: ignore[assignment]

    reports: dict[str, Any] = {}
    early_stops: list[dict[str, Any]] = []
    try:
        for cid, think_off, effort in CONDITIONS:
            # Bind cell settings into build_attacker for this condition only.
            def _build(
                *,
                _think_off=think_off,
                _effort=effort,
                **kwargs,
            ):
                return build_a3_v2_3_attacker(
                    pool=kwargs["pool"],
                    recorder=kwargs.get("recorder"),
                    thinking_disabled=_think_off,
                    reasoning_effort=_effort,
                )

            base.build_attacker = _build  # type: ignore[assignment]
            rows = base.run_condition(
                run_dir=run_dir,
                condition_id=cid,
                attacker_kind="a3",
                llm_model=MODEL_PRO,
                anchors=anchors,
                policy=policy,
                defender=defender,
                provider=provider,
                raw_path=raw_path,
                thinking_disabled=think_off,
                reasoning_effort=effort,
                expected_pool_fingerprints=expected_pools,
            )
            # Restore experiment-local manifest (run_condition may write generic pin fields).
            probe = build_a3_v2_3_attacker(
                pool=first_pool,
                recorder=None,
                thinking_disabled=think_off,
                reasoning_effort=effort,
            )
            man = cell_manifest(
                condition_id=cid,
                thinking_disabled=think_off,
                reasoning_effort=effort,
                attacker=probe,
            )
            base.write_json(run_dir / cid / "condition_manifest.json", man)
            summary = summarise_condition(rows, run_dir / cid)
            summary["manifest"] = man
            reports[cid] = summary
            base.write_json(run_dir / cid / "condition_ablation_summary.json", summary)

            if (not think_off) and summary.get("raw_total"):
                # Stop criterion: thinking cell produces only empty content.
                if (
                    summary["raw_empty"] == summary["raw_total"]
                    and summary["parse_ok"] == 0
                    and summary["env_step_called"] == 0
                ):
                    early_stops.append(
                        {
                            "condition_id": cid,
                            "reason": "empty_content_collapse",
                            "raw_empty": summary["raw_empty"],
                            "raw_total": summary["raw_total"],
                        }
                    )
                    print(
                        f"EARLY STOP FLAG: {cid} empty-content collapse "
                        f"({summary['raw_empty']}/{summary['raw_total']}).",
                        flush=True,
                    )
    finally:
        base.build_attacker = original_build  # type: ignore[assignment]
        base.PINNED_A3_PROMPT_VERSION = original_pin
        base.assert_thinking_cell_config = original_assert_thinking  # type: ignore[assignment]

    report = {
        "status": "development_a3_v2_3_reasoning_effort_sanity_not_findings",
        "prompt_version": ABLATION_PROMPT_VERSION,
        "N": N_SANITY,
        "anchors": list(anchors),
        "official_reasoning_efforts": sorted(OFFICIAL_REASONING_EFFORTS),
        "conditions": reports,
        "early_stops": early_stops,
        "automatic_winner": None,
        "month7_opened": False,
        "note": (
            "Runner-only ablation. Parser still consumes message.content only. "
            "Not a dissertation final result."
        ),
    }
    base.write_json(run_dir / "SANITY_REPORT.json", report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
