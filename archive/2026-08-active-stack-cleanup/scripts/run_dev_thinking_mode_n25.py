#!/usr/bin/env python3
"""Pinned N=25 Pro thinking-mode development benchmark (NOT findings).

Four cells on the same Month-6 same-arena anchors (K=10, Q=5, m=2):
  A1-Pro-ThinkOff / A1-Pro-ThinkOn
  A3-Pro-ThinkOff / A3-Pro-ThinkOn

Pins (fail closed before any API call):
  A1 ``a1_oneshot_v4_4_adversarial_objective``
  A3 ``a3_episodic_reflective_v2_4_adversarial_objective``
  model ``deepseek-v4-pro``
  thinking ON requires ``reasoning_effort=max``
  provenance / D1 / governance / reference-pool fingerprints

Does not redesign attackers, prompts, schemas, governance, K, Q, m, D1,
reference pools, or Month-7 rules. Does not overwrite historical artefacts.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    PINNED_A3_PROMPT_VERSION,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
    REASONING_EFFORT_MAX,
    BenchmarkPinError,
    assert_thinking_cell_config,
    pinned_attacker_summary,
)
from attack_lab.cases import DEFAULT_RAW_PATH  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory  # noqa: E402
from attack_lab.reference_pool import ReferencePoolProvider  # noqa: E402
from attack_lab.types import to_jsonable  # noqa: E402

THINKING_CONDITIONS: tuple[tuple[str, str, bool, str | None], ...] = (
    ("A1-Pro-ThinkOff", "a1", True, None),
    ("A1-Pro-ThinkOn", "a1", False, REASONING_EFFORT_MAX),
    ("A3-Pro-ThinkOff", "a3", True, None),
    ("A3-Pro-ThinkOn", "a3", False, REASONING_EFFORT_MAX),
)


def expected_pins() -> dict[str, str]:
    return {
        "a1_prompt_version": "a1_oneshot_v4_4_adversarial_objective",
        "a3_prompt_version": "a3_episodic_reflective_v2_4_adversarial_objective",
        "model": MODEL_PRO,
        "reasoning_effort_when_thinking_enabled": REASONING_EFFORT_MAX,
    }


def assert_authoritative_prompt_pins() -> None:
    if PINNED_A1_PROMPT_VERSION != expected_pins()["a1_prompt_version"]:
        raise BenchmarkPinError(
            f"A1 pin must be V4.4; got {PINNED_A1_PROMPT_VERSION!r}"
        )
    if PINNED_A3_PROMPT_VERSION != expected_pins()["a3_prompt_version"]:
        raise BenchmarkPinError(
            f"A3 pin must be V2.4; got {PINNED_A3_PROMPT_VERSION!r}"
        )


def load_expected_pool_fingerprints(anchors: Sequence[str]) -> dict[str, str]:
    prior_arena = json.loads((base.PRIOR_ARENA_SMOKE / "arena_precompute.json").read_text())
    out: dict[str, str] = {}
    for aid in anchors:
        if aid not in prior_arena:
            raise BenchmarkPinError(
                f"Missing prior-arena pool fingerprint for anchor {aid}"
            )
        out[str(aid)] = str(prior_arena[aid]["pool_fingerprint"])
    return out


def preflight_thinking_cells() -> list[dict[str, Any]]:
    """Validate cell registrations without constructing LLM clients."""
    assert_authoritative_prompt_pins()
    manifests: list[dict[str, Any]] = []
    for cid, kind, think_off, effort in THINKING_CONDITIONS:
        assert_thinking_cell_config(
            thinking_disabled=think_off,
            reasoning_effort=effort,
            expect_thinking_disabled=think_off,
        )
        prompt_version = (
            PINNED_A1_PROMPT_VERSION if kind == "a1" else PINNED_A3_PROMPT_VERSION
        )
        manifests.append(
            {
                "condition_id": cid,
                "attacker_kind": kind,
                "attacker_version": prompt_version,
                "prompt_version": prompt_version,
                "prompt_hash": base.prompt_family_hash(prompt_version),
                "model": MODEL_PRO,
                "thinking_disabled": think_off,
                "thinking_enabled": not think_off,
                "reasoning_effort": effort,
            }
        )
    return manifests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to frozen BAF Base.csv (default: DEFAULT_RAW_PATH).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate pins/cells/defence identity and exit (no DeepSeek calls).",
    )
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        raise SystemExit(f"Raw dataset missing: {raw_path}")

    cell_manifests = preflight_thinking_cells()
    policy = CompiledGovernancePolicy.load(base.GOVERNANCE_PATH)
    missing_proxies = {
        "name_email_alignment",
        "home_phone_configuration",
        "mobile_phone_configuration",
    } - set(policy.available_action_keys)
    if missing_proxies:
        raise SystemExit(
            f"Governance available actions missing abstract proxies: "
            f"{sorted(missing_proxies)}"
        )
    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    base.preflight_defence_and_pins(
        policy=policy,
        defender=defender,
        artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
    )
    provider = ReferencePoolProvider.from_config(
        base.build_pool_config(), raw_path=raw_path
    )
    anchors25 = base.resolve_anchors(n=25)
    base.verify_same_arena(anchors25, provider, defender, raw_path)
    expected_pools = load_expected_pool_fingerprints(anchors25)

    # Construct each cell attacker once before any API call.
    first_pool = provider.get_pool(str(anchors25[0]), seed=base.REFERENCE_POOL_SEED)
    constructed_manifests: list[dict[str, Any]] = []
    for cid, kind, think_off, effort in THINKING_CONDITIONS:
        attacker = base.build_attacker(
            condition_id=cid,
            attacker_kind=kind,
            llm_model=MODEL_PRO,
            pool=first_pool,
            recorder=None,
            thinking_disabled=think_off,
            reasoning_effort=effort,
        )
        constructed_manifests.append(
            base.attacker_condition_manifest(
                condition_id=cid,
                attacker_kind=kind,
                llm_model=MODEL_PRO,
                attacker=attacker,
                thinking_disabled=think_off,
                reasoning_effort=effort,
            )
        )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_OK",
                    "pins": pinned_attacker_summary(),
                    "expected_pins": expected_pins(),
                    "cell_manifests": cell_manifests,
                    "constructed_manifests": constructed_manifests,
                    "n_anchors": len(anchors25),
                    "month7_opened": False,
                    "deepseek_calls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"dev_thinking_mode_n25_m{base.M_MAX}_Q{base.Q_MAX}_"
        f"seed{base.EXPERIMENT_SEED}_{stamp}",
        parent=ROOT / "05_outputs" / "scratch" / "smoke",
        stage="scratch",
    )
    base.write_json(
        run_dir / "run_config.json",
        {
            "status": "development_thinking_mode_selection_not_findings",
            "created_utc": base.utc_now(),
            "N": 25,
            "K": base.K,
            "Q": base.Q_MAX,
            "m": base.M_MAX,
            "experiment_seed": base.EXPERIMENT_SEED,
            "reference_pool_seed": base.REFERENCE_POOL_SEED,
            "model": MODEL_PRO,
            "pins": pinned_attacker_summary(),
            "expected_pins": expected_pins(),
            "require_reference_provenance": PINNED_REQUIRE_REFERENCE_PROVENANCE,
            "conditions": constructed_manifests,
            "month7_opened": False,
            "d1_artefact_dir": str(DEFAULT_C1_ARTEFACT_DIR),
            "d1_artefact_id": defender.artefact_id,
            "raw_path": str(raw_path),
            "governance_fingerprint": policy.policy_fingerprint,
            "note": (
                "Development thinking-mode evidence only. "
                "Not a Month-7 result and not a dissertation final finding."
            ),
        },
    )
    base.write_json(
        run_dir / "benchmark_anchors.json",
        {"anchor_ids": anchors25, "N": 25},
    )

    by_condition: dict[str, list[dict[str, Any]]] = {}
    for cid, kind, think_off, effort in THINKING_CONDITIONS:
        by_condition[cid] = base.run_condition(
            run_dir=run_dir,
            condition_id=cid,
            attacker_kind=kind,
            llm_model=MODEL_PRO,
            anchors=anchors25,
            policy=policy,
            defender=defender,
            provider=provider,
            raw_path=raw_path,
            thinking_disabled=think_off,
            reasoning_effort=effort,
            expected_pool_fingerprints=expected_pools,
        )

    report: dict[str, Any] = {
        "status": "development_thinking_mode_selection_not_findings",
        "automatic_winner": None,
        "pins": pinned_attacker_summary(),
        "conditions": {
            cid: {
                "success": f"{sum(1 for r in rows if r.get('success'))}/{len(rows)}",
                "asr_curve": base.asr_curve(rows),
                "integrity": base.condition_integrity(rows),
                "manifest": (
                    json.loads(
                        (run_dir / cid / "condition_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    if (run_dir / cid / "condition_manifest.json").is_file()
                    else None
                ),
            }
            for cid, rows in by_condition.items()
        },
        "paired_anchors": base.paired_anchor_table(by_condition),
        "month7_opened": False,
        "note": "Researcher review only; no automatic thinking-mode selection.",
    }
    base.write_json(run_dir / "THINKING_MODE_REPORT.json", report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
