#!/usr/bin/env python3
"""Month-6 D1+D2-S evaluation for the frozen Pro / ThinkOff configuration.

Reuses complete, pin-matched attack artefacts.  Does not refit D2-S, does not
retune from attack outcomes, does not open Month 7, and does not call an LLM
unless the required artefacts are missing or fail the pin check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SRC = IMPL / "src"
SCRIPTS = IMPL / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

import run_dev_model_selection as base  # noqa: E402
import run_month6_frozen_a0_a3 as frozen  # noqa: E402
from attack_lab.benchmark_pins import (  # noqa: E402
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
    PINNED_GOVERNANCE_FINGERPRINT,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
)
from attack_lab.cases import DEFAULT_RAW_PATH  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, SCRATCH_CALIBRATION_ROOT, new_run_directory  # noqa: E402
from attack_lab.reference_pool import ReferencePoolProvider  # noqa: E402
from d2.calibrate import (  # noqa: E402
    extract_d1_pass_features,
    month6_legitimate_d1_pass,
    score_submissions,
    thresholds_for_budgets,
)
from d2.contract import AGGREGATION_METHOD_ID, RELATIONSHIP_IDS, SCORE_CONTRACT_ID  # noqa: E402
from d2.plotting import plot_security_curve  # noqa: E402
from d2.scoring import D2SScorer  # noqa: E402

COMPLETE_BENCHMARK = (
    ROOT
    / "05_outputs"
    / "scratch"
    / "smoke"
    / "dev_model_selection_benchmark_m2_Q5_seed1_20260812T232252Z"
)
INCOMPLETE_CONFIRMATORY = (
    ROOT
    / "05_outputs"
    / "experiments"
    / "comparisons"
    / "a0_a3"
    / "month6_frozen"
    / "month6_frozen_a0_a3_n50_m2_Q5_seed1_20260813T215711Z"
)
FROZEN_D2S_REFERENCE = (
    ROOT
    / "05_outputs"
    / "scratch"
    / "calibration"
    / "d2s_reference_d2s-v1.0.0-pairwise8-20260816_20260816T005649Z"
    / "d2s_reference.json"
)
CONDITIONS = {
    "A0": "A0",
    "A1": "A1-Pro",
    "A2": "A2",
    "A3": "A3-Pro",
}


def _fail(message: str) -> None:
    raise SystemExit(f"FAIL CLOSED: {message}")


def anchor_set_fingerprint(anchor_ids: list[str]) -> str:
    payload = json.dumps(list(anchor_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_rarity_observed_state(scorer: D2SScorer) -> dict[str, Any]:
    table = scorer.tables["C01"]
    x = max(table.n_x, key=table.n_x.get)
    p_present = table.probability(x, "1")
    p_absent = table.probability(x, "0")
    r_present = table.raw_rarity(x, "1")
    r_absent = table.raw_rarity(x, "0")
    if abs(r_present - (1.0 - p_present)) > 1e-12:
        _fail("C01 present rarity is not 1-P(observed present | context).")
    if abs(r_absent - (1.0 - p_absent)) > 1e-12:
        _fail("C01 absent rarity is not 1-P(observed absent | context).")
    if abs(r_present - r_absent) < 1e-12:
        _fail("C01 present and absent rarities are identical; presence appears hardcoded.")
    return {
        "relationship": "C01",
        "conditioner": x,
        "p_observed_present": p_present,
        "p_observed_absent": p_absent,
        "rarity_present": r_present,
        "rarity_absent": r_absent,
        "formula": "1 - P(observed target state | observed context)",
    }


def verify_episode_pins(path: Path, *, condition_id: str, expected_anchors: list[str]) -> dict[str, Any]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    steps = episode.get("steps") or []
    defence_ids = {
        (step.get("internal_defence") or {}).get("artefact_id")
        for step in steps
        if step.get("internal_defence")
    }
    defence_ids.discard(None)
    if defence_ids and defence_ids != {PINNED_D1_ARTEFACT_ID}:
        _fail(f"{path}: D1 artefact {defence_ids} != {PINNED_D1_ARTEFACT_ID}")
    meta = {}
    for step in steps:
        meta = step.get("research_meta") or meta
    thinking = None
    model = None
    prompt = None
    gower = None
    # Prefer query model_config when present.
    local_cfg = path.parent / "model_config.json"
    query_cfgs = sorted(path.parent.glob("query_*/model_config.json"))
    cfg_path = local_cfg if local_cfg.is_file() else (query_cfgs[0] if query_cfgs else None)
    if cfg_path is not None:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        thinking = cfg.get("thinking_disabled")
        model = cfg.get("model")
        prompt = cfg.get("prompt_version", prompt)
    index_hint = path.parent / "episode_diag.json"
    if index_hint.is_file():
        diag = json.loads(index_hint.read_text(encoding="utf-8"))
        thinking = diag.get("thinking_disabled", thinking)
        model = diag.get("llm_model", model)
        prompt = diag.get("prompt_version", prompt)
        gower = diag.get("gower_policy", gower)
    return {
        "anchor_id": str(episode.get("case_id")),
        "success": bool(episode.get("success")),
        "thinking_disabled": thinking,
        "model": model,
        "prompt_version": prompt,
        "gower_policy": gower,
        "n_d1_pass": len(extract_d1_pass_features(episode)),
    }


def verify_complete_pro_thinkoff_artefact(
    benchmark_root: Path,
    expected_anchors: list[str],
) -> dict[str, Any]:
    if "month7" in str(benchmark_root).lower() or "month_7" in str(benchmark_root).lower():
        _fail(f"Month-7 path: {benchmark_root}")
    run_config = json.loads((benchmark_root / "run_config.json").read_text(encoding="utf-8"))
    stored_anchors = [
        str(x) for x in json.loads((benchmark_root / "benchmark_anchors.json").read_text())["anchor_ids"]
    ]
    if stored_anchors != expected_anchors:
        _fail("Benchmark anchor list does not match the frozen 50-anchor ranking.")
    if run_config.get("month7_opened") is not False:
        _fail("Benchmark month7_opened is not false.")
    if str(run_config.get("d1_artefact_id")) != PINNED_D1_ARTEFACT_ID:
        _fail("Benchmark D1 artefact pin mismatch.")
    if str(run_config.get("governance_fingerprint")) != PINNED_GOVERNANCE_FINGERPRINT:
        _fail("Benchmark governance pin mismatch.")
    if int(run_config.get("Q")) != 5 or int(run_config.get("m")) != 2 or int(run_config.get("K")) != 10:
        _fail("Benchmark Q/m/K pin mismatch.")
    if run_config.get("require_reference_provenance") is not True:
        _fail("require_reference_provenance is not true.")
    pins = run_config.get("pins") or {}
    if pins.get("a1_prompt_version") != PINNED_A1_PROMPT_VERSION:
        _fail("A1 prompt pin mismatch.")
    if pins.get("a2_gower_policy") != PINNED_A2_GOWER_POLICY:
        _fail("A2 Gower pin mismatch.")
    if pins.get("a3_prompt_version") != PINNED_A3_PROMPT_VERSION:
        _fail("A3 prompt pin mismatch.")

    summaries: dict[str, Any] = {}
    for label, directory in CONDITIONS.items():
        root = benchmark_root / "benchmark" / directory
        paths = sorted(root.glob("anchor_*/seed_*/episode_result.json"))
        if len(paths) != 50:
            _fail(f"{directory} has {len(paths)} episodes; expected 50.")
        observed_anchors = [p.parts[-3].removeprefix("anchor_") for p in paths]
        if sorted(observed_anchors) != sorted(expected_anchors):
            _fail(f"{directory} anchor set mismatch.")
        thinking_flags: set[Any] = set()
        models: set[Any] = set()
        n_success = 0
        n_d1_pass = 0
        for path in paths:
            info = verify_episode_pins(path, condition_id=label, expected_anchors=expected_anchors)
            n_success += int(info["success"])
            n_d1_pass += int(info["n_d1_pass"])
            if label in {"A1", "A3"}:
                if info["thinking_disabled"] is not True:
                    _fail(f"{path}: thinking_disabled is {info['thinking_disabled']!r}")
                if info["model"] != MODEL_PRO:
                    _fail(f"{path}: model {info['model']!r} is not {MODEL_PRO!r}")
                thinking_flags.add(info["thinking_disabled"])
                models.add(info["model"])
        summary = json.loads((root / "condition_summary.json").read_text(encoding="utf-8"))
        if label == "A1":
            if summary.get("prompt_version") != PINNED_A1_PROMPT_VERSION:
                _fail("A1-Pro prompt_version mismatch.")
            if summary.get("llm_model") != MODEL_PRO:
                _fail("A1-Pro model mismatch.")
        if label == "A3":
            if summary.get("prompt_version") != PINNED_A3_PROMPT_VERSION:
                _fail("A3-Pro prompt_version mismatch.")
            if summary.get("llm_model") != MODEL_PRO:
                _fail("A3-Pro model mismatch.")
        if label == "A2" and summary.get("gower_policy") != PINNED_A2_GOWER_POLICY:
            _fail("A2 Gower mismatch.")
        summaries[label] = {
            "condition_dir": directory,
            "n_episodes": 50,
            "d1_successes_from_episodes": n_success,
            "d1_pass_submissions": n_d1_pass,
            "summary_success": summary.get("success"),
            "thinking_disabled_values": sorted(thinking_flags, key=str),
            "models": sorted(models, key=str),
            "month7_opened": summary.get("month7_opened"),
        }
        if n_success != n_d1_pass:
            _fail(f"{directory}: D1 success count {n_success} != D1-PASS submissions {n_d1_pass}.")
    return {
        "benchmark_root": str(benchmark_root),
        "anchor_set_fingerprint": anchor_set_fingerprint(expected_anchors),
        "n_anchors": len(expected_anchors),
        "conditions": summaries,
        "require_reference_provenance": PINNED_REQUIRE_REFERENCE_PROVENANCE,
    }


def load_condition_episodes(benchmark_root: Path, directory: str) -> list[dict[str, Any]]:
    root = benchmark_root / "benchmark" / directory
    episodes = []
    for path in sorted(root.glob("anchor_*/seed_*/episode_result.json")):
        episodes.append(json.loads(path.read_text(encoding="utf-8")))
    return episodes


def evaluate_condition(
    *,
    label: str,
    episodes: list[dict[str, Any]],
    scorer: D2SScorer,
    thresholds: pd.DataFrame,
) -> dict[str, Any]:
    n_anchors = len(episodes)
    d1_pass_features: list[dict[str, Any]] = []
    d1_pass_anchor_ids: list[str] = []
    for episode in episodes:
        found = extract_d1_pass_features(episode)
        if found:
            d1_pass_features.append(found[0])
            d1_pass_anchor_ids.append(str(episode.get("case_id")))
    d1_pass_n = len(d1_pass_features)
    d1_asr = d1_pass_n / n_anchors if n_anchors else float("nan")
    scores = score_submissions(scorer, d1_pass_features)
    rows = []
    for record in thresholds.to_dict(orient="records"):
        threshold = float(record["threshold"])
        n_review = int((scores >= threshold).sum()) if d1_pass_n else 0
        n_clear = d1_pass_n - n_review
        interception = n_review / d1_pass_n if d1_pass_n else float("nan")
        conditional_clear = n_clear / d1_pass_n if d1_pass_n else float("nan")
        end_to_end_rate = n_clear / n_anchors if n_anchors else float("nan")
        rows.append(
            {
                "budget": float(record["budget"]),
                "threshold": threshold,
                "benign_review_rate": float(record["benign_review_rate"]),
                "d1_asr": d1_asr,
                "d1_pass_count": d1_pass_n,
                "d2_interception_among_d1_pass": interception,
                "conditional_d2_clear_rate": conditional_clear,
                "end_to_end_bypass_count": n_clear,
                "end_to_end_bypass_rate": end_to_end_rate,
            }
        )
    return {
        "label": label,
        "n_anchors": n_anchors,
        "d1_asr": d1_asr,
        "d1_pass_count": d1_pass_n,
        "d1_pass_anchor_ids": d1_pass_anchor_ids,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--reference", type=Path, default=FROZEN_D2S_REFERENCE)
    parser.add_argument("--benchmark", type=Path, default=COMPLETE_BENCHMARK)
    args = parser.parse_args()

    if not args.reference.is_file():
        _fail(f"Frozen D2-S reference missing: {args.reference}")
    scorer = D2SScorer.load(args.reference)
    if scorer.month7_opened:
        _fail("D2-S artefact reports Month 7 opened.")
    if scorer.fingerprint != (
        "cfd5330f096dabb1749be447ee4da4d5f498d2599f4f22c24a0b706e570bfd94"
    ):
        _fail("D2-S fingerprint drifted from the frozen V1 artefact.")
    if list(scorer.tables) != list(RELATIONSHIP_IDS):
        _fail("D2-S relationship set drifted.")
    if AGGREGATION_METHOD_ID != "equal_mean_v1":
        _fail("D2-S aggregation method drifted.")
    rarity_check = verify_rarity_observed_state(scorer)

    if "month7" in str(args.raw).lower():
        _fail("Refusing a Month-7 raw path.")
    policy = CompiledGovernancePolicy.load(base.GOVERNANCE_PATH)
    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    base.preflight_defence_and_pins(
        policy=policy,
        defender=defender,
        artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
    )
    provider = ReferencePoolProvider.from_config(
        base.build_pool_config(), raw_path=args.raw
    )
    expected_anchors = base.resolve_anchors(n=50)
    if len(expected_anchors) != 50:
        _fail(f"Expected 50 frozen anchors, got {len(expected_anchors)}.")
    expected_fp = anchor_set_fingerprint(expected_anchors)
    base.verify_same_arena(expected_anchors, provider, defender, args.raw)
    preflight = frozen.hard_preflight(
        raw_path=args.raw,
        policy=policy,
        defender=defender,
        provider=provider,
        anchors=expected_anchors,
    )

    incomplete_a3 = len(
        list((INCOMPLETE_CONFIRMATORY / "A3-Pro").glob("anchor_*/seed_*/episode_result.json"))
    ) if INCOMPLETE_CONFIRMATORY.is_dir() else 0
    artefact = verify_complete_pro_thinkoff_artefact(args.benchmark, expected_anchors)

    legit = month6_legitimate_d1_pass(args.raw, verify_hash=True)
    legit_scores = scorer.score_many(legit)["d2_score"].to_numpy()
    budget_table = thresholds_for_budgets(legit_scores)

    evaluations = {}
    for label, directory in CONDITIONS.items():
        episodes = load_condition_episodes(args.benchmark, directory)
        evaluations[label] = evaluate_condition(
            label=label,
            episodes=episodes,
            scorer=scorer,
            thresholds=budget_table,
        )

    curve_rows = []
    for i, record in enumerate(budget_table.to_dict(orient="records")):
        row = {
            "threshold": float(record["threshold"]),
            "benign_review_rate": float(record["benign_review_rate"]),
        }
        for label in ("A0", "A1", "A2", "A3"):
            item = evaluations[label]["rows"][i]
            row[f"{label}_interception"] = item["d2_interception_among_d1_pass"]
            row[f"{label}_conditional_clear"] = item["conditional_d2_clear_rate"]
            row[f"{label}_end_to_end_bypass"] = item["end_to_end_bypass_rate"]
        curve_rows.append(row)
    curve = pd.DataFrame(curve_rows)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"d2s_month6_pro_thinkoff_{SCORE_CONTRACT_ID}_{stamp}",
        parent=SCRATCH_CALIBRATION_ROOT,
        stage="scratch",
    )
    plot_path = plot_security_curve(
        curve,
        run_dir / "d2s_month6_pro_thinkoff_interception_curve.png",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e2e_path = run_dir / "d2s_month6_pro_thinkoff_end_to_end_bypass_curve.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for column, label in (
        ("A0_end_to_end_bypass", "A0"),
        ("A1_end_to_end_bypass", "A1"),
        ("A2_end_to_end_bypass", "A2"),
        ("A3_end_to_end_bypass", "A3"),
    ):
        ax.plot(curve["benign_review_rate"], curve[column], marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Benign review rate")
    ax.set_ylabel("End-to-end D1+D2 bypass rate (CLEAR / all anchors)")
    ax.set_title("D2-S Month-6 Pro/ThinkOff end-to-end bypass (development only)")
    ax.set_xlim(0.0, 0.22)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(e2e_path, dpi=160)
    plt.close(fig)

    budget_table.to_csv(run_dir / "benign_review_thresholds.csv", index=False)
    curve.to_csv(run_dir / "pro_thinkoff_security_curve.csv", index=False)
    for label, ev in evaluations.items():
        pd.DataFrame(ev["rows"]).to_csv(run_dir / f"{label}_threshold_table.csv", index=False)

    summary = {
        "report_id": "FROZEN_MONTH6_D1_D2S_PRO_THINKOFF_REPORT",
        "score_contract_id": SCORE_CONTRACT_ID,
        "d2s_fingerprint": scorer.fingerprint,
        "month7_opened": False,
        "attackers_rerun": False,
        "attacker_reuse_reason": (
            "Complete same-arena N=50 Pro/ThinkOff artefacts already exist and "
            "passed fail-closed pin checks. The later confirmatory directory is "
            f"incomplete (A3-Pro {incomplete_a3}/50) and was not mixed in."
        ),
        "incomplete_confirmatory_run": str(INCOMPLETE_CONFIRMATORY),
        "incomplete_confirmatory_a3_episodes": incomplete_a3,
        "benchmark_root": str(args.benchmark),
        "anchor_set_fingerprint": expected_fp,
        "anchors": expected_anchors,
        "rarity_check": rarity_check,
        "live_preflight": {
            "status": preflight.get("status"),
            "d1_artefact_id": preflight.get("d1_artefact_id"),
            "governance_fingerprint": preflight.get("governance_fingerprint"),
            "month7_opened": preflight.get("month7_opened"),
            "probes": preflight.get("probes"),
        },
        "artefact_verification": artefact,
        "month6_legitimate_d1_pass_n": int(len(legit)),
        "benign_review_table": budget_table.to_dict(orient="records"),
        "evaluations": {
            label: {
                "n_anchors": ev["n_anchors"],
                "d1_asr": ev["d1_asr"],
                "d1_pass_count": ev["d1_pass_count"],
                "rows": ev["rows"],
            }
            for label, ev in evaluations.items()
        },
        "curve_paths": {
            "conditional_interception": str(plot_path),
            "end_to_end_bypass": str(e2e_path),
        },
        "metric_definitions": {
            "d1_asr": "D1 PASS episodes / all original attack anchors",
            "d2_interception_among_d1_pass": "REVIEW / D1_PASS",
            "conditional_d2_clear_rate": "CLEAR / D1_PASS",
            "end_to_end_bypass_rate": "D1 PASS AND D2 CLEAR / all original attack anchors",
        },
        "note": "No CLEAR/REVIEW operating point was selected.",
    }
    (run_dir / "PRO_THINKOFF_EVAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
