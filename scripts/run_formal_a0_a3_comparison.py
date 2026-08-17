#!/usr/bin/env python3
"""Authoritative formal A0--A3 comparison runner with explicit version pins.

Library defaults remain historical.  This runner fail-closes unless A1 V4.3,
A2 public-reference Gower v2, A3 V2.3, and require_reference_provenance=True
are selected explicitly.  Without ``--launch`` this command performs a
read-only preflight.  Every launch creates a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker  # noqa: E402
from attack_lab.attackers.a1_planner import OneShotLLMPlanner  # noqa: E402
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher  # noqa: E402
from attack_lab.attackers.a3_agent import A3ModelConfig, EpisodicLLMAgent  # noqa: E402
from attack_lab.benchmark_pins import (  # noqa: E402
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
    inspect_constructed_attacker,
    preflight_formal_payload,
)
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.cases import load_starting_case  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.experiment_config import (  # noqa: E402
    FormalExperimentConfig,
    canonical_json_hash,
    sha256_file,
)
from attack_lab.feedback import FeedbackPolicy  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.logger import TrajectoryLogger  # noqa: E402
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator  # noqa: E402
from attack_lab.paths import new_run_directory  # noqa: E402
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402
from baf_data.config import FROZEN_CONFIG  # noqa: E402

DEFAULT_CONFIG = IMPL / "config" / "formal_a0_a3_comparison.json"
DEFAULT_RAW = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")
FORMAL_OUTPUT_ROOT = (
    ROOT / "05_outputs" / "experiments" / "comparisons" / "a0_a3" / "formal"
)


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(IMPL), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def implementation_snapshot() -> dict[str, Any]:
    files = sorted((SRC / "attack_lab").glob("*.py"))
    files += sorted((SRC / "attack_lab" / "attackers").glob("*.py"))
    files.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    entries = []
    for path in files:
        relative = str(path.relative_to(IMPL))
        file_hash = sha256_file(path)
        entries.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")
    status = _git_output("status", "--short")
    return {
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_short": status.splitlines() if status else [],
        "implementation_snapshot_sha256": digest.hexdigest(),
        "files": entries,
    }


def preflight(
    *, config_path: Path, raw_path: Path
) -> tuple[FormalExperimentConfig, list[str], dict[str, Any]]:
    config = FormalExperimentConfig.load(config_path)
    p = config.payload
    errors: list[str] = []

    anchors_path = Path(p["anchors"]["path"])
    if not anchors_path.is_file():
        errors.append(f"missing anchor file: {anchors_path}")
        anchor_ids: list[str] = []
    else:
        actual = sha256_file(anchors_path)
        if actual != p["anchors"]["sha256"]:
            errors.append("anchor-set SHA-256 mismatch")
        anchor_payload = json.loads(anchors_path.read_text(encoding="utf-8"))
        anchor_ids = [str(x) for x in anchor_payload["anchor_ids"]]
        if len(anchor_ids) != int(p["anchors"]["sample_size"]):
            errors.append("anchor sample_size disagrees with anchor file")
        if len(anchor_ids) != len(set(anchor_ids)):
            errors.append("anchor file contains duplicates")

    governance_path = Path(p["governance"]["path"])
    policy = CompiledGovernancePolicy.load(governance_path)
    if policy.policy_version != p["governance"]["version"]:
        errors.append("governance version mismatch")
    if policy.policy_fingerprint != p["governance"]["fingerprint"]:
        errors.append("governance fingerprint mismatch")
    if sha256_file(governance_path) != p["governance"]["file_sha256"]:
        errors.append("governance file SHA-256 mismatch")

    d1_dir = Path(p["d1"]["artifact_dir"])
    for name, expected in p["d1"]["artifact_sha256"].items():
        path = d1_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"D1 artefact mismatch: {name}")

    if not raw_path.is_file():
        errors.append(f"raw dataset missing: {raw_path}")
    elif sha256_file(raw_path) != FROZEN_CONFIG.expected_sha256:
        errors.append("raw dataset SHA-256 mismatch")

    snapshot = implementation_snapshot()
    if snapshot["implementation_snapshot_sha256"] != p["code_snapshot_sha256"]:
        errors.append("implementation snapshot differs from frozen formal config")

    a3_cfg = A3ModelConfig(**p["attackers"]["a3"]["model_config"])
    if a3_cfg.config_hash() != p["attackers"]["a3"]["config_hash"]:
        errors.append("A3 model/prompt config hash mismatch")
    errors.extend(preflight_formal_payload(p))
    return config, anchor_ids, {"errors": errors, "code_revision": snapshot}


def _build_attacker(
    attacker_id: str,
    *,
    config: FormalExperimentConfig,
    seed: int,
    pool: Any,
    budget: AttackBudget,
):
    entry = config.payload["attackers"][attacker_id]
    if attacker_id == "a0":
        attacker = ConstrainedRandomAttacker(
            seed=seed, reference_pool=pool, m_max=budget.m_max, attacker_id="a0"
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id="a0",
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
        )
        return attacker
    if attacker_id == "a1":
        model_config = dict(entry["model_config"])
        model_config["prompt_version"] = PINNED_A1_PROMPT_VERSION
        attacker = OneShotLLMPlanner(
            experiment_seed=seed,
            reference_pool=pool,
            budget=budget,
            attacker_id="a1",
            **model_config,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id="a1",
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
            llm_model=str(model_config.get("model") or attacker.model),
        )
        return attacker
    if attacker_id == "a2":
        attacker = SurrogateGuidedSearcher(
            budget=budget,
            reference_pool=pool,
            experiment_seed=seed,
            attacker_id="a2",
            gower_policy=PINNED_A2_GOWER_POLICY,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id="a2",
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
        )
        return attacker
    if attacker_id == "a3":
        model_config = dict(entry["model_config"])
        model_config["prompt_version"] = PINNED_A3_PROMPT_VERSION
        attacker = EpisodicLLMAgent(
            experiment_seed=seed,
            reference_pool=pool,
            budget=budget,
            attacker_id="a3",
            **model_config,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id="a3",
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
            llm_model=str(model_config.get("model") or attacker.model),
        )
        return attacker
    raise ValueError(f"Unknown attacker: {attacker_id}")


def _asr_curve(rows: Sequence[Mapping[str, Any]], q_max: int) -> dict[str, float]:
    n = len(rows)
    return {
        f"ASR@{q}": (
            sum(
                1
                for row in rows
                if row["attempts_to_success"] is not None
                and int(row["attempts_to_success"]) <= q
            )
            / n
            if n
            else 0.0
        )
        for q in range(1, q_max + 1)
    }


def run_formal(
    *,
    config: FormalExperimentConfig,
    anchor_ids: Sequence[str],
    raw_path: Path,
    code_revision: Mapping[str, Any],
) -> Path:
    p = config.payload
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"formal_a0_a3_n{len(anchor_ids)}_q{config.q_max}_m{config.m_max}_{stamp}",
        parent=FORMAL_OUTPUT_ROOT,
        stage="experiments",
    )
    (run_dir / "FROZEN_FORMAL_CONFIG.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "FORMAL_A0_A3_IN_PROGRESS",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "formal_config_hash": config.config_hash,
                "code_revision": to_jsonable(dict(code_revision)),
                "month7_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    policy = CompiledGovernancePolicy.load(Path(p["governance"]["path"]))
    defender = FrozenXGBoostDefender.from_artefact_dir(Path(p["d1"]["artifact_dir"]))
    pool_base = ReferencePoolConfig.load(Path(p["reference_pool"]["config_path"]))
    pool_cfg = ReferencePoolConfig(
        K=config.k,
        seed=int(p["reference_pool"]["seed"]),
        context_fields=pool_base.context_fields,
        action_fields=pool_base.action_fields,
        read_only_context_fields=pool_base.read_only_context_fields,
        excluded_fields=pool_base.excluded_fields,
        label="formal_a0_a3_reference_pool",
        source_path=pool_base.source_path,
    )
    provider = ReferencePoolProvider.from_config(pool_cfg, raw_path=raw_path)
    budget = AttackBudget(q_max=config.q_max, m_max=config.m_max)
    budget_spec = budget.to_budget_spec(label="formal_a0_a3_frozen_budget")
    rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        for attacker_id in config.attacker_ids:
            for anchor_id in anchor_ids:
                starting = load_starting_case(
                    anchor_id,
                    raw_path=raw_path,
                    defender=defender,
                    artefact_dir=Path(p["d1"]["artifact_dir"]),
                )
                pool = provider.get_pool(
                    anchor_id, seed=int(p["reference_pool"]["seed"])
                )
                episode_dir = run_dir / f"seed_{seed}" / attacker_id / f"anchor_{anchor_id}"
                episode_dir.mkdir(parents=True, exist_ok=False)
                logger = TrajectoryLogger(
                    run_dir=episode_dir,
                    run_id=f"{attacker_id}_{anchor_id}_seed{seed}",
                )
                logger.write_manifest(
                    config.episode_manifest(
                        attacker_id=attacker_id,
                        anchor_id=anchor_id,
                        seed=seed,
                        reference_pool_fingerprint=pool.pool_fingerprint,
                        code_revision=code_revision,
                    )
                )
                attacker = _build_attacker(
                    attacker_id,
                    config=config,
                    seed=seed,
                    pool=pool,
                    budget=budget,
                )
                match = MatchOrchestrator().run_episode(
                    attacker,
                    MatchConfig(
                        attacker_id=attacker_id,
                        anchor=starting,
                        policy=policy,
                        budget=budget_spec,
                        feedback_policy=FeedbackPolicy(mode="label_only"),
                        defender=defender,
                        seed=seed,
                        logger=logger,
                        reference_pool=pool,
                        require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
                    ),
                )
                (episode_dir / "match_result.json").write_text(
                    json.dumps(match.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                row = {
                    "attacker": attacker_id,
                    "seed": seed,
                    "anchor_id": anchor_id,
                    "success": match.success,
                    "attempts_to_success": match.attempts_to_success,
                    "q_used": match.q_used,
                    "invalid_submissions": match.invalid_submissions,
                    "stop_reason": match.stop_reason,
                }
                rows.append(row)
                with (run_dir / "episode_index.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["attacker"]), int(row["seed"]))].append(row)
    summary = {
        "status": "FORMAL_A0_A3_COMPLETE",
        "formal_config_hash": config.config_hash,
        "n_episodes": len(rows),
        "cells": {
            f"{attacker}_seed{seed}": {
                "n": len(cell),
                "asr_curve": _asr_curve(cell, config.q_max),
                "invalid_submissions": sum(int(r["invalid_submissions"]) for r in cell),
            }
            for (attacker, seed), cell in sorted(grouped.items())
        },
        "aggregation_plan": p["aggregation"],
    }
    (run_dir / "formal_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the frozen formal experiment; omit for read-only preflight.",
    )
    args = parser.parse_args(argv)
    config, anchor_ids, report = preflight(
        config_path=args.config, raw_path=args.raw
    )
    if report["errors"]:
        raise SystemExit("PRE-FLIGHT FAILED:\n- " + "\n- ".join(report["errors"]))
    print(
        json.dumps(
            {
                "status": "PRE-FLIGHT PASSED",
                "formal_config_hash": config.config_hash,
                "sample_size": len(anchor_ids),
                "seeds": list(config.seeds),
                "attackers": list(config.attacker_ids),
                "expected_episodes": len(anchor_ids)
                * len(config.seeds)
                * len(config.attacker_ids),
                "code_snapshot_sha256": report["code_revision"][
                    "implementation_snapshot_sha256"
                ],
                "month7_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.launch:
        print("Preflight only. Add --launch to start the formal comparison.")
        return 0
    output = run_formal(
        config=config,
        anchor_ids=anchor_ids,
        raw_path=args.raw,
        code_revision=report["code_revision"],
    )
    print(f"Formal comparison complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
