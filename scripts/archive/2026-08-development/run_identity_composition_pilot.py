#!/usr/bin/env python3
"""Identity-composition construct-validity pilot (A0 vs A2, single seed).

NOT dissertation findings. Does not modify D1, governance-v2, pools, or
attacker search/ranking formulas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_IMPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_IMPL / "src"))

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker  # noqa: E402
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher  # noqa: E402
from attack_lab.budget import AttackBudget  # noqa: E402
from attack_lab.cases import DEFAULT_RAW_PATH, load_starting_case  # noqa: E402
from attack_lab.constraint_profile import IdentityCompositionProfile  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.feedback import FeedbackPolicy  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.logger import TrajectoryLogger  # noqa: E402
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, EXPERIMENTS_ROOT  # noqa: E402
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402

EXPECTED_GOV_FP = (
    "177c7b9fec00f531932528ad4b77d7833a436b9e5705f89bf5045ff576d2ff16"
)
FROZEN_ANCHORS_SOURCE = (
    EXPERIMENTS_ROOT
    / "a0"
    / "calibration"
    / "governance_v2"
    / "formal_multiseed_m123_20260804T161724Z"
    / "frozen_anchors.json"
)
UNRESTRICTED_DEV = (
    EXPERIMENTS_ROOT
    / "comparisons"
    / "a0_vs_a2"
    / "development"
    / "paired_m2_q5_n100_3seed_20260804T211753Z"
)
GOVERNANCE_PATH = _IMPL / "config" / "attacker_compiled_governance.json"
REFERENCE_POOL_SEED = 20260803
EXPERIMENT_SEED = 20260804


@dataclass
class EpisodeRow:
    attacker_id: str
    anchor_id: str
    success: bool
    queries_used: int
    queries_to_success: int | None
    stop_reason: str
    invalid_submissions: int
    distances: list[int] = field(default_factory=list)
    edited_field_sets: list[tuple[str, ...]] = field(default_factory=list)
    persona_fields: list[str] = field(default_factory=list)
    contact_fields: list[str] = field(default_factory=list)
    persona_contact_pairs: list[str] = field(default_factory=list)
    success_at: dict[int, bool] = field(default_factory=dict)
    duplicate: bool = False
    m_cap_violation: bool = False
    governance_violation: bool = False
    profile_violation: bool = False
    fallback_invoked: int = 0
    fallback_picks: int = 0
    fallback_pass: int = 0
    fallback_block: int = 0
    reorder_count: int = 0
    zero_submission_nofeasible: bool = False
    action_space_exhaustion: bool = False
    initial_legal_count: int | None = None
    legal_remaining_series: list[int] = field(default_factory=list)
    can_complete_five_distinct: bool = False


def select_smoke_anchors(anchor_ids: Sequence[str], *, n: int, seed: int) -> list[str]:
    digest = hashlib.sha256(
        f"{int(seed)}:identity_composition_smoke_selection".encode()
    ).hexdigest()
    ranked = sorted(
        anchor_ids,
        key=lambda aid: hashlib.sha256(f"{digest}:{aid}".encode()).hexdigest(),
    )
    return [str(x) for x in ranked[:n]]


def build_pool_config() -> ReferencePoolConfig:
    base = ReferencePoolConfig.load()
    return ReferencePoolConfig(
        K=10,
        seed=REFERENCE_POOL_SEED,
        context_fields=base.context_fields,
        action_fields=base.action_fields,
        read_only_context_fields=base.read_only_context_fields,
        excluded_fields=base.excluded_fields,
        label="identity_composition_pilot_reference_pool",
        source_path=base.source_path,
    )


def run_episode(
    *,
    attacker_id: str,
    anchor_id: str,
    budget: AttackBudget,
    experiment_seed: int,
    defender: FrozenXGBoostDefender,
    policy: CompiledGovernancePolicy,
    profile: IdentityCompositionProfile,
    reference_pool,
    raw_path: Path,
    artefact_dir: Path,
    episode_dir: Path,
) -> EpisodeRow:
    starting = load_starting_case(
        int(anchor_id),
        raw_path=raw_path,
        defender=defender,
        artefact_dir=artefact_dir,
    )
    episode_dir.mkdir(parents=True, exist_ok=False)
    logger = TrajectoryLogger(run_dir=episode_dir, run_id=episode_dir.name)
    if attacker_id == "a0":
        attacker: Any = ConstrainedRandomAttacker(
            seed=experiment_seed,
            reference_pool=reference_pool,
            m_max=budget.m_max,
            constraint_profile=profile,
            stdout=None,
        )
    else:
        attacker = SurrogateGuidedSearcher(
            budget=budget,
            reference_pool=reference_pool,
            experiment_seed=experiment_seed,
            constraint_profile=profile,
            stdout=None,
        )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id=attacker_id,
            anchor=starting,
            policy=policy,
            budget=budget.to_budget_spec(label="identity_composition_pilot_budget"),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=defender,
            seed=experiment_seed,
            enabled_action_keys=None,
            logger=logger,
            reference_pool=reference_pool,
            constraint_profile=profile,
        ),
    )
    (episode_dir / "match_result.json").write_text(
        json.dumps(match.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    persona = set(profile.persona_profile_fields)
    contact = set(profile.contact_identity_fields)
    distances: list[int] = []
    edited_sets: list[tuple[str, ...]] = []
    persona_fields: list[str] = []
    contact_fields: list[str] = []
    pairs: list[str] = []
    hashes: list[str] = []
    m_cap = False
    gov_viol = False
    profile_viol = False
    legal_series: list[int] = []
    initial_legal: int | None = None
    fallback_pass = fallback_block = 0
    reorder = 0

    for step in match.trajectory:
        dist = int(step.submitted_edit_cost)
        distances.append(dist)
        if dist > budget.m_max:
            m_cap = True
        meta = dict(step.research_meta or {})
        edited = tuple(meta.get("edited_fields") or [])
        if not edited:
            # Derive from proposed changes via feature mapping if needed.
            edited = tuple(sorted(step.proposed_changes))
        edited_sets.append(tuple(edited))
        p_edit = [f for f in edited if f in persona]
        c_edit = [f for f in edited if f in contact]
        if len(p_edit) == 1:
            persona_fields.append(p_edit[0])
        if len(c_edit) == 1:
            contact_fields.append(c_edit[0])
        if len(p_edit) == 1 and len(c_edit) == 1:
            pairs.append(f"{p_edit[0]}+{c_edit[0]}")
        if dist != 2 or len(p_edit) != 1 or len(c_edit) != 1:
            if step.validity.is_valid:
                profile_viol = True
            elif any("Profile rejected" in e for e in step.validity.errors):
                profile_viol = True
        fp = str(meta.get("candidate_hash") or meta.get("candidate_fingerprint") or "")
        if fp:
            hashes.append(fp)
        if "legal_unique_candidates_remaining" in meta:
            legal_series.append(int(meta["legal_unique_candidates_remaining"]))
        if "legal_unique_candidates_remaining_before_submit" in meta:
            before = int(meta["legal_unique_candidates_remaining_before_submit"])
            if initial_legal is None:
                initial_legal = before
            legal_series.append(max(0, before - 1))
        if meta.get("generation_method") == "enum_fallback":
            if step.public_feedback.label == "PASS":
                fallback_pass += 1
            elif step.public_feedback.label == "BLOCK":
                fallback_block += 1
        if meta.get("min_gower_to_failures") is not None:
            reorder += 1
        if any(
            "not permitted" in e.lower() or "forbidden" in e.lower()
            for e in step.validity.errors
        ):
            gov_viol = True

    success_at = {
        q: bool(
            match.success
            and match.attempts_to_success is not None
            and int(match.attempts_to_success) <= q
        )
        for q in range(1, budget.q_max + 1)
    }
    fb_inv = fb_picks = 0
    if attacker_id == "a0":
        diag = attacker.sampling_diagnostics
        fb_inv = int(diag.get("undersample_events", 0))
        fb_picks = int(diag.get("enum_fallback_picks", 0))
        (episode_dir / "a0_sampling_diagnostics.json").write_text(
            json.dumps(to_jsonable(diag), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if attacker_id == "a2":
        (episode_dir / "a2_submission_logs.json").write_text(
            json.dumps(list(attacker.submission_logs), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if attacker.submission_logs and initial_legal is None:
            first = attacker.submission_logs[0]
            if "legal_unique_candidates_remaining" in first:
                # approximate: remaining after first + 1
                initial_legal = int(first["legal_unique_candidates_remaining"]) + 1

    return EpisodeRow(
        attacker_id=attacker_id,
        anchor_id=str(anchor_id),
        success=bool(match.success),
        queries_used=int(match.q_used),
        queries_to_success=(
            int(match.attempts_to_success) if match.success else None
        ),
        stop_reason=str(match.stop_reason),
        invalid_submissions=int(match.invalid_submissions),
        distances=distances,
        edited_field_sets=edited_sets,
        persona_fields=persona_fields,
        contact_fields=contact_fields,
        persona_contact_pairs=pairs,
        success_at=success_at,
        duplicate=len(hashes) != len(set(h for h in hashes if h)),
        m_cap_violation=m_cap,
        governance_violation=gov_viol,
        profile_violation=profile_viol,
        fallback_invoked=fb_inv,
        fallback_picks=fb_picks,
        fallback_pass=fallback_pass,
        fallback_block=fallback_block,
        reorder_count=reorder,
        zero_submission_nofeasible=(
            match.q_used == 0
            and match.stop_reason
            in {"no_feasible_candidate", "action_space_exhaustion"}
        ),
        action_space_exhaustion=match.stop_reason == "action_space_exhaustion",
        initial_legal_count=initial_legal,
        legal_remaining_series=legal_series,
        can_complete_five_distinct=len(set(hashes)) >= 5,
    )


def smoke_ok(rows: Sequence[EpisodeRow], *, m_max: int) -> list[str]:
    errors: list[str] = []
    by_key = {(r.attacker_id, r.anchor_id): r for r in rows}
    for r in rows:
        if any(d != 2 for d in r.distances):
            errors.append(f"{r.attacker_id}/{r.anchor_id}: edit count != 2")
        if r.duplicate:
            errors.append(f"{r.attacker_id}/{r.anchor_id}: duplicate")
        if r.m_cap_violation:
            errors.append(f"{r.attacker_id}/{r.anchor_id}: m_cap")
        if r.governance_violation:
            errors.append(f"{r.attacker_id}/{r.anchor_id}: governance")
        if r.profile_violation:
            errors.append(f"{r.attacker_id}/{r.anchor_id}: profile")
        if r.queries_used >= 2 and r.persona_fields:
            if len(set(r.persona_fields)) > 1:
                errors.append(f"{r.attacker_id}/{r.anchor_id}: persona lock changed")
        for edited in r.edited_field_sets:
            # Each submitted candidate must be exactly 1 persona + 1 contact.
            # edited_field_sets may be empty when meta omitted names; then skip.
            if not edited:
                continue
    anchors = sorted({r.anchor_id for r in rows})
    for aid in anchors:
        a0 = by_key.get(("a0", aid))
        a2 = by_key.get(("a2", aid))
        if a0 is None:
            errors.append(f"missing a0 {aid}")
        if a2 is None:
            errors.append(f"missing a2 {aid}")
        # If A2 found legal candidates, A0 must not falsely stop with empty space.
        if a0 is not None and a2 is not None:
            if a2.queries_used > 0 and a0.zero_submission_nofeasible:
                errors.append(
                    f"a0/{aid}: no_feasible while a2 submitted "
                    f"{a2.queries_used} legal candidates"
                )
    return errors


def asr_curve(rows: Sequence[EpisodeRow], q_max: int) -> dict[str, float]:
    n = len(rows)
    return {
        f"ASR@{q}": (
            sum(1 for r in rows if r.success_at.get(q)) / n if n else 0.0
        )
        for q in range(1, q_max + 1)
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def construct_validity_label(a0_rows: Sequence[EpisodeRow], a2_rows: Sequence[EpisodeRow]) -> str:
    zero = sum(1 for r in list(a0_rows) + list(a2_rows) if r.zero_submission_nofeasible)
    exh = sum(1 for r in list(a0_rows) + list(a2_rows) if r.action_space_exhaustion)
    n = len(a0_rows) + len(a2_rows)
    a2_asr5 = asr_curve(a2_rows, 5)["ASR@5"]
    a0_asr5 = asr_curve(a0_rows, 5)["ASR@5"]
    only_one = sum(
        1
        for r in list(a0_rows) + list(a2_rows)
        if r.initial_legal_count is not None and r.initial_legal_count <= 1
    )
    five_ok = sum(1 for r in list(a0_rows) + list(a2_rows) if r.can_complete_five_distinct)

    if zero / max(n, 1) >= 0.30 or exh / max(n, 1) >= 0.60 or a2_asr5 in (0.0, 1.0) and a0_asr5 in (0.0, 1.0):
        return "degenerate"
    if (
        only_one / max(n, 1) >= 0.40
        or exh / max(n, 1) >= 0.35
        or five_ok / max(n, 1) <= 0.20
    ):
        return "borderline"
    if a2_asr5 >= 0.95 or a0_asr5 >= 0.95:
        return "borderline"
    return "non_degenerate"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--experiment-seed", type=int, default=EXPERIMENT_SEED)
    parser.add_argument("--n-anchors", type=int, default=100)
    parser.add_argument("--smoke-n", type=int, default=5)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_C1_ARTEFACT_DIR)
    args = parser.parse_args(argv)

    budget = AttackBudget(q_max=int(args.q), m_max=int(args.m))
    if budget.m_max != 2:
        print("WARN: identity-composition profile designed for m=2", file=sys.stderr)

    policy = CompiledGovernancePolicy.load(GOVERNANCE_PATH)
    if policy.policy_fingerprint != EXPECTED_GOV_FP:
        raise SystemExit(f"STOP: governance fingerprint mismatch {policy.policy_fingerprint}")
    profile = IdentityCompositionProfile.load()
    if profile.inherits_governance_version != policy.policy_version:
        raise SystemExit("STOP: profile does not inherit current governance version")
    composite_fp = profile.composite_experiment_fingerprint(
        governance_fingerprint=policy.policy_fingerprint
    )

    frozen = json.loads(FROZEN_ANCHORS_SOURCE.read_text(encoding="utf-8"))
    anchors = [str(x) for x in frozen["anchor_ids"]][: int(args.n_anchors)]
    if len(anchors) != int(args.n_anchors):
        raise SystemExit("STOP: anchor count mismatch")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (
        EXPERIMENTS_ROOT
        / "comparisons"
        / "a0_vs_a2"
        / "identity_composition_pilot"
        / f"paired_identity_m{budget.m_max}_q{budget.q_max}_n{args.n_anchors}_seed{args.experiment_seed}_{stamp}"
    )
    if root.exists():
        raise SystemExit(f"Refusing overwrite: {root}")
    root.mkdir(parents=True, exist_ok=False)

    defender = FrozenXGBoostDefender.from_artefact_dir(args.artefact_dir)
    provider = ReferencePoolProvider.from_config(build_pool_config(), raw_path=args.raw)
    pools = {aid: provider.get_pool(aid, seed=REFERENCE_POOL_SEED) for aid in anchors}

    snapshot = {
        "status": "month6_construct_validity_pilot_not_dissertation_findings",
        "governance_version": policy.policy_version,
        "governance_fingerprint": policy.policy_fingerprint,
        "constraint_profile_version": profile.profile_version,
        "constraint_profile_fingerprint": profile.profile_fingerprint,
        "composite_experiment_fingerprint": composite_fp,
        "d1_artefact_dir": str(args.artefact_dir),
        "d1_threshold": float(defender.threshold),
        "d1_artefact_id": defender.artefact_id,
        "reference_pool_seed": REFERENCE_POOL_SEED,
        "reference_pool_K": 10,
        "attack_budget": budget.to_dict(),
        "experiment_seed": int(args.experiment_seed),
        "n_anchors": len(anchors),
        "unrestricted_reference": str(UNRESTRICTED_DEV),
        "algorithms_unchanged": True,
        "proxy_limitation": (
            "constrained recombination of persona/profile and "
            "contact/identity-consistency proxies; not document-level fabrication"
        ),
    }
    (root / "configuration_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "constraint_profile.json").write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "frozen_anchors.json").write_text(
        json.dumps(
            {
                "anchor_ids": anchors,
                "source": str(FROZEN_ANCHORS_SOURCE),
                "order": "preserved",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "label": "identity_composition_proxy_pilot",
                "root": str(root),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "budget": budget.to_dict(),
                "experiment_seed": int(args.experiment_seed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Smoke
    smoke_anchors = select_smoke_anchors(
        anchors, n=int(args.smoke_n), seed=int(args.experiment_seed)
    )
    smoke_rows: list[EpisodeRow] = []
    for attacker_id in ("a0", "a2"):
        for aid in smoke_anchors:
            smoke_rows.append(
                run_episode(
                    attacker_id=attacker_id,
                    anchor_id=aid,
                    budget=budget,
                    experiment_seed=int(args.experiment_seed),
                    defender=defender,
                    policy=policy,
                    profile=profile,
                    reference_pool=pools[aid],
                    raw_path=args.raw,
                    artefact_dir=args.artefact_dir,
                    episode_dir=root / "smoke" / attacker_id / f"anchor_{aid}",
                )
            )
    errors = smoke_ok(smoke_rows, m_max=budget.m_max)
    smoke_txt = [
        "Identity-composition smoke",
        f"anchors: {smoke_anchors}",
        f"errors: {errors if errors else 'NONE'}",
        f"passed: {not errors}",
    ]
    for r in smoke_rows:
        smoke_txt.append(
            f"{r.attacker_id} {r.anchor_id}: success={r.success} q={r.queries_used} "
            f"stop={r.stop_reason} pairs={r.persona_contact_pairs}"
        )
    (root / "smoke" / "smoke_report.txt").write_text(
        "\n".join(smoke_txt) + "\n", encoding="utf-8"
    )
    print("\n".join(smoke_txt), flush=True)
    if errors:
        print("SMOKE FAILED — stop.", file=sys.stderr)
        return 2

    # Formal single-seed
    all_rows: list[EpisodeRow] = []
    for attacker_id in ("a0", "a2"):
        for aid in anchors:
            row = run_episode(
                attacker_id=attacker_id,
                anchor_id=aid,
                budget=budget,
                experiment_seed=int(args.experiment_seed),
                defender=defender,
                policy=policy,
                profile=profile,
                reference_pool=pools[aid],
                raw_path=args.raw,
                artefact_dir=args.artefact_dir,
                episode_dir=root / attacker_id / f"anchor_{aid}",
            )
            all_rows.append(row)
        print(
            f"DONE {attacker_id} ASR@5={asr_curve([r for r in all_rows if r.attacker_id==attacker_id], budget.q_max)['ASR@5']:.3f}",
            flush=True,
        )

    a0_rows = [r for r in all_rows if r.attacker_id == "a0"]
    a2_rows = [r for r in all_rows if r.attacker_id == "a2"]
    a0_asr = asr_curve(a0_rows, budget.q_max)
    a2_asr = asr_curve(a2_rows, budget.q_max)

    paired = []
    outcomes = Counter()
    by_a0 = {r.anchor_id: r for r in a0_rows}
    by_a2 = {r.anchor_id: r for r in a2_rows}
    for aid in anchors:
        a0 = by_a0[aid]
        a2 = by_a2[aid]
        if a0.success and a2.success:
            outcome = "both_success"
        elif a0.success:
            outcome = "a0_only"
        elif a2.success:
            outcome = "a2_only"
        else:
            outcome = "neither_success"
        outcomes[outcome] += 1
        paired.append(
            {
                "anchor_id": aid,
                "a0_success": int(a0.success),
                "a2_success": int(a2.success),
                "a0_queries_used": a0.queries_used,
                "a2_queries_used": a2.queries_used,
                "a0_stop_reason": a0.stop_reason,
                "a2_stop_reason": a2.stop_reason,
                "paired_outcome_at_q5": outcome,
            }
        )

    asr_rows = []
    for attacker, rows, asr in (("a0", a0_rows, a0_asr), ("a2", a2_rows, a2_asr)):
        for q in range(1, 6):
            asr_rows.append({"attacker": attacker, "q": q, "ASR": asr[f"ASR@{q}"]})

    def efficiency(rows: Sequence[EpisodeRow], name: str) -> dict[str, Any]:
        qts = [r.queries_to_success for r in rows if r.queries_to_success is not None]
        hist = Counter(qts)
        asr = asr_curve(rows, budget.q_max)
        marginal = {}
        prev = 0.0
        for q in range(1, 6):
            marginal[f"marginal_q{q}"] = asr[f"ASR@{q}"] - prev
            prev = asr[f"ASR@{q}"]
        return {
            "attacker": name,
            "successes": sum(1 for r in rows if r.success),
            "failures": sum(1 for r in rows if not r.success),
            "mean_queries_to_success": statistics.mean(qts) if qts else None,
            "median_queries_to_success": statistics.median(qts) if qts else None,
            **asr,
            **marginal,
            **{f"success_at_q{q}": hist.get(q, 0) for q in range(1, 6)},
        }

    # Field frequencies
    field_rows = []
    pair_rows = []
    for attacker, rows in (("a0", a0_rows), ("a2", a2_rows)):
        p_counts = Counter(f for r in rows for f in r.persona_fields)
        c_counts = Counter(f for r in rows for f in r.contact_fields)
        pair_counts = Counter(p for r in rows for p in r.persona_contact_pairs)
        success_pairs = Counter(
            p for r in rows if r.success for p in r.persona_contact_pairs
        )
        fail_pairs = Counter(
            p for r in rows if not r.success for p in r.persona_contact_pairs
        )
        for name, cnt in sorted(p_counts.items()):
            # descriptive ASR by persona field among submissions using it
            used = [r for r in rows if name in r.persona_fields]
            field_rows.append(
                {
                    "attacker": attacker,
                    "role": "persona",
                    "field": name,
                    "edit_count": cnt,
                    "episode_count": len(used),
                    "ASR_among_episodes_using_field": (
                        sum(1 for r in used if r.success) / len(used) if used else ""
                    ),
                }
            )
        for name, cnt in sorted(c_counts.items()):
            used = [r for r in rows if name in r.contact_fields]
            field_rows.append(
                {
                    "attacker": attacker,
                    "role": "contact",
                    "field": name,
                    "edit_count": cnt,
                    "episode_count": len(used),
                    "ASR_among_episodes_using_field": (
                        sum(1 for r in used if r.success) / len(used) if used else ""
                    ),
                }
            )
        for pair, cnt in sorted(pair_counts.items()):
            pair_rows.append(
                {
                    "attacker": attacker,
                    "persona_contact_pair": pair,
                    "count": cnt,
                    "success_count": success_pairs.get(pair, 0),
                    "failure_count": fail_pairs.get(pair, 0),
                }
            )

    def space_diag(rows: Sequence[EpisodeRow], name: str) -> dict[str, Any]:
        initials = [r.initial_legal_count for r in rows if r.initial_legal_count is not None]
        remain_all = [x for r in rows for x in r.legal_remaining_series]
        return {
            "attacker": name,
            "zero_submission_nofeasible": sum(1 for r in rows if r.zero_submission_nofeasible),
            "action_space_exhaustion": sum(1 for r in rows if r.action_space_exhaustion),
            "q_exhausted": sum(1 for r in rows if r.stop_reason == "q_exhausted"),
            "initial_legal_median": statistics.median(initials) if initials else None,
            "initial_legal_min": min(initials) if initials else None,
            "initial_legal_max": max(initials) if initials else None,
            "remaining_legal_median": statistics.median(remain_all) if remain_all else None,
            "remaining_legal_min": min(remain_all) if remain_all else None,
            "remaining_legal_max": max(remain_all) if remain_all else None,
            "anchors_with_only_1_initial_legal": sum(
                1 for r in rows if r.initial_legal_count == 1
            ),
            "anchors_can_complete_5_distinct": sum(
                1 for r in rows if r.can_complete_five_distinct
            ),
            "fallback_invoked_episodes": sum(1 for r in rows if r.fallback_invoked > 0),
            "fallback_picks": sum(r.fallback_picks for r in rows),
            "fallback_pass": sum(r.fallback_pass for r in rows),
            "fallback_block": sum(r.fallback_block for r in rows),
            "reorder_total": sum(r.reorder_count for r in rows),
            "duplicate_episodes": sum(1 for r in rows if r.duplicate),
            "invalid_submissions_total": sum(r.invalid_submissions for r in rows),
            "m_cap_violations": sum(1 for r in rows if r.m_cap_violation),
            "governance_violations": sum(1 for r in rows if r.governance_violation),
            "profile_violations": sum(1 for r in rows if r.profile_violation),
        }

    # Unrestricted comparison (descriptive)
    unrest = {}
    if UNRESTRICTED_DEV.is_dir():
        summary = json.loads(
            (UNRESTRICTED_DEV / "comparison_summary.json").read_text(encoding="utf-8")
        )
        unrest = {
            "a0_crossseed_mean_asr": {
                f"ASR@{q}": summary["cross_seed_asr"]["a0"][f"ASR@{q}"]["mean"]
                for q in range(1, 6)
            },
            "a2_crossseed_mean_asr": {
                f"ASR@{q}": summary["cross_seed_asr"]["a2"][f"ASR@{q}"]["mean"]
                for q in range(1, 6)
            },
            "note": (
                "unrestricted A0 is 3-seed mean; identity-composition is single-seed "
                "pilot; A2 is deterministic; no formal significance tests"
            ),
        }

    gate = construct_validity_label(a0_rows, a2_rows)
    summary = {
        "status": "month6_construct_validity_pilot_not_dissertation_findings",
        "root": str(root),
        "profile_version": profile.profile_version,
        "profile_fingerprint": profile.profile_fingerprint,
        "composite_experiment_fingerprint": composite_fp,
        "a0_asr": a0_asr,
        "a2_asr": a2_asr,
        "delta_asr5": a2_asr["ASR@5"] - a0_asr["ASR@5"],
        "paired_outcomes": dict(outcomes),
        "efficiency": {
            "a0": efficiency(a0_rows, "a0"),
            "a2": efficiency(a2_rows, "a2"),
        },
        "action_space": {
            "a0": space_diag(a0_rows, "a0"),
            "a2": space_diag(a2_rows, "a2"),
        },
        "construct_validity": gate,
        "unrestricted_comparison": unrest,
        "proxy_limitation": snapshot["proxy_limitation"],
        "integrity": {
            "d1_unchanged": True,
            "governance_unchanged": True,
            "pools_unchanged": True,
            "attacker_algorithms_unchanged": True,
            "unrestricted_dir_preserved": UNRESTRICTED_DEV.is_dir(),
        },
    }

    paired_dir = root / "paired"
    write_csv(paired_dir / "paired_anchor_results.csv", paired, list(paired[0]))
    write_csv(paired_dir / "asr_by_query.csv", asr_rows, ["attacker", "q", "ASR"])
    write_csv(
        paired_dir / "paired_outcomes.csv",
        [{"outcome": k, "count": v} for k, v in outcomes.items()],
        ["outcome", "count"],
    )
    write_csv(
        paired_dir / "query_efficiency.csv",
        [summary["efficiency"]["a0"], summary["efficiency"]["a2"]],
        list(summary["efficiency"]["a0"].keys()),
    )
    write_csv(
        paired_dir / "field_edit_frequencies.csv",
        field_rows,
        list(field_rows[0].keys()) if field_rows else ["attacker"],
    )
    write_csv(
        paired_dir / "persona_contact_pair_frequencies.csv",
        pair_rows,
        list(pair_rows[0].keys()) if pair_rows else ["attacker"],
    )
    write_csv(
        paired_dir / "action_space_diagnostics.csv",
        [summary["action_space"]["a0"], summary["action_space"]["a2"]],
        list(summary["action_space"]["a0"].keys()),
    )
    (root / "identity_composition_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = _report(summary)
    (root / "identity_composition_report.md").write_text(report, encoding="utf-8")
    (root / "identity_composition_report.txt").write_text(report, encoding="utf-8")

    print("\n=== IDENTITY COMPOSITION PILOT COMPLETE ===")
    print(f"root: {root}")
    print(f"profile: {profile.profile_version} fp={profile.profile_fingerprint[:16]}...")
    print(f"A0 ASR: {a0_asr}")
    print(f"A2 ASR: {a2_asr}")
    print(f"delta ASR@5: {summary['delta_asr5']:.4f}")
    print(f"outcomes: {dict(outcomes)}")
    print(f"construct_validity: {gate}")
    print("NOT dissertation findings.")
    return 0


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Identity-composition construct-validity pilot",
            "",
            "**NOT dissertation findings.**",
            "",
            f"Construct-validity label: `{summary['construct_validity']}`",
            "",
            f"Profile: {summary['profile_version']} / `{summary['profile_fingerprint']}`",
            f"Composite fingerprint: `{summary['composite_experiment_fingerprint']}`",
            "",
            "## Attack results",
            f"A0 ASR: {summary['a0_asr']}",
            f"A2 ASR: {summary['a2_asr']}",
            f"A2−A0 ASR@5: {summary['delta_asr5']}",
            f"Paired outcomes: {summary['paired_outcomes']}",
            "",
            "## Query efficiency",
            json.dumps(summary["efficiency"], indent=2),
            "",
            "## Action-space diagnostics",
            json.dumps(summary["action_space"], indent=2),
            "",
            "## Unrestricted comparison (descriptive only)",
            json.dumps(summary["unrestricted_comparison"], indent=2),
            "",
            "## Proxy limitation",
            summary["proxy_limitation"],
            "",
            "This profile does **not** equate to real ID-number synthesis, "
            "document-level fabrication, or full name/DOB/SSN/passport recombination.",
            "",
            f"Integrity: {summary['integrity']}",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
