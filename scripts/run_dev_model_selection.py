#!/usr/bin/env python3
"""Pinned development model-selection runner (NOT dissertation findings).

Phases:
  sanity     A1-Pro + A3-Pro on the leading 15 same-arena Month-6 anchors
  benchmark  A0, A1-Flash, A1-Pro, A2, A3-Flash, A3-Pro on the leading 50

Does not redesign attackers, change prompts, governance, D1, K, or Q/m.
Does not open Month 7.  Does not overwrite historical artefacts.
Pro is an additional selectable model value on the same A1 V4.4 /
A3 V2.4 implementations; Flash remains preserved and is not renamed or migrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

IMPL = Path(__file__).resolve().parents[1]
ROOT = IMPL.parent
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

_env_path = ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker  # noqa: E402
from attack_lab.attackers.a1_planner import (  # noqa: E402
    DeepSeekPlannerClient,
    LLMCompletion,
    OneShotLLMPlanner,
)
from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES  # noqa: E402
from attack_lab.archive.contracts.a1_v4_2_contract import (  # noqa: E402
    scan_attacker_visible_hidden_mentions,
)
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher  # noqa: E402
from attack_lab.attackers.a3_agent import EpisodicLLMAgent  # noqa: E402
from attack_lab.benchmark_pins import (  # noqa: E402
    MODEL_FLASH,
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
    PINNED_GOVERNANCE_FINGERPRINT,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
    REASONING_EFFORT_MAX,
    assert_pinned_defence_identity,
    assert_reference_pool_fingerprint,
    assert_thinking_cell_config,
    condition_manifest,
    estimate_deepseek_cost_usd,
    inspect_constructed_attacker,
    pinned_attacker_summary,
    require_supported_llm_model,
)
from attack_lab.budget import AttackBudget, BudgetSpec  # noqa: E402
from attack_lab.cases import DEFAULT_RAW_PATH, load_starting_case  # noqa: E402
from attack_lab.defender import FrozenXGBoostDefender  # noqa: E402
from attack_lab.feedback import FeedbackPolicy  # noqa: E402
from attack_lab.governance import CompiledGovernancePolicy  # noqa: E402
from attack_lab.logger import TrajectoryLogger  # noqa: E402
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator  # noqa: E402
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR, new_run_directory  # noqa: E402
from attack_lab.reference_actions import (  # noqa: E402
    audit_reference_provenance,
    provenance_audit_counts,
)
from attack_lab.reference_pool import (  # noqa: E402
    ReferencePoolConfig,
    ReferencePoolProvider,
)
from attack_lab.types import to_jsonable  # noqa: E402

K = 10
M_MAX = 2
Q_MAX = 5
EXPERIMENT_SEED = 1
REFERENCE_POOL_SEED = 20260803
FEEDBACK_MODE = "label_only"
MAX_A1_LOCAL_GENERATION_ATTEMPTS = 3

PRIOR_ARENA_SMOKE = (
    ROOT
    / "05_outputs"
    / "scratch"
    / "smoke"
    / "a0_a1_a2_k10_integration_smoke_N25_m2_Q5_seed1_20260811T185354Z"
)
REPAIR_N15_SMOKE = (
    ROOT
    / "05_outputs"
    / "scratch"
    / "smoke"
    / "abstract_proxy_action_equality_repair_N15_m2_Q5_seed1_20260812T155629Z"
)
FROZEN_ANCHORS_SOURCE = (
    ROOT
    / "05_outputs"
    / "experiments"
    / "a0"
    / "calibration"
    / "governance_v2"
    / "formal_multiseed_m123_20260804T161724Z"
    / "frozen_anchors.json"
)
GOVERNANCE_PATH = IMPL / "config" / "attacker_compiled_governance.json"
EXPECTED_GOV = PINNED_GOVERNANCE_FINGERPRINT
EXPECTED_D1 = PINNED_D1_ARTEFACT_ID

ANCHORS_15 = [
    "833593",
    "880887",
    "852535",
    "822823",
    "847673",
    "876140",
    "836887",
    "837488",
    "815786",
    "900876",
    "813798",
    "867883",
    "872683",
    "833490",
    "897178",
]
ANCHORS_25 = ANCHORS_15 + [
    "819054",
    "815497",
    "795826",
    "882729",
    "803185",
    "807296",
    "861572",
    "873113",
    "848026",
    "831820",
]

HIDDEN_PROMPT_FILES = {
    "a1_prompt_full.txt",
    "a3_prompt_payload.json",
    "a3_prompt_full.txt",
    "public_transcript.txt",
}


@dataclass
class TransportCallRecord:
    requested_model: str
    returned_model: str
    system_fingerprint: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_ms: float
    thinking_disabled: bool
    reasoning_effort: str | None


@dataclass
class RecordingLLMClient:
    """Pass-through client that records requested/returned model identity."""

    inner: DeepSeekPlannerClient
    calls: list[TransportCallRecord] = field(default_factory=list)

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
        reasoning_effort: str | None = None,
    ) -> LLMCompletion:
        completion = self.inner.complete(
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            thinking_disabled=thinking_disabled,
            reasoning_effort=reasoning_effort,
        )
        self.calls.append(
            TransportCallRecord(
                requested_model=str(model),
                returned_model=str(completion.model),
                system_fingerprint=completion.system_fingerprint,
                prompt_tokens=int(completion.prompt_tokens),
                completion_tokens=int(completion.completion_tokens),
                cached_tokens=int(completion.cached_tokens),
                latency_ms=float(completion.latency_ms),
                thinking_disabled=bool(thinking_disabled),
                reasoning_effort=reasoning_effort,
            )
        )
        return completion


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def feature_fingerprint(features: Mapping[str, Any]) -> str:
    payload = json.dumps(
        to_jsonable(dict(features)), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_integration_smoke_anchors(
    anchor_ids: Sequence[str], *, n: int, seed: int
) -> list[str]:
    digest = hashlib.sha256(
        f"{int(seed)}:a0_a1_a2_integration_smoke_anchor_selection".encode("utf-8")
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    ordered = [str(x) for x in anchor_ids]
    rng.shuffle(ordered)
    if n > len(ordered):
        raise SystemExit(f"Need {n} anchors; only {len(ordered)} available.")
    return ordered[:n]


def build_pool_config() -> ReferencePoolConfig:
    base = ReferencePoolConfig.load()
    return ReferencePoolConfig(
        K=K,
        seed=REFERENCE_POOL_SEED,
        context_fields=base.context_fields,
        action_fields=base.action_fields,
        read_only_context_fields=base.read_only_context_fields,
        excluded_fields=base.excluded_fields,
        label="dev_model_selection_reference_pool",
        source_path=base.source_path,
    )


def asr_curve(rows: Sequence[Mapping[str, Any]], q_max: int = Q_MAX) -> dict[str, float]:
    n = len(rows)
    return {
        f"ASR@{q}": (
            sum(
                1
                for row in rows
                if row.get("attempts_to_success") is not None
                and int(row["attempts_to_success"]) <= q
            )
            / n
            if n
            else 0.0
        )
        for q in range(1, q_max + 1)
    }


def first_success_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [
        int(row["attempts_to_success"])
        for row in rows
        if row.get("attempts_to_success") is not None
    ]
    n = len(rows)
    return {
        "n": n,
        "n_success": len(successes),
        "mean_q_to_success_among_successes": (
            float(sum(successes) / len(successes)) if successes else None
        ),
        "median_q_to_success_among_successes": (
            float(sorted(successes)[len(successes) // 2]) if successes else None
        ),
        "success_at_q": {str(q): successes.count(q) for q in range(1, Q_MAX + 1)},
    }


def scan_raw_proxy_exposure(episode_dir: Path) -> list[str]:
    hits: list[str] = []
    for path in episode_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name not in HIDDEN_PROMPT_FILES and "prompt" not in path.name.lower():
            continue
        if path.suffix not in {".txt", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for raw in sorted(PROXY_RAW_FEATURE_NAMES):
            if raw in text:
                hits.append(f"{path.relative_to(episode_dir)}:{raw}")
    return hits


def scan_hidden_exposure(episode_dir: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in episode_dir.rglob("*"):
        if not path.is_file() or path.name not in HIDDEN_PROMPT_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for item in scan_attacker_visible_hidden_mentions(text):
            cls = str(item.get("class") or "")
            if cls.startswith("A_") or cls.startswith("B_"):
                findings.append({"file": path.name, **item})
    return findings


def audit_local_repair_pin(query_records: Sequence[Mapping[str, Any]]) -> bool:
    for rec in query_records:
        locals_ = list(rec.get("local_generation_records") or [])
        if len(locals_) < 2:
            continue
        pinned = None
        for item in locals_:
            if item.get("local_rejection_reason") is None and item.get("env_step_called"):
                if pinned is not None:
                    if item.get("strategy_label") != pinned.get("strategy_label"):
                        return False
                    if item.get("adaptation_note") != pinned.get("adaptation_note"):
                        return False
                continue
            if (
                pinned is None
                and item.get("strategy_label")
                and item.get("adaptation_note")
            ):
                pinned = item
                continue
            if pinned is not None and item.get("local_rejection_reason"):
                if item.get("strategy_label") not in {None, pinned.get("strategy_label")}:
                    return False
                if item.get("adaptation_note") not in {
                    None,
                    pinned.get("adaptation_note"),
                }:
                    return False
        changes = dict(rec.get("changes") or {})
        reflection = changes.get("reflection_update")
        if pinned is not None and isinstance(reflection, Mapping):
            if reflection.get("hypothesis") != pinned.get("adaptation_note"):
                return False
            if rec.get("strategy_label") != pinned.get("strategy_label"):
                return False
    return True


def audit_a3_prompt_mechanics(run_dir: Path) -> dict[str, Any]:
    reflection_timing_ok = True
    stable_slots_ok = True
    episode_maps: list[dict[str, str]] = []
    for prompt_path in sorted(run_dir.rglob("a3_prompt_payload.json")):
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        if payload.get("prompt_version") != PINNED_A3_PROMPT_VERSION:
            reflection_timing_ok = False
            continue
        q = int((payload.get("budget") or {}).get("query_index") or 0)
        memory = payload.get("episodic_memory") or []
        if q > 1:
            if not memory:
                reflection_timing_ok = False
            elif str(memory[-1].get("public_label")) not in {
                "PASS",
                "BLOCK",
                "INVALID",
            }:
                reflection_timing_ok = False
        mapping = {
            str(e.get("action_slot_id")): str(e.get("action_key"))
            for e in (payload.get("episode_action_slot_map") or [])
        }
        if mapping:
            episode_maps.append(mapping)
    if len(episode_maps) >= 2:
        base = episode_maps[0]
        for later in episode_maps[1:]:
            for sid, key in base.items():
                if sid in later and later[sid] != key:
                    stable_slots_ok = False
    return {
        "reflection_timing_ok": reflection_timing_ok,
        "stable_action_slot_mapping_ok": stable_slots_ok,
    }


def catalogue_has_all_abstract_proxies(episode_dir: Path) -> bool:
    needed = {
        "name_email_alignment",
        "home_phone_configuration",
        "mobile_phone_configuration",
    }
    for path in episode_dir.rglob("*prompt_payload.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload)
        if all(name in text for name in needed):
            return True
    prompt_txt = episode_dir / "a1_prompt_full.txt"
    if prompt_txt.exists():
        text = prompt_txt.read_text(encoding="utf-8", errors="ignore")
        return all(name in text for name in needed)
    return False


def abstract_proxy_submissions(match) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for step in match.trajectory:
        for key in (step.proposed_changes or {}):
            if key in {
                "name_email_alignment",
                "home_phone_configuration",
                "mobile_phone_configuration",
            }:
                counts[str(key)] += 1
    return dict(counts)


def audit_match_provenance(match, starting, pool) -> dict[str, Any]:
    backed = 0
    non_ref = 0
    for step in match.trajectory:
        meta = dict(step.research_meta or {})
        edited = list(meta.get("edited_fields") or [])
        candidate = dict(step.validity.candidate_features or {})
        if not edited:
            continue
        if not candidate:
            non_ref += len(edited)
            continue
        audit = audit_reference_provenance(
            anchor=starting.features,
            candidate=candidate,
            pool=pool,
            changed_fields=edited,
        )
        counts = provenance_audit_counts(audit)
        backed += int(counts["reference_backed"])
        non_ref += int(counts["non_reference_backed"])
    return {
        "reference_backed": backed,
        "non_reference_backed": non_ref,
        "provenance_rate": (backed / (backed + non_ref)) if (backed + non_ref) else 1.0,
    }


def transport_summary(recorder: RecordingLLMClient | None, llm_model: str | None) -> dict[str, Any]:
    if recorder is None or llm_model is None:
        return {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "estimated_cost_usd": 0.0,
            "requested_models": [],
            "returned_models": [],
            "system_fingerprints": [],
        }
    prompt = sum(c.prompt_tokens for c in recorder.calls)
    completion = sum(c.completion_tokens for c in recorder.calls)
    cached = sum(c.cached_tokens for c in recorder.calls)
    return {
        "llm_calls": len(recorder.calls),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "estimated_cost_usd": estimate_deepseek_cost_usd(
            model=llm_model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
        ),
        "requested_models": sorted({c.requested_model for c in recorder.calls}),
        "returned_models": sorted({c.returned_model for c in recorder.calls}),
        "system_fingerprints": sorted(
            {c.system_fingerprint for c in recorder.calls if c.system_fingerprint}
        ),
        "calls": [
            {
                "requested_model": c.requested_model,
                "returned_model": c.returned_model,
                "system_fingerprint": c.system_fingerprint,
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "cached_tokens": c.cached_tokens,
                "latency_ms": c.latency_ms,
                "thinking_disabled": c.thinking_disabled,
                "reasoning_effort": c.reasoning_effort,
            }
            for c in recorder.calls
        ],
    }


def build_attacker(
    *,
    condition_id: str,
    attacker_kind: str,
    llm_model: str | None,
    pool,
    recorder: RecordingLLMClient | None,
    thinking_disabled: bool = True,
    reasoning_effort: str | None = None,
):
    budget = AttackBudget(q_max=Q_MAX, m_max=M_MAX)
    if attacker_kind in {"a1", "a3"}:
        assert_thinking_cell_config(
            thinking_disabled=thinking_disabled,
            reasoning_effort=reasoning_effort,
            expect_thinking_disabled=thinking_disabled,
        )
    if attacker_kind == "a0":
        attacker = ConstrainedRandomAttacker(
            seed=EXPERIMENT_SEED,
            reference_pool=pool,
            m_max=M_MAX,
            attacker_id="a0",
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id=condition_id if condition_id == "A0" else "a0",
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
        )
        return attacker
    if attacker_kind == "a1":
        assert llm_model is not None
        attacker = OneShotLLMPlanner(
            experiment_seed=EXPERIMENT_SEED,
            reference_pool=pool,
            budget=budget,
            attacker_id="a1",
            prompt_version=PINNED_A1_PROMPT_VERSION,
            model=llm_model,
            max_local_generation_attempts=MAX_A1_LOCAL_GENERATION_ATTEMPTS,
            thinking_disabled=bool(thinking_disabled),
            reasoning_effort=reasoning_effort,
            llm_client=recorder,
            stdout=None,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id=condition_id,
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
            llm_model=llm_model,
            expect_thinking_disabled=bool(thinking_disabled),
        )
        return attacker
    if attacker_kind == "a2":
        attacker = SurrogateGuidedSearcher(
            budget=budget,
            reference_pool=pool,
            experiment_seed=EXPERIMENT_SEED,
            attacker_id="a2",
            gower_policy=PINNED_A2_GOWER_POLICY,
            stdout=None,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id=condition_id if condition_id == "A2" else "a2",
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
        )
        return attacker
    if attacker_kind == "a3":
        assert llm_model is not None
        attacker = EpisodicLLMAgent(
            experiment_seed=EXPERIMENT_SEED,
            reference_pool=pool,
            budget=budget,
            attacker_id="a3",
            prompt_version=PINNED_A3_PROMPT_VERSION,
            model=llm_model,
            thinking_disabled=bool(thinking_disabled),
            reasoning_effort=reasoning_effort,
            llm_client=recorder,
            stdout=None,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id=condition_id,
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
            llm_model=llm_model,
            expect_thinking_disabled=bool(thinking_disabled),
        )
        return attacker
    raise SystemExit(f"Unknown attacker_kind={attacker_kind!r} for {condition_id}")


def prompt_family_hash(prompt_version: str | None) -> str | None:
    if not prompt_version:
        return None
    return hashlib.sha256(str(prompt_version).encode("utf-8")).hexdigest()


def attacker_condition_manifest(
    *,
    condition_id: str,
    attacker_kind: str,
    llm_model: str | None,
    attacker: Any,
    thinking_disabled: bool | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    prompt_version = getattr(attacker, "prompt_version", None)
    gower_policy = getattr(attacker, "gower_policy", None)
    config_hash = getattr(attacker, "config_hash", None)
    if callable(config_hash):
        config_hash = config_hash()
    if not config_hash:
        config_hash = getattr(attacker, "_config_hash", None)
    return condition_manifest(
        condition_id=condition_id,
        attacker_kind=attacker_kind,
        prompt_version=prompt_version,
        model=llm_model or getattr(attacker, "model", None),
        thinking_disabled=thinking_disabled,
        reasoning_effort=reasoning_effort,
        config_hash=config_hash,
        prompt_hash=prompt_family_hash(prompt_version),
        gower_policy=gower_policy,
    )


def preflight_defence_and_pins(
    *,
    policy: CompiledGovernancePolicy,
    defender: FrozenXGBoostDefender,
    artefact_dir: Path,
) -> None:
    assert_pinned_defence_identity(
        d1_artefact_id=defender.artefact_id,
        governance_fingerprint=policy.policy_fingerprint,
        require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
        month7_path_fragment=str(artefact_dir),
    )
    if str(PINNED_A1_PROMPT_VERSION) != "a1_oneshot_v4_3_public_reference_view":
        raise SystemExit(
            f"Authoritative A1 pin drifted: {PINNED_A1_PROMPT_VERSION!r}"
        )
    if str(PINNED_A3_PROMPT_VERSION) != (
        "a3_episodic_reflective_v2_3_public_reference_view"
    ):
        raise SystemExit(
            f"Authoritative A3 pin drifted: {PINNED_A3_PROMPT_VERSION!r}"
        )


def run_episode(
    *,
    condition_id: str,
    attacker_kind: str,
    llm_model: str | None,
    anchor_id: str,
    policy: CompiledGovernancePolicy,
    defender: FrozenXGBoostDefender,
    provider: ReferencePoolProvider,
    episode_dir: Path,
    raw_path: Path,
    thinking_disabled: bool = True,
    reasoning_effort: str | None = None,
    expected_pool_fingerprint: str | None = None,
) -> dict[str, Any]:
    if "month7" in str(episode_dir).lower() or "month_7" in str(episode_dir).lower():
        raise SystemExit("Refusing to write under a Month-7 path.")
    starting = load_starting_case(
        anchor_id,
        raw_path=raw_path,
        defender=defender,
        artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
    )
    if starting.data_split != "dev_month6":
        raise SystemExit(f"Anchor {anchor_id} is not month-6: {starting.data_split}")
    if starting.initial_decision != "BLOCK":
        raise SystemExit(f"Anchor {anchor_id} is not D1 BLOCK.")
    pool = provider.get_pool(anchor_id, seed=REFERENCE_POOL_SEED)
    if expected_pool_fingerprint is not None:
        assert_reference_pool_fingerprint(
            observed=pool.pool_fingerprint,
            expected=expected_pool_fingerprint,
            anchor_id=str(anchor_id),
        )
    recorder = None
    if llm_model is not None:
        recorder = RecordingLLMClient(
            inner=DeepSeekPlannerClient(default_model=llm_model)
        )
    attacker = build_attacker(
        condition_id=condition_id,
        attacker_kind=attacker_kind,
        llm_model=llm_model,
        pool=pool,
        recorder=recorder,
        thinking_disabled=thinking_disabled,
        reasoning_effort=reasoning_effort,
    )
    logger = TrajectoryLogger(
        run_dir=episode_dir,
        run_id=f"{condition_id}_{anchor_id}_seed{EXPERIMENT_SEED}",
    )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id=attacker.attacker_id,
            anchor=starting,
            policy=policy,
            budget=BudgetSpec.development_dummy(
                q_max=Q_MAX, m_max=M_MAX, label="dev_model_selection_qm"
            ),
            feedback_policy=FeedbackPolicy(mode=FEEDBACK_MODE),
            defender=defender,
            seed=EXPERIMENT_SEED,
            logger=logger,
            reference_pool=pool,
            require_reference_provenance=PINNED_REQUIRE_REFERENCE_PROVENANCE,
        ),
    )
    (episode_dir / "match_result.json").write_text(
        json.dumps(match.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = audit_match_provenance(match, starting, pool)
    m_viol = sum(
        1
        for step in match.trajectory
        if int(getattr(step, "submitted_edit_cost", 0) or 0) > M_MAX
    )
    q_viol = max(0, int(match.q_used) - Q_MAX)
    if int(match.scored_defender_queries) != int(match.q_used):
        q_viol += abs(int(match.scored_defender_queries) - int(match.q_used))
    hidden = scan_hidden_exposure(episode_dir)
    raw_proxy = scan_raw_proxy_exposure(episode_dir)
    post_freeze = 0
    if attacker_kind == "a1":
        call = getattr(attacker, "call_record", None)
        if call is not None and len(call.retry_ledger) != int(call.llm_call_count):
            post_freeze = 1
        if int(getattr(call, "q_used_before_freeze", 0) or 0) != 0:
            post_freeze += 1
        if int(getattr(call, "d1_calls_before_freeze", 0) or 0) != 0:
            post_freeze += 1
    a3_mech = {}
    if attacker_kind == "a3":
        query_records = [r.to_dict() for r in attacker.query_records]
        a3_mech = {
            "local_repair_reflection_pin_ok": audit_local_repair_pin(query_records),
            **audit_a3_prompt_mechanics(episode_dir),
            "reflection_modes": dict(
                Counter(
                    str((mem.get("reflection_update") or {}).get("mode") or "")
                    for mem in getattr(attacker, "_v2_episodic_memory", [])
                    if (mem.get("reflection_update") or {}).get("mode")
                )
            ),
            "selection_count_exceeds_residual_m_after_repair": int(
                attacker.aggregate_counters().get(
                    "selection_count_exceeds_residual_m_after_repair"
                )
                or 0
            ),
        }
    transport = transport_summary(recorder, llm_model)
    if recorder is not None:
        write_json(episode_dir / "llm_transport_identity.json", transport)
    cfg_hash = getattr(attacker, "config_hash", None)
    if callable(cfg_hash):
        cfg_hash = cfg_hash()
    if not cfg_hash:
        cfg_hash = getattr(attacker, "_config_hash", None)
    runtime_prompt_hash = getattr(attacker, "_prompt_hash", None) or getattr(
        attacker, "prompt_hash", None
    )
    if callable(runtime_prompt_hash):
        runtime_prompt_hash = runtime_prompt_hash()
    if not runtime_prompt_hash:
        runtime_prompt_hash = None
    prompt_version = getattr(attacker, "prompt_version", None)
    row = {
        "condition_id": condition_id,
        "attacker_kind": attacker_kind,
        "attacker_version": prompt_version or getattr(attacker, "gower_policy", None),
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "prompt_hash": runtime_prompt_hash or prompt_family_hash(prompt_version),
        "gower_policy": getattr(attacker, "gower_policy", None),
        "thinking_disabled": (
            bool(getattr(attacker, "thinking_disabled", thinking_disabled))
            if attacker_kind in {"a1", "a3"}
            else None
        ),
        "thinking_enabled": (
            (not bool(getattr(attacker, "thinking_disabled", thinking_disabled)))
            if attacker_kind in {"a1", "a3"}
            else None
        ),
        "reasoning_effort": (
            getattr(attacker, "reasoning_effort", reasoning_effort)
            if attacker_kind in {"a1", "a3"}
            else None
        ),
        "config_hash": cfg_hash,
        "anchor_id": str(anchor_id),
        "success": bool(match.success),
        "stop_reason": str(match.stop_reason),
        "q_used": int(match.q_used),
        "attempts_to_success": match.attempts_to_success,
        "scored_defender_queries": int(match.scored_defender_queries),
        "invalid_submissions": int(match.invalid_submissions),
        "reaching_d1": int(match.scored_defender_queries) >= 1,
        "Q_violations": q_viol,
        "m_violations": m_viol,
        "hidden_exposure": len(hidden),
        "raw_proxy_exposure": len(raw_proxy),
        "raw_proxy_exposure_hits": raw_proxy,
        "post_freeze_adaptation": post_freeze,
        "catalogue_has_all_abstract_proxies": catalogue_has_all_abstract_proxies(
            episode_dir
        ),
        "abstract_proxy_submissions": abstract_proxy_submissions(match),
        "anchor_feature_fingerprint": feature_fingerprint(starting.features),
        "pool_fingerprint": pool.pool_fingerprint,
        "governance_fingerprint": policy.policy_fingerprint,
        "d1_artefact_id": defender.artefact_id,
        "month7_opened": False,
        **provenance,
        **a3_mech,
        **{f"transport_{k}": v for k, v in transport.items() if k != "calls"},
    }
    write_json(episode_dir / "episode_diag.json", row)
    return row


def condition_integrity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in rows if not r.get("runner_exception")]
    catalogue_rows = [
        r for r in evaluated if r.get("catalogue_has_all_abstract_proxies") is not None
    ]
    return {
        "n": len(rows),
        "completed": len(evaluated),
        "runner_exceptions": sum(1 for r in rows if r.get("runner_exception")),
        "success": sum(1 for r in evaluated if r.get("success")),
        "Q_violations": sum(int(r.get("Q_violations") or 0) for r in evaluated),
        "m_violations": sum(int(r.get("m_violations") or 0) for r in evaluated),
        "hidden_exposure": sum(int(r.get("hidden_exposure") or 0) for r in evaluated),
        "raw_proxy_exposure": sum(
            int(r.get("raw_proxy_exposure") or 0) for r in evaluated
        ),
        "non_reference_backed": sum(
            int(r.get("non_reference_backed") or 0) for r in evaluated
        ),
        "post_freeze_adaptation": sum(
            int(r.get("post_freeze_adaptation") or 0) for r in evaluated
        ),
        "reflection_timing_ok": all(
            r.get("reflection_timing_ok", True) for r in evaluated
        ),
        "local_repair_reflection_pin_ok": all(
            r.get("local_repair_reflection_pin_ok", True) for r in evaluated
        ),
        "stable_action_slot_mapping_ok": all(
            r.get("stable_action_slot_mapping_ok", True) for r in evaluated
        ),
        "catalogue_all_abstract_proxies": (
            all(bool(r.get("catalogue_has_all_abstract_proxies")) for r in catalogue_rows)
            if catalogue_rows
            else True
        ),
        "requested_models": sorted(
            {
                m
                for r in evaluated
                for m in (r.get("transport_requested_models") or [])
            }
        ),
        "returned_models": sorted(
            {
                m
                for r in evaluated
                for m in (r.get("transport_returned_models") or [])
            }
        ),
        "system_fingerprints": sorted(
            {
                fp
                for r in evaluated
                for fp in (r.get("transport_system_fingerprints") or [])
            }
        ),
        "llm_calls": sum(int(r.get("transport_llm_calls") or 0) for r in evaluated),
        "prompt_tokens": sum(
            int(r.get("transport_prompt_tokens") or 0) for r in evaluated
        ),
        "completion_tokens": sum(
            int(r.get("transport_completion_tokens") or 0) for r in evaluated
        ),
        "estimated_cost_usd": sum(
            float(r.get("transport_estimated_cost_usd") or 0.0) for r in evaluated
        ),
    }


def returned_model_identifies_as_pro(model: str) -> bool:
    """True if a live API returned-model string is Pro, not Flash.

    Accepts the request alias ``deepseek-v4-pro`` and concrete provider
    versions such as ``DeepSeek-V4-Pro-0813``.  Empty strings and any
    Flash identity fail closed.
    """
    text = str(model or "").strip()
    if not text:
        return False
    lowered = text.lower().replace("_", "-")
    if "flash" in lowered:
        return False
    return lowered == MODEL_PRO or "v4-pro" in lowered


def sanity_gates(condition_id: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    integ = condition_integrity(rows)
    errors: list[str] = []
    if integ["n"] != 15:
        errors.append(f"{condition_id}: expected 15 episode rows, got {integ['n']}")
    if integ["runner_exceptions"]:
        errors.append(
            f"{condition_id}: runner_exceptions={integ['runner_exceptions']}"
        )
    if integ["completed"] != 15:
        errors.append(
            f"{condition_id}: expected 15 completed episodes, got {integ['completed']}"
        )
    if integ["Q_violations"]:
        errors.append(f"{condition_id}: Q violations={integ['Q_violations']}")
    if integ["m_violations"]:
        errors.append(f"{condition_id}: m violations={integ['m_violations']}")
    if integ["hidden_exposure"]:
        errors.append(f"{condition_id}: hidden exposure={integ['hidden_exposure']}")
    if integ["raw_proxy_exposure"]:
        errors.append(f"{condition_id}: raw proxy exposure={integ['raw_proxy_exposure']}")
    if integ["non_reference_backed"]:
        errors.append(
            f"{condition_id}: provenance violations={integ['non_reference_backed']}"
        )
    if condition_id.startswith("A1") and integ["post_freeze_adaptation"]:
        errors.append(
            f"{condition_id}: post-freeze adaptation={integ['post_freeze_adaptation']}"
        )
    if condition_id.startswith("A3"):
        if not integ["reflection_timing_ok"]:
            errors.append(f"{condition_id}: reflection timing failed")
        if not integ["local_repair_reflection_pin_ok"]:
            errors.append(f"{condition_id}: local repair reflection pin failed")
        if not integ["stable_action_slot_mapping_ok"]:
            errors.append(f"{condition_id}: stable action-slot mapping failed")
    requested = set(integ["requested_models"])
    if requested and requested != {MODEL_PRO}:
        errors.append(f"{condition_id}: requested models {sorted(requested)} != {{Pro}}")
    if condition_id in {"A1-Pro", "A3-Pro"}:
        returned = [str(m) for m in (integ["returned_models"] or [])]
        if not returned:
            errors.append(
                f"{condition_id}: returned_models empty; live API model identity missing"
            )
        else:
            incompatible = [
                m for m in returned if not returned_model_identifies_as_pro(m)
            ]
            if incompatible:
                errors.append(
                    f"{condition_id}: returned models not Pro-compatible: "
                    f"{incompatible}"
                )
    if not integ["catalogue_all_abstract_proxies"]:
        errors.append(f"{condition_id}: abstract proxy catalogue incomplete")
    return errors


def resolve_anchors(*, n: int) -> list[str]:
    frozen = json.loads(FROZEN_ANCHORS_SOURCE.read_text(encoding="utf-8"))
    all_ids = [str(x) for x in frozen["anchor_ids"]]
    if len(all_ids) != 100:
        raise SystemExit(f"Expected 100 frozen anchors, got {len(all_ids)}.")
    selected = select_integration_smoke_anchors(
        all_ids, n=n, seed=EXPERIMENT_SEED
    )
    if selected[:15] != ANCHORS_15:
        raise SystemExit("Leading 15 anchors drifted from the repaired same-arena set.")
    if n >= 25 and selected[:25] != ANCHORS_25:
        raise SystemExit("Leading 25 anchors drifted from the N=25 same-arena set.")
    prior_cfg = json.loads((PRIOR_ARENA_SMOKE / "smoke_config.json").read_text())
    if [str(x) for x in prior_cfg["selected_anchor_ids"]] != ANCHORS_25:
        raise SystemExit("Prior N=25 arena anchors do not match the frozen ranking.")
    repair_cfg = json.loads((REPAIR_N15_SMOKE / "a1_v4_3_smoke_config.json").read_text())
    if [str(x) for x in repair_cfg["selected_anchor_ids"]] != ANCHORS_15:
        raise SystemExit("Repaired N=15 anchors do not match the leading subset.")
    return selected


def verify_same_arena(
    anchors: Sequence[str],
    provider: ReferencePoolProvider,
    defender,
    raw_path: Path,
) -> None:
    prior_arena = json.loads((PRIOR_ARENA_SMOKE / "arena_precompute.json").read_text())
    for aid in anchors:
        if aid not in prior_arena:
            continue
        pool = provider.get_pool(str(aid), seed=REFERENCE_POOL_SEED)
        assert_reference_pool_fingerprint(
            observed=pool.pool_fingerprint,
            expected=prior_arena[aid]["pool_fingerprint"],
            anchor_id=str(aid),
        )
        starting = load_starting_case(
            aid,
            raw_path=raw_path,
            defender=defender,
            artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
        )
        expected_fp = prior_arena[aid].get("feature_fingerprint")
        if expected_fp and feature_fingerprint(starting.features) != expected_fp:
            raise SystemExit(f"Feature fingerprint mismatch for anchor {aid}")


def run_condition(
    *,
    run_dir: Path,
    condition_id: str,
    attacker_kind: str,
    llm_model: str | None,
    anchors: Sequence[str],
    policy: CompiledGovernancePolicy,
    defender: FrozenXGBoostDefender,
    provider: ReferencePoolProvider,
    raw_path: Path,
    thinking_disabled: bool = True,
    reasoning_effort: str | None = None,
    expected_pool_fingerprints: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    if llm_model is not None:
        require_supported_llm_model(llm_model)
    if attacker_kind in {"a1", "a3"}:
        assert_thinking_cell_config(
            thinking_disabled=thinking_disabled,
            reasoning_effort=reasoning_effort,
            expect_thinking_disabled=thinking_disabled,
        )
    rows: list[dict[str, Any]] = []
    cond_dir = run_dir / condition_id
    cond_dir.mkdir(parents=True, exist_ok=True)
    # Fail closed before any API call: construct attacker on the first anchor pool.
    first_pool = provider.get_pool(str(anchors[0]), seed=REFERENCE_POOL_SEED)
    if expected_pool_fingerprints and str(anchors[0]) in expected_pool_fingerprints:
        assert_reference_pool_fingerprint(
            observed=first_pool.pool_fingerprint,
            expected=expected_pool_fingerprints[str(anchors[0])],
            anchor_id=str(anchors[0]),
        )
    probe = build_attacker(
        condition_id=condition_id,
        attacker_kind=attacker_kind,
        llm_model=llm_model,
        pool=first_pool,
        recorder=None,
        thinking_disabled=thinking_disabled,
        reasoning_effort=reasoning_effort,
    )
    manifest = attacker_condition_manifest(
        condition_id=condition_id,
        attacker_kind=attacker_kind,
        llm_model=llm_model,
        attacker=probe,
        thinking_disabled=(
            thinking_disabled if attacker_kind in {"a1", "a3"} else None
        ),
        reasoning_effort=(
            reasoning_effort if attacker_kind in {"a1", "a3"} else None
        ),
    )
    write_json(cond_dir / "condition_manifest.json", manifest)
    for anchor_id in anchors:
        episode_dir = cond_dir / f"anchor_{anchor_id}" / f"seed_{EXPERIMENT_SEED}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        print(f"[{utc_now()}] {condition_id} anchor={anchor_id}", flush=True)
        try:
            row = run_episode(
                condition_id=condition_id,
                attacker_kind=attacker_kind,
                llm_model=llm_model,
                anchor_id=str(anchor_id),
                policy=policy,
                defender=defender,
                provider=provider,
                episode_dir=episode_dir,
                raw_path=raw_path,
                thinking_disabled=thinking_disabled,
                reasoning_effort=reasoning_effort,
                expected_pool_fingerprint=(
                    expected_pool_fingerprints.get(str(anchor_id))
                    if expected_pool_fingerprints
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "condition_id": condition_id,
                "attacker_kind": attacker_kind,
                "llm_model": llm_model,
                "anchor_id": str(anchor_id),
                "success": False,
                "stop_reason": "runner_exception",
                "runner_exception": True,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "attempts_to_success": None,
                # Q/catalogue were not evaluated for this incomplete episode.
                "Q_violations": None,
                "catalogue_has_all_abstract_proxies": None,
                "m_violations": None,
                "hidden_exposure": None,
                "raw_proxy_exposure": None,
                "non_reference_backed": None,
                "post_freeze_adaptation": None,
                "month7_opened": False,
            }
            write_json(episode_dir / "episode_error.json", row)
            print(row["error"], flush=True)
        rows.append(row)
        with (run_dir / "episode_index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
    summary = {
        "condition_id": condition_id,
        "attacker_kind": attacker_kind,
        "attacker_version": manifest.get("attacker_version"),
        "llm_model": llm_model,
        "prompt_version": (
            PINNED_A1_PROMPT_VERSION
            if attacker_kind == "a1"
            else PINNED_A3_PROMPT_VERSION
            if attacker_kind == "a3"
            else None
        ),
        "prompt_hash": manifest.get("prompt_hash"),
        "gower_policy": PINNED_A2_GOWER_POLICY if attacker_kind == "a2" else None,
        "model": manifest.get("model"),
        "thinking_disabled": manifest.get("thinking_disabled"),
        "thinking_enabled": manifest.get("thinking_enabled"),
        "reasoning_effort": manifest.get("reasoning_effort"),
        "config_hash": manifest.get("config_hash"),
        "condition_manifest": manifest,
        "n": len(rows),
        "success": f"{sum(1 for r in rows if r.get('success'))}/{len(rows)}",
        "asr_curve": asr_curve(rows),
        "first_success": first_success_summary(rows),
        "integrity": condition_integrity(rows),
        "month7_opened": False,
        "status": "development_model_selection_not_findings",
    }
    write_json(cond_dir / "condition_summary.json", summary)
    return rows


def paired_anchor_table(by_condition: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    anchors = [str(r["anchor_id"]) for r in next(iter(by_condition.values()))]
    table = []
    for aid in anchors:
        row: dict[str, Any] = {"anchor_id": aid}
        for cid, rows in by_condition.items():
            match = next((r for r in rows if str(r["anchor_id"]) == aid), None)
            if match is None:
                row[cid] = None
                continue
            row[cid] = {
                "success": bool(match.get("success")),
                "attempts_to_success": match.get("attempts_to_success"),
                "stop_reason": match.get("stop_reason"),
            }
        table.append(row)
    return table


def flash_vs_pro(flash_rows: Sequence[Mapping[str, Any]], pro_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paired = []
    both = 0
    flash_only = 0
    pro_only = 0
    neither = 0
    for f_row, p_row in zip(flash_rows, pro_rows, strict=True):
        if str(f_row["anchor_id"]) != str(p_row["anchor_id"]):
            raise SystemExit("Flash/Pro anchor order mismatch.")
        fs = bool(f_row.get("success"))
        ps = bool(p_row.get("success"))
        if fs and ps:
            both += 1
        elif fs:
            flash_only += 1
        elif ps:
            pro_only += 1
        else:
            neither += 1
        paired.append(
            {
                "anchor_id": str(f_row["anchor_id"]),
                "flash_success": fs,
                "pro_success": ps,
                "flash_q": f_row.get("attempts_to_success"),
                "pro_q": p_row.get("attempts_to_success"),
            }
        )
    return {
        "n": len(paired),
        "both_success": both,
        "flash_only": flash_only,
        "pro_only": pro_only,
        "neither": neither,
        "flash_asr_curve": asr_curve(flash_rows),
        "pro_asr_curve": asr_curve(pro_rows),
        "paired": paired,
        "automatic_winner": None,
        "note": "Researcher review only; no automatic model selection.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("sanity", "benchmark", "all"),
        default="all",
        help="sanity=A1/A3-Pro N=15; benchmark=six conditions N=50; all=sanity then benchmark.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to frozen BAF Base.csv (default: DEFAULT_RAW_PATH).",
    )
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        raise SystemExit(f"Raw dataset missing: {raw_path}")
    policy = CompiledGovernancePolicy.load(GOVERNANCE_PATH)
    missing_proxies = {
        "name_email_alignment",
        "home_phone_configuration",
        "mobile_phone_configuration",
    } - set(policy.available_action_keys)
    if missing_proxies:
        raise SystemExit(f"Governance available actions missing abstract proxies: {sorted(missing_proxies)}")
    defender = FrozenXGBoostDefender.from_artefact_dir(DEFAULT_C1_ARTEFACT_DIR)
    # Fail closed on pin / D1 / governance / Month-7 before any API calls.
    preflight_defence_and_pins(
        policy=policy,
        defender=defender,
        artefact_dir=DEFAULT_C1_ARTEFACT_DIR,
    )
    provider = ReferencePoolProvider.from_config(
        build_pool_config(), raw_path=raw_path
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"dev_model_selection_{args.phase}_m{M_MAX}_Q{Q_MAX}_seed{EXPERIMENT_SEED}_{stamp}",
        parent=ROOT / "05_outputs" / "scratch" / "smoke",
        stage="scratch",
    )
    write_json(
        run_dir / "run_config.json",
        {
            "status": "development_model_selection_not_findings",
            "phase": args.phase,
            "created_utc": utc_now(),
            "K": K,
            "Q": Q_MAX,
            "m": M_MAX,
            "experiment_seed": EXPERIMENT_SEED,
            "reference_pool_seed": REFERENCE_POOL_SEED,
            "pins": pinned_attacker_summary(),
            "require_reference_provenance": PINNED_REQUIRE_REFERENCE_PROVENANCE,
            "month7_opened": False,
            "d1_artefact_dir": str(DEFAULT_C1_ARTEFACT_DIR),
            "d1_artefact_id": defender.artefact_id,
            "raw_path": str(raw_path),
            "governance_fingerprint": policy.policy_fingerprint,
            "frozen_anchors_source": str(FROZEN_ANCHORS_SOURCE),
            "prior_arena_smoke": str(PRIOR_ARENA_SMOKE),
            "repair_n15_smoke": str(REPAIR_N15_SMOKE),
            "note": (
                "Development model-selection evidence only. "
                "Not a Month-7 result and not a dissertation final finding."
            ),
        },
    )

    sanity_rows: dict[str, list[dict[str, Any]]] = {}
    if args.phase in {"sanity", "all"}:
        anchors15 = resolve_anchors(n=15)
        verify_same_arena(anchors15, provider, defender, raw_path)
        write_json(run_dir / "sanity_anchors.json", {"anchor_ids": anchors15, "N": 15})
        for cid, kind, model in (
            ("A1-Pro", "a1", MODEL_PRO),
            ("A3-Pro", "a3", MODEL_PRO),
        ):
            sanity_rows[cid] = run_condition(
                run_dir=run_dir / "sanity",
                condition_id=cid,
                attacker_kind=kind,
                llm_model=model,
                anchors=anchors15,
                policy=policy,
                defender=defender,
                provider=provider,
                raw_path=raw_path,
            )
        gate_errors: list[str] = []
        for cid, rows in sanity_rows.items():
            gate_errors.extend(sanity_gates(cid, rows))
        sanity_report = {
            "status": "SANITY_FAIL" if gate_errors else "SANITY_PASS",
            "errors": gate_errors,
            "conditions": {
                cid: {
                    "success": f"{sum(1 for r in rows if r.get('success'))}/{len(rows)}",
                    "asr_curve": asr_curve(rows),
                    "integrity": condition_integrity(rows),
                }
                for cid, rows in sanity_rows.items()
            },
            "month7_opened": False,
            "note": "Backend/schema/mechanics validation only; not findings.",
        }
        write_json(run_dir / "SANITY_REPORT.json", sanity_report)
        print(json.dumps(sanity_report, indent=2, sort_keys=True), flush=True)
        if gate_errors:
            print("SANITY FAILED. Stopping before N=50.", flush=True)
            return 2
        if args.phase == "sanity":
            return 0

    anchors50 = resolve_anchors(n=50)
    verify_same_arena(anchors50, provider, defender, raw_path)
    write_json(run_dir / "benchmark_anchors.json", {"anchor_ids": anchors50, "N": 50})
    conditions = (
        ("A0", "a0", None),
        ("A1-Flash", "a1", MODEL_FLASH),
        ("A1-Pro", "a1", MODEL_PRO),
        ("A2", "a2", None),
        ("A3-Flash", "a3", MODEL_FLASH),
        ("A3-Pro", "a3", MODEL_PRO),
    )
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for cid, kind, model in conditions:
        by_condition[cid] = run_condition(
            run_dir=run_dir / "benchmark",
            condition_id=cid,
            attacker_kind=kind,
            llm_model=model,
            anchors=anchors50,
            policy=policy,
            defender=defender,
            provider=provider,
            raw_path=raw_path,
        )
    report = {
        "status": "development_model_selection_N50_not_findings",
        "month7_opened": False,
        "automatic_winner": None,
        "N": 50,
        "K": K,
        "Q": Q_MAX,
        "m": M_MAX,
        "anchors": anchors50,
        "pins": pinned_attacker_summary(),
        "asr_curves": {cid: asr_curve(rows) for cid, rows in by_condition.items()},
        "first_success": {
            cid: first_success_summary(rows) for cid, rows in by_condition.items()
        },
        "paired_anchor_outcomes": paired_anchor_table(by_condition),
        "A1_Flash_vs_Pro": flash_vs_pro(
            by_condition["A1-Flash"], by_condition["A1-Pro"]
        ),
        "A3_Flash_vs_Pro": flash_vs_pro(
            by_condition["A3-Flash"], by_condition["A3-Pro"]
        ),
        "integrity": {
            cid: condition_integrity(rows) for cid, rows in by_condition.items()
        },
        "note": (
            "Development model-selection evidence only. "
            "Do not treat as Month-7 or dissertation findings. "
            "No automatic backend winner is selected."
        ),
    }
    write_json(run_dir / "DEV_MODEL_SELECTION_N50_REPORT.json", report)
    print(json.dumps({k: report[k] for k in report if k != "paired_anchor_outcomes"}, indent=2, sort_keys=True))
    print(f"Wrote {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
