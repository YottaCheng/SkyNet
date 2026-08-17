"""Public reference interface correction tests (A1 V4.3 / A2 v2 / A3 V2.3)."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker
from attack_lab.attackers.a1_planner import (
    OneShotLLMPlanner,
    PROMPT_VERSION_V4_2,
    PROMPT_VERSION_V4_3,
    SUPPORTED_PROMPT_VERSIONS,
    render_a1_messages,
)
from attack_lab.archive.contracts.a1_v4_1_contract import build_v4_1_action_slots
from attack_lab.archive.contracts.a1_v4_2_contract import (
    PROMPT_VERSION_V4_2 as CONTRACT_V4_2,
    build_v4_2_prompt_payload,
)
from attack_lab.attackers.a1_v4_3_contract import (
    PROMPT_VERSION_V4_3 as CONTRACT_V4_3,
    build_v4_3_prompt_payload,
)
from attack_lab.archive.contracts.a1_v4_contract import (
    build_v4_choice_catalog,
    build_v4_static_plan_options,
)
from attack_lab.attackers.a2_search import (
    GOWER_POLICY_LEGACY_V1,
    GOWER_POLICY_PUBLIC_REFERENCE_V2,
    SurrogateGuidedSearcher,
)
from attack_lab.attackers.a3_agent import (
    PROMPT_VERSION_A3_V2_2,
    PROMPT_VERSION_A3_V2_3,
)
from attack_lab.archive.contracts.a3_v2_2_contract import (
    PROMPT_VERSION_A3_V2_2 as CONTRACT_A3_V2_2,
    build_a3_v2_2_episode_action_slots,
    build_a3_v2_2_prompt_payload,
)
from attack_lab.attackers.a3_v2_3_contract import (
    PROMPT_VERSION_A3_V2_3 as CONTRACT_A3_V2_3,
    build_a3_v2_3_episode_action_slots,
    build_a3_v2_3_prompt_payload,
)
from attack_lab.archive.contracts.a3_v2_1_contract import (
    public_slot_entries,
    writable_slots_from_episode_map,
)
from attack_lab.budget import AttackBudget
from attack_lab.public_reference_view import (
    TRUSTED_PROXY_RAW_TARGETS,
    assert_public_reference_view_safe,
    build_canonical_public_reference_view,
    choice_public_value_lookup,
    public_safe_gower_field_names,
    public_safe_reference_field_names,
)
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from test_a1_planner import CountingBlockDefender, ScriptedLLMClient, _make_env, _qm_budget
from test_a2_search import _make_env as _make_a2_env


PROXY_RAWS = frozenset(
    {
        "name_email_similarity",
        "phone_home_valid",
        "phone_mobile_valid",
    }
)


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    ).get_pool(starting_case.case_id)


def _a1_env(starting_case, governance_policy, reference_pool, tmp_path):
    enabled = (
        "income",
        "customer_age",
        "current_address_months_count",
        "keep_alive_session",
        "payment_type",
        "device_os",
        "proposed_credit_limit",
        "employment_status",
        "housing_status",
    )
    return _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
        defender=CountingBlockDefender(),
    )


def _catalog_plans_slots(env, pool, *, m_max=2, q_max=5):
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=pool, anchor=env.starting_case.features
    )
    plans = build_v4_static_plan_options(
        validator=env.validator,
        pool=pool,
        anchor=env.starting_case.features,
        catalog=catalog,
        m_max=m_max,
        q_max=q_max,
    )
    slots = build_v4_1_action_slots(catalog)
    return catalog, plans, slots


def test_public_safe_view_deterministic_and_clean(reference_pool):
    assert TRUSTED_PROXY_RAW_TARGETS == PROXY_RAWS
    fields = public_safe_reference_field_names(reference_pool)
    assert fields == public_safe_reference_field_names(reference_pool)
    assert not (set(fields) & PROXY_RAWS)
    assert "fraud_bool" not in fields
    assert "source_row_id" not in fields

    view_a = build_canonical_public_reference_view(reference_pool)
    view_b = build_canonical_public_reference_view(reference_pool)
    assert view_a == view_b
    assert int(view_a["K"]) == int(reference_pool.K)
    assert len(view_a["profiles"]) == int(reference_pool.K)
    assert [p["profile_id"] for p in view_a["profiles"]] == [
        p.profile_id for p in reference_pool.profiles
    ]
    assert_public_reference_view_safe(view_a)
    assert "source_row_ids" not in view_a
    for profile in view_a["profiles"]:
        assert "source_row_id" not in profile
        assert "fraud_bool" not in profile["fields"]
        assert not (set(profile["fields"]) & PROXY_RAWS)
        # Legitimate public fields preserved when present on the underlying profile.
        underlying = next(
            p for p in reference_pool.profiles if p.profile_id == profile["profile_id"]
        )
        for name in fields:
            if name in underlying.fields:
                assert name in profile["fields"]
                assert profile["fields"][name] == underlying.fields[name]


def test_a2_gower_fields_equal_public_safe_intersect_action(reference_pool):
    gower = public_safe_gower_field_names(reference_pool)
    public = set(public_safe_reference_field_names(reference_pool))
    expected = tuple(
        name for name in reference_pool.action_fields if name in public
    )
    assert gower == expected
    assert not (set(gower) & PROXY_RAWS)


def test_a1_v4_3_payload_exposes_safe_k10_and_mapping(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _a1_env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    payload = build_v4_3_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=env.starting_case.case_id,
        catalog=catalog,
        static_plans=plans,
        action_slots=slots,
    )
    assert payload["prompt_version"] == CONTRACT_V4_3
    public = payload["public_reference_profiles"]
    assert int(public["K"]) == reference_pool.K
    assert len(public["profiles"]) == reference_pool.K
    assert set(public["public_safe_fields"]) == set(
        public_safe_reference_field_names(reference_pool)
    )
    blob = json.dumps(payload, sort_keys=True)
    for term in PROXY_RAWS:
        assert f'"{term}"' not in blob
    # Mapping preserved and recoverable.
    choice = payload["choice_catalogue"][0]
    recovered = choice_public_value_lookup(
        catalog_choices=payload["choice_catalogue"],
        public_view={"profiles": public["profiles"]},
        choice_id=choice["choice_id"],
    )
    assert recovered is not None
    assert choice["action_key"] not in PROXY_RAWS
    # V4.2 untouched builder still emits V4.2 only.
    v42 = build_v4_2_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=env.starting_case.case_id,
        catalog=catalog,
        static_plans=plans,
        action_slots=slots,
    )
    assert v42["prompt_version"] == CONTRACT_V4_2
    assert "public_reference_profiles" not in v42


def test_a1_v4_3_no_post_feedback_adaptation_and_v4_2_supported(
    starting_case, governance_policy, reference_pool, tmp_path
):
    assert CONTRACT_V4_3 in SUPPORTED_PROMPT_VERSIONS
    assert CONTRACT_V4_2 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V4_2 == "a1_oneshot_v4_2_bounded_unique_action_slots"
    assert PROMPT_VERSION_V4_3 == "a1_oneshot_v4_3_public_reference_view"
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "attack_lab"
        / "archive"
        / "contracts"
    )
    v42_text = (src / "a1_v4_2_contract.py").read_text(encoding="utf-8")
    assert 'PROMPT_VERSION_V4_2 = "a1_oneshot_v4_2_bounded_unique_action_slots"' in v42_text
    # One-shot path still freezes before env.step (no feedback-driven replanning).
    env = _a1_env(starting_case, governance_policy, reference_pool, tmp_path / "a1")
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = plans[0]
    # Build a minimal legal portfolio for scripted client.
    pairs = []
    used: set[str] = set()
    for choice_id in plan.allowed_query_choice_ids:
        choice = catalog.get(choice_id)
        if choice is None or choice.action_key in used:
            continue
        slot = next(
            slots.get(sid)
            for sid in slots.ordered_slot_ids
            if slots.get(sid) and slots.get(sid).action_key == choice.action_key
        )
        pairs.append((slot.action_slot_id, choice_id))
        used.add(choice.action_key)
        if len(pairs) >= min(2, max(1, plan.residual_m)):
            break
    assert pairs
    candidates = [
        {
            "strategy_label": f"c{i}",
            "selections": {pairs[i % len(pairs)][0]: pairs[i % len(pairs)][1]},
        }
        for i in range(5)
    ]
    # Ensure uniqueness by rotating secondary slots when available.
    if len(pairs) >= 2:
        for i in range(5):
            candidates[i]["selections"] = {
                pairs[i % len(pairs)][0]: pairs[i % len(pairs)][1]
            }
            if i > 0 and len(pairs) > 1:
                # Distinct fingerprints via different primary choices.
                candidates[i]["selections"] = {
                    pairs[(i + j) % len(pairs)][0]: pairs[(i + j) % len(pairs)][1]
                    for j in range(1)
                }
    # Use distinct single-slot selections from distinct choices if possible.
    distinct = []
    seen_choice: set[str] = set()
    for choice_id in plan.allowed_query_choice_ids:
        choice = catalog.get(choice_id)
        if choice is None or choice_id in seen_choice:
            continue
        slot = next(
            (
                slots.get(sid)
                for sid in slots.ordered_slot_ids
                if slots.get(sid) and slots.get(sid).action_key == choice.action_key
            ),
            None,
        )
        if slot is None:
            continue
        distinct.append((slot.action_slot_id, choice_id))
        seen_choice.add(choice_id)
        if len(distinct) >= 5:
            break
    if len(distinct) >= 5:
        candidates = [
            {
                "strategy_label": f"c{i}",
                "selections": {distinct[i][0]: distinct[i][1]},
            }
            for i in range(5)
        ]
    client = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "static_plan_id": plan.static_plan_id,
                    "candidates": candidates,
                }
            )
        ]
    )
    planner = OneShotLLMPlanner(
        budget=AttackBudget(q_max=5, m_max=2),
        reference_pool=reference_pool,
        experiment_seed=1,
        prompt_version=PROMPT_VERSION_V4_3,
        llm_client=client,
    )
    frozen = planner.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert env.attempts_used == 0
    payload = planner._prompt_payload
    assert "public_reference_profiles" in payload
    messages = render_a1_messages(payload)
    assert "public_reference_profiles" in messages[1]["content"]


def test_a3_v2_3_payload_exposes_safe_k10_and_mapping(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _a1_env(starting_case, governance_policy, reference_pool, tmp_path / "a3")
    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
    )
    episode_slots = build_a3_v2_3_episode_action_slots(
        catalog, validator=env.validator
    )
    slots = writable_slots_from_episode_map(
        episode_slots, validator=env.validator, include_static=True
    )
    slot_entries = public_slot_entries(slots, validator=env.validator)
    payload = build_a3_v2_3_prompt_payload(
        case_id=env.starting_case.case_id,
        visible_anchor=env.validator.visible_fields(env.starting_case.features),
        current_application=env.observation().visible_fields,
        budget=AttackBudget(q_max=5, m_max=2),
        q_remaining=5,
        query_index=1,
        static_edit_cost=0,
        residual_m=2,
        locked_static_values={},
        slots=slots,
        slot_entries=slot_entries,
        episodic_memory=[],
        pool=reference_pool,
        catalog=catalog,
        episode_slot_map=public_slot_entries(episode_slots, validator=env.validator),
    )
    assert payload["prompt_version"] == CONTRACT_A3_V2_3
    public = payload["public_reference_profiles"]
    assert int(public["K"]) == reference_pool.K
    assert payload["choice_to_reference_mapping"]
    choice = payload["choice_to_reference_mapping"][0]
    recovered = choice_public_value_lookup(
        catalog_choices=payload["choice_catalogue"],
        public_view={"profiles": public["profiles"]},
        choice_id=choice["choice_id"],
    )
    assert recovered is not None
    blob = json.dumps(payload, sort_keys=True)
    for term in PROXY_RAWS:
        assert f'"{term}"' not in blob
    # Slot IDs stable vs V2.2 builder.
    slots_v22 = build_a3_v2_2_episode_action_slots(catalog, validator=env.validator)
    assert episode_slots.ordered_slot_ids == slots_v22.ordered_slot_ids
    # V2.2 unchanged: no public profiles.
    v22 = build_a3_v2_2_prompt_payload(
        case_id=env.starting_case.case_id,
        visible_anchor=env.validator.visible_fields(env.starting_case.features),
        current_application=env.observation().visible_fields,
        budget=AttackBudget(q_max=5, m_max=2),
        q_remaining=5,
        query_index=1,
        static_edit_cost=0,
        residual_m=2,
        locked_static_values={},
        slots=slots,
        slot_entries=slot_entries,
        episodic_memory=[],
    )
    assert v22["prompt_version"] == CONTRACT_A3_V2_2
    assert "public_reference_profiles" not in v22
    assert PROMPT_VERSION_A3_V2_2 == "a3_episodic_reflective_v2_2_k10_bounded_cardinality"
    assert PROMPT_VERSION_A3_V2_3 == (
        "a3_episodic_reflective_v2_3_public_reference_view"
    )


def test_a2_corrected_gower_excludes_proxy_keeps_ranking(
    starting_case, governance_policy, reference_pool, tmp_path
):
    budget = AttackBudget(q_max=5, m_max=2)
    legacy = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=1,
        gower_policy=GOWER_POLICY_LEGACY_V1,
    )
    corrected = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=1,
        gower_policy=GOWER_POLICY_PUBLIC_REFERENCE_V2,
    )
    env_legacy = _make_a2_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path / "legacy",
        budget=budget,
        reference_pool=reference_pool,
    )
    env_corr = _make_a2_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path / "corr",
        budget=budget,
        reference_pool=reference_pool,
    )
    legacy._reset_episode_state(env_legacy)
    corrected._reset_episode_state(env_corr)
    expected = set(public_safe_gower_field_names(reference_pool)).intersection(
        env_corr.starting_case.features
    )
    assert set(corrected.gower_field_names) == expected
    assert corrected.proxy_raw_geometry_access is False
    assert not (set(corrected.gower_field_names) & PROXY_RAWS)
    # Historical path remains addressable and may still include proxy raws.
    assert legacy.gower_policy == GOWER_POLICY_LEGACY_V1
    assert set(legacy.gower_field_names)
    # Ranking still active on corrected path.
    remaining = corrected._enumerate_legal_unique(env_corr)
    assert remaining
    ranked = corrected._rank_candidates(remaining)
    assert ranked
    assert hasattr(ranked[0][0], "fingerprint")


def test_historical_immutability_constants_and_a0():
    assert PROMPT_VERSION_V4_2 == "a1_oneshot_v4_2_bounded_unique_action_slots"
    assert CONTRACT_A3_V2_2 == "a3_episodic_reflective_v2_2_k10_bounded_cardinality"
    src = inspect.getsource(ConstrainedRandomAttacker)
    assert "public_reference_profiles" not in src
    assert "GOWER_POLICY_PUBLIC_REFERENCE_V2" not in src
    # Corrected A2 identifier present as versioned policy, not replacing legacy.
    assert GOWER_POLICY_PUBLIC_REFERENCE_V2 == "a2_public_reference_gower_v2"
    assert GOWER_POLICY_LEGACY_V1 == "a2_legacy_full_action_gower_v1"


def test_a1_a3_public_safe_fields_equal(reference_pool):
    a1 = set(public_safe_reference_field_names(reference_pool))
    a3 = set(public_safe_reference_field_names(reference_pool))
    assert a1 == a3
    a2 = set(public_safe_gower_field_names(reference_pool))
    assert a2 == a1.intersection(reference_pool.action_fields)
