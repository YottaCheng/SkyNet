"""A1 V4.2 bounded unique-action-slot regression tests (no DeepSeek)."""

from __future__ import annotations

import json

import pytest
from attack_lab.attacker_interface import AttackerEpisode
from attack_lab.attackers.a1_planner import (
    OneShotLLMPlanner,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V3,
    PROMPT_VERSION_V4,
    PROMPT_VERSION_V4_1,
    PROMPT_VERSION_V4_2,
    SUPPORTED_PROMPT_VERSIONS,
)
from attack_lab.archive.contracts.a1_v4_contract import (
    build_v4_choice_catalog,
    build_v4_static_plan_options,
)
from attack_lab.archive.contracts.a1_v4_1_contract import build_v4_1_action_slots
from attack_lab.archive.contracts.a1_v4_2_contract import (
    build_v4_2_prompt_payload,
    build_v4_2_repair_output_schema,
    parse_a1_v4_2_plan,
    parse_a1_v4_2_slot_replacements,
)
from attack_lab.budget import AttackBudget
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import PublicFeedback
from test_a1_planner import (
    CountingBlockDefender,
    ScriptedLLMClient,
    _make_env,
    _qm_budget,
)


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    ).get_pool(starting_case.case_id)


def _env(starting_case, governance_policy, reference_pool, tmp_path, enabled=None):
    defender = CountingBlockDefender()
    enabled = enabled or (
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
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
        defender=defender,
    )
    return env, defender


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


def _slot_for_action(slots, action_key: str):
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        if slot.action_key == action_key:
            return slot
    raise AssertionError(f"no action slot for {action_key}")


def _distinct_slot_choices(plan, slots, catalog, n: int) -> list[tuple[str, str]]:
    """Return n (slot_id, choice_id) pairs with distinct action slots."""
    out: list[tuple[str, str]] = []
    used_actions: set[str] = set()
    for choice_id in plan.allowed_query_choice_ids:
        choice = catalog.get(choice_id)
        assert choice is not None
        if choice.action_key in used_actions:
            continue
        slot = _slot_for_action(slots, choice.action_key)
        out.append((slot.action_slot_id, choice_id))
        used_actions.add(choice.action_key)
        if len(out) == n:
            return out
    raise AssertionError(f"need {n} distinct action slots")


def _selection_portfolio(plan, slots, catalog, q_max: int = 5) -> list[dict]:
    pairs = []
    used: set[str] = set()
    for choice_id in plan.allowed_query_choice_ids:
        if choice_id in used:
            continue
        choice = catalog.get(choice_id)
        assert choice is not None
        slot = _slot_for_action(slots, choice.action_key)
        pairs.append(
            {
                "strategy_label": f"c{len(pairs)+1}",
                "selections": {slot.action_slot_id: choice_id},
            }
        )
        used.add(choice_id)
        if len(pairs) == q_max:
            return pairs
    raise AssertionError("insufficient distinct choices for portfolio")


def test_v4_2_version_selectable() -> None:
    assert PROMPT_VERSION_V4_2 == "a1_oneshot_v4_2_bounded_unique_action_slots"
    for version in (
        PROMPT_VERSION_V1,
        PROMPT_VERSION_V2,
        PROMPT_VERSION_V3,
        PROMPT_VERSION_V4,
        PROMPT_VERSION_V4_1,
        PROMPT_VERSION_V4_2,
    ):
        assert version in SUPPORTED_PROMPT_VERSIONS


def test_v4_2_schema_encodes_per_plan_max_properties(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    payload = build_v4_2_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=str(env.starting_case.case_id),
        catalog=catalog,
        static_plans=plans,
        action_slots=slots,
    )
    assert payload["prompt_version"] == PROMPT_VERSION_V4_2
    branches = payload["output_schema"]["oneOf"]
    assert len(branches) == len(plans)
    by_id = {
        branch["properties"]["static_plan_id"]["const"]: branch for branch in branches
    }
    for plan in plans:
        branch = by_id[plan.static_plan_id]
        selections = branch["properties"]["candidates"]["items"]["properties"][
            "selections"
        ]
        assert selections["minProperties"] == 1
        assert selections["maxProperties"] == int(plan.residual_m)


def test_case_a_over_cardinality_uses_slot_repair_not_full_regen(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0 and p.residual_m == 2)
    triple = _distinct_slot_choices(plan, slots, catalog, 3)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    # Candidate #2 (1-based index 2) exceeds residual_m; others legal.
    bad = [dict(x) for x in good]
    bad[1] = {
        "strategy_label": "too_many",
        "selections": {slot: choice for slot, choice in triple},
    }
    parsed, status = parse_a1_v4_2_plan(
        json.dumps({"static_plan_id": plan.static_plan_id, "candidates": bad}),
        q_max=5,
        static_plans=plans,
    )
    assert status == "ok" and parsed is not None
    assert len(parsed["candidates"][1]["selections"]) == 3

    repair = {
        "replacements": [
            {
                "candidate_index": 2,
                "strategy_label": "fixed",
                "selections": good[1]["selections"],
            }
        ]
    }
    client = ScriptedLLMClient(
        [
            json.dumps({"static_plan_id": plan.static_plan_id, "candidates": bad}),
            json.dumps(repair),
        ]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=51,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_2,
        llm_client=client,
    )
    q_before = env.ledger.q_remaining
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.attempts_used == 0
    assert env.ledger.q_remaining == q_before == 5
    assert attacker.call_record.q_used_before_freeze == 0
    assert attacker.call_record.d1_calls_before_freeze == 0
    assert attacker.call_record.local_repair_count == 1
    assert attacker.call_record.retry_ledger[0].invalid_candidate_indices == (2,)
    assert (
        attacker.call_record.governance_reject_counts.get(
            "selection_count_exceeds_residual_m", 0
        )
        >= 1
    )
    # Valid slots pinned; only #2 repaired.
    assert frozen[0].research_meta["selections"] == good[0]["selections"]
    assert frozen[1].research_meta["selections"] == good[1]["selections"]
    assert frozen[2].research_meta["selections"] == good[2]["selections"]
    assert frozen[3].research_meta["selections"] == good[3]["selections"]
    assert frozen[4].research_meta["selections"] == good[4]["selections"]
    assert all(len(p.research_meta["selections"]) <= 2 for p in frozen)
    # Second LLM call must be slot repair, not a full-plan regeneration.
    assert client.calls[1]["messages"][1]["content"].count("'replacements'") >= 0
    assert "replacements" in client.calls[1]["messages"][1]["content"]
    assert "LOCAL RULE-COMPLIANCE SLOT REPAIR" in client.calls[1]["messages"][1]["content"]


def test_case_b_two_distinct_slots_legal(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.residual_m == 2 and p.static_edit_cost == 0)
    pair = _distinct_slot_choices(plan, slots, catalog, 2)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    good[0] = {
        "strategy_label": "two",
        "selections": {pair[0][0]: pair[0][1], pair[1][0]: pair[1][1]},
    }
    # Ensure uniqueness vs others
    fps_seed = {json.dumps(c["selections"], sort_keys=True) for c in good[1:]}
    assert json.dumps(good[0]["selections"], sort_keys=True) not in fps_seed
    client = ScriptedLLMClient(
        [json.dumps({"static_plan_id": plan.static_plan_id, "candidates": good})]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=52,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_2,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert len(frozen[0].research_meta["selections"]) == 2


def test_case_c_residual_1_rejects_two_and_repair_schema_max_1(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.residual_m == 1)
    pair = _distinct_slot_choices(plan, slots, catalog, 2)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    bad = [dict(x) for x in good]
    bad[0] = {
        "strategy_label": "over",
        "selections": {pair[0][0]: pair[0][1], pair[1][0]: pair[1][1]},
    }
    parsed, status = parse_a1_v4_2_plan(
        json.dumps({"static_plan_id": plan.static_plan_id, "candidates": bad}),
        q_max=5,
        static_plans=plans,
    )
    assert status == "ok" and parsed is not None

    schema = build_v4_2_repair_output_schema(
        slot_enum=list(slots.ordered_slot_ids),
        residual_m=1,
        requested_indices=[1],
    )
    assert (
        schema["properties"]["replacements"]["items"]["properties"]["selections"][
            "maxProperties"
        ]
        == 1
    )

    repair = {
        "replacements": [
            {
                "candidate_index": 1,
                "strategy_label": "one",
                "selections": good[0]["selections"],
            }
        ]
    }
    attacker = OneShotLLMPlanner(
        experiment_seed=53,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_2,
        llm_client=ScriptedLLMClient(
            [
                json.dumps({"static_plan_id": plan.static_plan_id, "candidates": bad}),
                json.dumps(repair),
            ]
        ),
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.attempts_used == 0
    assert attacker.call_record.retry_ledger[0].invalid_candidate_indices == (1,)
    assert (
        attacker.call_record.governance_reject_counts.get(
            "selection_count_exceeds_residual_m", 0
        )
        >= 1
    )


def test_case_d_residual_1_single_selection_legal(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.residual_m == 1)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    assert all(len(c["selections"]) == 1 for c in good)
    attacker = OneShotLLMPlanner(
        experiment_seed=54,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_2,
        llm_client=ScriptedLLMClient(
            [json.dumps({"static_plan_id": plan.static_plan_id, "candidates": good})]
        ),
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0


def test_case_e_wrong_slot_choice_still_rejected(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0 and p.residual_m >= 2)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    # Find two choices for same action (V4.1 failure mode) and wrong-slot pairing
    by_action: dict[str, list[str]] = {}
    for cid in plan.allowed_query_choice_ids:
        choice = catalog.get(cid)
        by_action.setdefault(choice.action_key, []).append(cid)
    action, ids = next((a, v) for a, v in by_action.items() if len(v) >= 2)
    slot = _slot_for_action(slots, action)
    other = next(
        slots.get(sid)
        for sid in slots.ordered_slot_ids
        if slots.get(sid).action_key != action
    )
    bad = [dict(x) for x in good]
    bad[0] = {
        "strategy_label": "mismatch",
        "selections": {slot.action_slot_id: ids[0], other.action_slot_id: ids[1]},
    }
    fix = {
        "replacements": [
            {
                "candidate_index": 1,
                "strategy_label": "ok",
                "selections": good[0]["selections"],
            }
        ]
    }
    attacker = OneShotLLMPlanner(
        experiment_seed=55,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_2,
        llm_client=ScriptedLLMClient(
            [
                json.dumps({"static_plan_id": plan.static_plan_id, "candidates": bad}),
                json.dumps(fix),
            ]
        ),
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert (
        attacker.call_record.governance_reject_counts.get("choice_not_in_action_slot", 0)
        >= 1
    )


def test_case_f_repair_over_residual_rejected(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    # Force cross-candidate duplicate so slot repair engages
    initial = [dict(x) for x in good]
    initial[2] = {
        "strategy_label": "dup",
        "selections": dict(good[0]["selections"]),
    }
    triple = _distinct_slot_choices(plan, slots, catalog, 3)
    bad_repair = {
        "replacements": [
            {
                "candidate_index": 3,
                "strategy_label": "too_many",
                "selections": {s: c for s, c in triple},
            }
        ]
    }
    parsed, status = parse_a1_v4_2_slot_replacements(
        json.dumps(bad_repair), requested_indices=[3], residual_m=2
    )
    assert parsed is None
    assert status == "selection_count_exceeds_residual_m"

    used = {next(iter(c["selections"].values())) for c in good}
    unused = next(cid for cid in plan.allowed_query_choice_ids if cid not in used)
    choice = catalog.get(unused)
    slot = _slot_for_action(slots, choice.action_key)
    good_repair = {
        "replacements": [
            {
                "candidate_index": 3,
                "strategy_label": "fixed",
                "selections": {slot.action_slot_id: unused},
            }
        ]
    }
    attacker = OneShotLLMPlanner(
        experiment_seed=56,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_2,
        llm_client=ScriptedLLMClient(
            [
                json.dumps({"static_plan_id": plan.static_plan_id, "candidates": initial}),
                json.dumps(bad_repair),
                json.dumps(good_repair),
            ]
        ),
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.attempts_used == 0
    assert env.ledger.q_remaining == 5
    # Only candidate 3 was the repair target on first validation failure
    assert attacker.call_record.retry_ledger[0].invalid_candidate_indices == (3,)


def test_case_g_no_llm_after_freeze_block_pass(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(
        starting_case, governance_policy, reference_pool, tmp_path / "wall"
    )
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0)
    response = json.dumps(
        {
            "static_plan_id": plan.static_plan_id,
            "candidates": _selection_portfolio(plan, slots, catalog, q_max=5),
        }
    )
    client = ScriptedLLMClient([response])
    attacker = OneShotLLMPlanner(
        experiment_seed=57,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_2,
        llm_client=client,
    )
    facade = AttackerEpisode(env)
    frozen = attacker.prepare_frozen_sequence(facade)
    assert len(frozen) == 5
    llm_at_freeze = len(client.calls)
    fps = [p.research_meta["candidate_fingerprint"] for p in frozen]
    facade.step(frozen[0])
    env._last_feedback = PublicFeedback(  # noqa: SLF001
        label="BLOCK",
        message="block1",
        attempt=1,
        remaining_attempts=4,
        q_remaining=4,
        m_max=2,
    )
    assert attacker.propose(facade) is not None
    env._last_feedback = PublicFeedback(  # noqa: SLF001
        label="PASS",
        message="pass",
        attempt=2,
        remaining_attempts=3,
        q_remaining=3,
        m_max=2,
    )
    assert len(client.calls) == llm_at_freeze
    assert [
        p.research_meta["candidate_fingerprint"] for p in attacker.frozen_proposals
    ] == fps


def test_v4_2_system_prompt_forbids_reference_id_emission(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    from attack_lab.attackers.a1_planner import render_a1_messages

    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    payload = build_v4_2_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=str(env.starting_case.case_id),
        catalog=catalog,
        static_plans=plans,
        action_slots=slots,
    )
    messages = render_a1_messages(payload)
    system = messages[0]["content"]
    assert "action_slot_id -> choice_id" in system
    assert "Never emit raw values, action_key, or reference_id" in system
    assert "Every change must cite a reference_id" not in system


def test_historical_system_prompt_still_cites_reference_id() -> None:
    from attack_lab.attackers.a1_planner import render_a1_messages

    for version in (
        PROMPT_VERSION_V1,
        PROMPT_VERSION_V2,
        PROMPT_VERSION_V3,
        PROMPT_VERSION_V4,
        PROMPT_VERSION_V4_1,
    ):
        messages = render_a1_messages({"prompt_version": version, "budget": {"q_max": 5}})
        assert "Every change must cite a reference_id from the provided pool" in messages[
            0
        ]["content"]
