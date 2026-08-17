"""A1 V4.1 unique-action-slot regression tests (no DeepSeek)."""

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
    SUPPORTED_PROMPT_VERSIONS,
)
from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_contract import (
    build_v4_choice_catalog,
    build_v4_prompt_payload,
    build_v4_static_plan_options,
    resolve_choice_ids_to_changes,
)
from attack_lab.archive.contracts.a1_v4_1_contract import (
    GENERIC_UNAVAILABLE_NOTICE,
    build_v4_1_action_slots,
    build_v4_1_prompt_payload,
    classify_attacker_visible_term_context,
    parse_a1_v4_1_plan,
    resolve_action_slot_selections,
    scan_attacker_visible_hidden_mentions,
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


def _same_action_choice_pair(catalog, plan):
    by_action: dict[str, list[str]] = {}
    for choice_id in plan.allowed_query_choice_ids:
        choice = catalog.get(choice_id)
        assert choice is not None
        by_action.setdefault(choice.action_key, []).append(choice_id)
    for action_key, ids in by_action.items():
        if len(ids) >= 2:
            return action_key, ids[0], ids[1]
    raise AssertionError("need two legal choices for the same underlying action")


def _slot_for_action(slots, action_key: str):
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        if slot.action_key == action_key:
            return slot
    raise AssertionError(f"no action slot for {action_key}")


def _selection_portfolio(plan, slots, catalog, q_max: int = 5) -> list[dict]:
    """Build q_max unique single-slot selections under residual_m."""
    used_choices: set[str] = set()
    candidates: list[dict] = []
    for choice_id in plan.allowed_query_choice_ids:
        if choice_id in used_choices:
            continue
        choice = catalog.get(choice_id)
        assert choice is not None
        slot = _slot_for_action(slots, choice.action_key)
        candidates.append(
            {
                "strategy_label": f"c{len(candidates)+1}",
                "selections": {slot.action_slot_id: choice_id},
            }
        )
        used_choices.add(choice_id)
        if len(candidates) == q_max:
            break
    assert len(candidates) == q_max
    return candidates


def test_v4_1_version_selectable_and_v4_frozen() -> None:
    assert PROMPT_VERSION_V4_1 == "a1_oneshot_v4_1_unique_action_slots"
    assert PROMPT_VERSION_V4_1 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V4 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V4 == "a1_oneshot_v4_hard_contract"
    for version in (
        PROMPT_VERSION_V1,
        PROMPT_VERSION_V2,
        PROMPT_VERSION_V3,
        PROMPT_VERSION_V4,
        PROMPT_VERSION_V4_1,
    ):
        assert version in SUPPORTED_PROMPT_VERSIONS


def test_v4_smoke_leak_flags_are_prohibition_wording_only() -> None:
    """Read-only classification of the completed V4 smoke prompt artefacts."""
    smoke = (
        "/Users/ziyaoch/ucl/dissertation/05_outputs/archive/smoke/"
        "a1_v4_k10_integration_smoke_N25_m2_Q5_seed1_20260811T221937Z"
    )
    prompt_path = (
        f"{smoke}/episodes/anchor_795826/a1/seed_1/a1_prompt_full.txt"
    )
    text = open(prompt_path, encoding="utf-8").read()
    assert '"proxy_raw_targets_forbidden"' in text
    assert '"explicitly_unavailable"' in text
    for term in (
        "name_email_similarity",
        "phone_home_valid",
        "phone_mobile_valid",
        "risk_score",
        "feature_importance",
        "shap",
    ):
        assert term in text
        assert classify_attacker_visible_term_context(text, term).startswith("C_")
    # Not actual hidden values / researcher-only dumps.
    assert "RESEARCHER_ONLY" not in text


def test_v4_1_prompt_removes_named_hidden_denylist(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    payload = build_v4_1_prompt_payload(
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
    assert payload["prompt_version"] == PROMPT_VERSION_V4_1
    assert payload["unavailable_information"] == GENERIC_UNAVAILABLE_NOTICE
    assert "explicitly_unavailable" not in payload
    assert "proxy_raw_targets_forbidden" not in payload.get("hard_contract", {})
    blob = json.dumps(payload)
    for term in PROXY_RAW_FEATURE_NAMES:
        assert term not in blob
    for term in ("risk_score", "feature_importance", "shap", "d1_threshold"):
        assert term not in blob
    assert scan_attacker_visible_hidden_mentions(blob) == []


def test_v4_old_style_same_action_two_choices_still_rejected_by_v4_resolver(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Document the V4 mechanical failure mode that V4.1 hardens."""
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, _slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0)
    _action, choice_a, choice_b = _same_action_choice_pair(catalog, plan)
    changes, reason = resolve_choice_ids_to_changes([choice_a, choice_b], catalog)
    assert changes is None
    assert reason == "duplicate_action_in_slot"


def test_v4_1_structurally_prevents_same_action_duplicate_and_repairs(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0 and p.residual_m >= 2)
    action_key, choice_a, choice_b = _same_action_choice_pair(catalog, plan)
    slot = _slot_for_action(slots, action_key)

    # Old V4 conceptual failure is schema-invalid under V4.1 (choice_ids forbidden).
    old_style = {
        "static_plan_id": plan.static_plan_id,
        "candidates": [
            {"strategy_label": "bad", "choice_ids": [choice_a, choice_b]},
            *[{"strategy_label": f"x{i}", "choice_ids": [choice_a]} for i in range(4)],
        ],
    }
    parsed, status = parse_a1_v4_1_plan(
        json.dumps(old_style),
        q_max=5,
        allowed_static_plan_ids=[p.static_plan_id for p in plans],
    )
    assert parsed is None
    assert status == "forbidden_output_key:choice_ids"

    # Even object-form cannot place two same-action choices: one slot key only.
    # Using a second slot with a choice from the first action fails pairing.
    other_slot = next(
        slots.get(sid)
        for sid in slots.ordered_slot_ids
        if slots.get(sid) is not None and slots.get(sid).action_key != action_key
    )
    assert other_slot is not None
    bad_pair, bad_reason = resolve_action_slot_selections(
        {slot.action_slot_id: choice_a, other_slot.action_slot_id: choice_b},
        slots=slots,
        catalog=catalog,
    )
    assert bad_pair is None
    assert bad_reason == "choice_not_in_action_slot"

    # Live local repair path: first portfolio uses an invalid second slot pairing
    # on candidate 1; repair uses two DIFFERENT action slots and freezes.
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    # Force candidate 1 invalid via wrong slot/choice pairing.
    bad_candidates = [dict(item) for item in good]
    bad_candidates[0] = {
        "strategy_label": "dup_action_attempt",
        "selections": {
            slot.action_slot_id: choice_a,
            other_slot.action_slot_id: choice_b,
        },
    }
    # Ensure residual allows 2 selections.
    assert plan.residual_m >= 2

    # Repair candidate 1 with two different legal action slots.
    second_action = other_slot.action_key
    second_choice = next(
        cid
        for cid in plan.allowed_query_choice_ids
        if catalog.get(cid) is not None
        and catalog.get(cid).action_key == second_action
        and cid != choice_a
    )
    repair = {
        "replacements": [
            {
                "candidate_index": 1,
                "strategy_label": "fixed_two_slots",
                "selections": {
                    slot.action_slot_id: choice_a,
                    other_slot.action_slot_id: second_choice,
                },
            }
        ]
    }
    client = ScriptedLLMClient(
        [
            json.dumps(
                {"static_plan_id": plan.static_plan_id, "candidates": bad_candidates}
            ),
            json.dumps(repair),
        ]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=41,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_1,
        llm_client=client,
    )
    q_before = env.ledger.q_remaining
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.attempts_used == 0
    assert env.ledger.q_remaining == q_before == 5
    assert attacker.call_record is not None
    assert attacker.call_record.q_used_before_freeze == 0
    assert attacker.call_record.d1_calls_before_freeze == 0
    assert attacker.call_record.local_repair_count >= 1
    assert attacker.call_record.retry_ledger[0].invalid_candidate_indices == (1,)
    assert (
        attacker.call_record.governance_reject_counts.get("choice_not_in_action_slot", 0)
        >= 1
    )
    # No automatic substitute: frozen candidate 1 matches the repaired selections.
    assert frozen[0].research_meta["selections"] == {
        slot.action_slot_id: choice_a,
        other_slot.action_slot_id: second_choice,
    }
    # Valid slots preserved.
    for idx in range(1, 5):
        assert frozen[idx].research_meta["selections"] == good[idx]["selections"]


def test_v4_1_resolve_unknown_choice_id_unit() -> None:
    from attack_lab.archive.contracts.a1_v4_1_contract import ActionSlot, ActionSlotCatalog
    from attack_lab.archive.contracts.a1_v4_contract import LegalChoice, V4ChoiceCatalog

    catalog = V4ChoiceCatalog(
        choices_by_id={
            "choice_001": LegalChoice(
                choice_id="choice_001",
                action_key="income",
                reference_id="ref_01",
                category="per_attempt",
            )
        },
        static_choice_ids=(),
        per_attempt_choice_ids=("choice_001",),
    )
    slots = ActionSlotCatalog(
        slots_by_id={
            "action_slot_01": ActionSlot(
                action_slot_id="action_slot_01",
                action_key="income",
                allowed_choice_ids=("choice_001", "choice_999"),
            )
        },
        ordered_slot_ids=("action_slot_01",),
    )
    changes, reason = resolve_action_slot_selections(
        {"action_slot_01": "choice_999"}, slots=slots, catalog=catalog
    )
    assert changes is None
    assert reason == "unknown_choice_id"


def test_v4_1_unknown_ids_and_residual_and_static_and_provenance(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    slot0 = next(iter(good[0]["selections"]))
    # unknown action_slot_id
    bad = [dict(x) for x in good]
    bad[0] = {
        "strategy_label": "u",
        "selections": {"action_slot_999": next(iter(good[0]["selections"].values()))},
    }
    # Fix unknown slot
    repair1 = {
        "replacements": [
            {
                "candidate_index": 1,
                "strategy_label": "ok",
                "selections": good[0]["selections"],
            }
        ]
    }
    # After first freeze attempt would succeed with repair1; also test unknown choice
    # in a separate episode below.
    client = ScriptedLLMClient(
        [
            json.dumps({"static_plan_id": plan.static_plan_id, "candidates": bad}),
            json.dumps(repair1),
        ]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=42,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_1,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.attempts_used == 0
    assert attacker.call_record.governance_reject_counts.get("unknown_action_slot_id") == 1
    for proposal in frozen:
        assert proposal.research_meta["prompt_version"] == PROMPT_VERSION_V4_1
        for action_key, selection in proposal.changes.items():
            assert action_key not in PROXY_RAW_FEATURE_NAMES
            assert selection.reference_id.startswith("ref_")

    # unknown choice_id
    env2, defender2 = _env(
        starting_case, governance_policy, reference_pool, tmp_path / "u2"
    )
    bad2 = [dict(x) for x in good]
    bad2[1] = {"strategy_label": "u2", "selections": {slot0: "choice_999"}}
    repair2 = {
        "replacements": [
            {
                "candidate_index": 2,
                "strategy_label": "ok2",
                "selections": good[1]["selections"],
            }
        ]
    }
    attacker2 = OneShotLLMPlanner(
        experiment_seed=43,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_1,
        llm_client=ScriptedLLMClient(
            [
                json.dumps({"static_plan_id": plan.static_plan_id, "candidates": bad2}),
                json.dumps(repair2),
            ]
        ),
    )
    frozen2 = attacker2.prepare_frozen_sequence(env2)
    assert len(frozen2) == 5
    assert defender2.calls == 0
    rejects = attacker2.call_record.governance_reject_counts
    assert rejects.get("choice_not_in_action_slot", 0) >= 1

    # residual_m cannot be exceeded
    env3, defender3 = _env(
        starting_case, governance_policy, reference_pool, tmp_path / "u3"
    )
    plan1 = next(p for p in plans if p.static_edit_cost == 1 and p.residual_m == 1)
    over = _selection_portfolio(plan1, slots, catalog, q_max=5)
    # force 2 selections on candidate 1 while residual_m=1
    a_key, c_a, _c_b = _same_action_choice_pair(catalog, plan1)
    slot_a = _slot_for_action(slots, a_key)
    other = next(
        slots.get(sid)
        for sid in slots.ordered_slot_ids
        if slots.get(sid).action_key != a_key
        and any(
            catalog.get(cid).action_key == slots.get(sid).action_key
            for cid in plan1.allowed_query_choice_ids
        )
    )
    c_other = next(
        cid
        for cid in plan1.allowed_query_choice_ids
        if catalog.get(cid).action_key == other.action_key
    )
    over[0] = {
        "strategy_label": "over",
        "selections": {slot_a.action_slot_id: c_a, other.action_slot_id: c_other},
    }
    fix = {
        "replacements": [
            {
                "candidate_index": 1,
                "strategy_label": "one",
                "selections": {slot_a.action_slot_id: c_a},
            }
        ]
    }
    attacker3 = OneShotLLMPlanner(
        experiment_seed=44,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_1,
        llm_client=ScriptedLLMClient(
            [
                json.dumps({"static_plan_id": plan1.static_plan_id, "candidates": over}),
                json.dumps(fix),
            ]
        ),
    )
    frozen3 = attacker3.prepare_frozen_sequence(env3)
    assert len(frozen3) == 5
    assert defender3.calls == 0
    assert attacker3.call_record.governance_reject_counts.get("budget_exceeded") == 1


def test_v4_1_cross_candidate_duplicate_still_repaired(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans, slots = _catalog_plans_slots(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0)
    good = _selection_portfolio(plan, slots, catalog, q_max=5)
    initial = [dict(x) for x in good]
    initial[2] = {
        "strategy_label": "dup",
        "selections": dict(good[0]["selections"]),
    }
    used = {next(iter(c["selections"].values())) for c in good}
    unused = next(cid for cid in plan.allowed_query_choice_ids if cid not in used)
    choice = catalog.get(unused)
    assert choice is not None
    slot = _slot_for_action(slots, choice.action_key)
    repair = {
        "replacements": [
            {
                "candidate_index": 3,
                "strategy_label": "fixed",
                "selections": {slot.action_slot_id: unused},
            }
        ]
    }
    attacker = OneShotLLMPlanner(
        experiment_seed=45,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_1,
        llm_client=ScriptedLLMClient(
            [
                json.dumps({"static_plan_id": plan.static_plan_id, "candidates": initial}),
                json.dumps(repair),
            ]
        ),
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert attacker._v4_selected_static_plan_id == plan.static_plan_id  # noqa: SLF001
    assert attacker.call_record.retry_ledger[0].invalid_candidate_indices == (3,)
    assert frozen[0].research_meta["selections"] == good[0]["selections"]
    assert frozen[2].research_meta["selections"] == {slot.action_slot_id: unused}


def test_v4_1_freeze_wall_no_replan_after_block(
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
        experiment_seed=46,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4_1,
        llm_client=client,
    )
    facade = AttackerEpisode(env)
    frozen = attacker.prepare_frozen_sequence(facade)
    assert len(frozen) == 5
    assert int(facade.attempts_used) == 0
    llm_at_freeze = len(client.calls)
    fps = [p.research_meta["candidate_fingerprint"] for p in frozen]
    static_id = attacker._v4_selected_static_plan_id  # noqa: SLF001

    facade.step(frozen[0])
    assert int(facade.attempts_used) == 1
    assert defender.calls == 1
    env._last_feedback = PublicFeedback(  # noqa: SLF001
        label="BLOCK",
        message="block1",
        attempt=1,
        remaining_attempts=4,
        q_remaining=4,
        m_max=2,
    )
    nxt = attacker.propose(facade)
    assert nxt is not None
    assert len(client.calls) == llm_at_freeze
    assert attacker._v4_selected_static_plan_id == static_id  # noqa: SLF001
    assert [
        p.research_meta["candidate_fingerprint"] for p in attacker.frozen_proposals
    ] == fps


def test_v4_prompt_payload_unchanged_version_string(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Frozen V4 builder still emits a1_oneshot_v4_hard_contract."""
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
    )
    plans = build_v4_static_plan_options(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
        catalog=catalog,
        m_max=2,
        q_max=5,
    )
    payload = build_v4_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=str(env.starting_case.case_id),
        catalog=catalog,
        static_plans=plans,
    )
    assert payload["prompt_version"] == "a1_oneshot_v4_hard_contract"
    assert "proxy_raw_targets_forbidden" in payload["hard_contract"]
