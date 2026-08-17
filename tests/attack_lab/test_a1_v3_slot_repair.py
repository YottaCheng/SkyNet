"""Focused contract tests for A1 V3 pre-freeze local slot repair."""

from __future__ import annotations

import json

import pytest
from attack_lab.attacker_interface import AttackerEpisode
from attack_lab.attackers.a1_planner import (
    OneShotLLMPlanner,
    PROMPT_VERSION_V3,
    build_a1_prompt_payload,
    render_a1_messages,
)
from attack_lab.budget import AttackBudget
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
from attack_lab.reference_actions import ReferenceSelection
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import PublicFeedback
from test_a1_planner import (
    CountingBlockDefender,
    ScriptedLLMClient,
    _make_env,
    _qm_budget,
    _valid_plan_response,
)


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    ).get_pool(starting_case.case_id)


def _refs(pool, case, action: str, count: int = 5) -> list[str]:
    """Return distinct resolved values so the plan has distinct fingerprints."""
    result, values = [], set()
    for profile in pool.profiles:
        value = profile.fields.get(action)
        if value == case.features[action] or repr(value) in values:
            continue
        values.add(repr(value))
        result.append(profile.profile_id)
        if len(result) == count:
            break
    assert len(result) >= count
    return result


def _plan(pool, case, *, action: str = "income") -> list[dict]:
    return [
        {
            "strategy_label": f"{action}_{index}",
            "changes": {action: {"reference_id": reference_id}},
        }
        for index, reference_id in enumerate(_refs(pool, case, action), start=1)
    ]


def _env(starting_case, governance_policy, reference_pool, tmp_path):
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case, governance_policy=governance_policy,
        reference_pool=reference_pool, tmp_path=tmp_path, budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session", "payment_type", "home_phone_configuration"),
        defender=defender,
    )
    return env, defender


def _attacker(reference_pool, client):
    return OneShotLLMPlanner(
        experiment_seed=13, reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2), prompt_version=PROMPT_VERSION_V3,
        llm_client=client,
    )


def test_v3_prompt_uses_action_key_vocabulary_only(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    payload = build_a1_prompt_payload(
        env=env, reference_pool=reference_pool, budget=AttackBudget(q_max=5, m_max=2),
        q_max=5, prompt_version=PROMPT_VERSION_V3,
    )
    text = json.dumps(payload) + "\n" + "\n".join(
        item["content"] for item in render_a1_messages(payload)
    )
    assert "home_phone_configuration" in payload["allowed_action_keys"]
    for raw_name in ("phone_home_valid", "phone_mobile_valid", "name_email_similarity"):
        assert raw_name not in text
    assert all("feature" not in item for item in payload["action_catalogue"])


def test_v3_wrong_count_repairs_slots_without_q_or_d1(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    plan = _plan(reference_pool, starting_case)
    client = ScriptedLLMClient([
        json.dumps({"candidates": plan[:2]}),
        json.dumps({"replacements": [
            {"candidate_index": i, **item} for i, item in enumerate(plan, start=1)
        ]}),
    ])
    frozen = _attacker(reference_pool, client).prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert env.ledger.q_remaining == 5 and defender.calls == 0
    assert "slot_repair" in client.calls[1]["messages"][1]["content"]


def test_v3_repairs_only_invalid_slots(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    plan = _plan(reference_pool, starting_case)
    bad = [dict(item) for item in plan]
    bad[1] = {"strategy_label": "bad_action", "changes": {"phone_home_valid": {"reference_id": "ref_01"}}}
    bad[3] = {"strategy_label": "bad_ref", "changes": {"income": {"reference_id": "ref_99"}}}
    client = ScriptedLLMClient([
        json.dumps({"candidates": bad}),
        json.dumps({"replacements": [
            {"candidate_index": 2, **plan[1]},
            {"candidate_index": 4, **plan[3]},
        ]}),
    ])
    attacker = _attacker(reference_pool, client)
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5 and defender.calls == 0 and env.ledger.q_remaining == 5
    assert [dict(frozen[i].changes) for i in (0, 2, 4)] == [
        {key: ReferenceSelection(value["reference_id"]) for key, value in plan[i]["changes"].items()}
        for i in (0, 2, 4)
    ]
    assert attacker.call_record and attacker.call_record.local_repair_count == 1


def test_v3_multiround_exhaustion_and_literal_are_local_only(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    plan = _plan(reference_pool, starting_case)
    bad = [dict(item) for item in plan]
    bad[0] = {"strategy_label": "literal", "changes": {"income": 0.5}}
    still_bad = {"replacements": [{"candidate_index": 1, "strategy_label": "still_literal", "changes": {"income": 0.4}}]}
    client = ScriptedLLMClient([json.dumps({"candidates": bad}), json.dumps(still_bad), json.dumps(still_bad)])
    attacker = _attacker(reference_pool, client)
    assert attacker.prepare_frozen_sequence(env) == ()
    assert attacker._pending_stop_reason == "local_generation_exhausted"  # noqa: SLF001
    assert len(client.calls) == 3 and env.ledger.q_remaining == 5 and defender.calls == 0


def test_v3_frozen_reference_selections_never_replan(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    client = ScriptedLLMClient([json.dumps({"candidates": _plan(reference_pool, starting_case)})])
    attacker = _attacker(reference_pool, client)
    frozen = attacker.prepare_frozen_sequence(env)
    assert all(isinstance(value, ReferenceSelection) for item in frozen for value in item.changes.values())
    proposal = attacker.propose(env)
    assert proposal is not None
    env.step(proposal)
    assert defender.calls == 1
    attacker.propose(env)
    assert len(client.calls) == 1


def test_v3_multiround_repair_preserves_fixed_slots(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """First repair fixes one slot; second repair fixes the remaining invalid slot."""
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    plan = _plan(reference_pool, starting_case)
    bad = [dict(item) for item in plan]
    bad[1] = {
        "strategy_label": "bad_action",
        "changes": {"phone_home_valid": {"reference_id": "ref_01"}},
    }
    bad[3] = {
        "strategy_label": "bad_ref",
        "changes": {"income": {"reference_id": "ref_99"}},
    }
    # Round 1 repair: fix only slot 2; leave slot 4 bad.
    repair1 = {
        "replacements": [
            {"candidate_index": 2, **plan[1]},
            {
                "candidate_index": 4,
                "strategy_label": "still_bad",
                "changes": {"income": {"reference_id": "ref_99"}},
            },
        ]
    }
    repair2 = {
        "replacements": [
            {"candidate_index": 4, **plan[3]},
        ]
    }
    client = ScriptedLLMClient(
        [json.dumps({"candidates": bad}), json.dumps(repair1), json.dumps(repair2)]
    )
    attacker = _attacker(reference_pool, client)
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert len(client.calls) == 3
    assert attacker.call_record is not None
    assert attacker.call_record.local_repair_count == 2
    assert env.ledger.q_remaining == 5
    assert defender.calls == 0
    # Slots 1/3/5 unchanged across both repairs.
    for index in (0, 2, 4):
        assert dict(frozen[index].changes) == {
            key: ReferenceSelection(value["reference_id"])
            for key, value in plan[index]["changes"].items()
        }
    # Second repair prompt asked only for remaining invalid index 4.
    repair_prompt = client.calls[2]["messages"][1]["content"]
    assert "Repair indices: [4]" in repair_prompt
    assert "invalid_candidate_indices" in repair_prompt


def test_v3_static_lock_inconsistency_is_local_only(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Episode-static inconsistency is repaired locally; no D1 / Q / ref substitution."""
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
        defender=defender,
    )
    income_refs = _refs(reference_pool, starting_case, "income", count=5)
    age_refs = _refs(reference_pool, starting_case, "customer_age", count=2)
    # Candidate 1 omits customer_age (locks to anchor); candidate 2 changes it.
    bad = [
        {
            "strategy_label": f"income_{i}",
            "changes": {"income": {"reference_id": income_refs[i]}},
        }
        for i in range(5)
    ]
    bad[1] = {
        "strategy_label": "age_break",
        "changes": {"customer_age": {"reference_id": age_refs[0]}},
    }
    good_slot2 = {
        "strategy_label": "income_1",
        "changes": {"income": {"reference_id": income_refs[1]}},
    }
    client = ScriptedLLMClient(
        [
            json.dumps({"candidates": bad}),
            json.dumps(
                {"replacements": [{"candidate_index": 2, **good_slot2}]}
            ),
        ]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=13,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V3,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.ledger.q_remaining == 5
    assert attacker.call_record is not None
    assert attacker.call_record.retry_ledger[0].invalid_candidate_indices == (2,)
    # Code must not invent a substitute reference_id in preserved/frozen changes.
    for proposal in frozen:
        for value in proposal.changes.values():
            assert isinstance(value, ReferenceSelection)
            assert value.reference_id in {
                profile.profile_id for profile in reference_pool.profiles
            }


def test_v3_attacker_episode_orchestrator_prefreeze_accounting(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Regression: V3 prepare uses AttackerEpisode accounting, not env.defender."""

    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "facade_env",
        budget=_qm_budget(5, 2),
        enabled=enabled,
        defender=defender,
    )
    facade = AttackerEpisode(env)
    assert not hasattr(facade, "defender")

    response = _valid_plan_response(
        starting_case,
        reference_pool,
        prompt_version=PROMPT_VERSION_V3,
        enabled=enabled,
        governance_policy=governance_policy,
        q_max=5,
    )
    client = ScriptedLLMClient(responses=[response])
    attacker = OneShotLLMPlanner(
        experiment_seed=17,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V3,
        llm_client=client,
    )

    # 1–3: prepare on the real facade must not raise; Q and step/D1 stay at 0.
    frozen = attacker.prepare_frozen_sequence(facade)
    assert len(frozen) == 5
    q_used = int(facade.budget.q_max) - int(facade.ledger.q_remaining)
    assert q_used == 0
    assert int(facade.attempts_used) == 0
    assert defender.calls == 0
    assert attacker.call_record is not None
    assert attacker.call_record.q_used_before_freeze == 0
    assert attacker.call_record.d1_calls_before_freeze == 0
    llm_at_freeze = len(client.calls)
    assert llm_at_freeze >= 1

    # 4: first env.step increments the normal public accounting.
    step = facade.step(frozen[0])
    assert step.validity.is_valid
    assert int(facade.attempts_used) == 1
    assert int(facade.budget.q_max) - int(facade.ledger.q_remaining) == 1
    assert defender.calls == 1

    # 5: PASS/BLOCK/INVALID feedback must not trigger post-freeze LLM replanning.
    facade_env_feedback = PublicFeedback(
        label="BLOCK",
        message="poison",
        attempt=int(facade.attempts_used),
        remaining_attempts=int(facade.ledger.q_remaining),
        q_remaining=int(facade.ledger.q_remaining),
        m_max=int(facade.ledger.m_max),
    )
    env._last_feedback = facade_env_feedback  # noqa: SLF001
    nxt = attacker.propose(facade)
    assert nxt is not None
    assert len(client.calls) == llm_at_freeze

    # MatchOrchestrator path (same facade type) must also complete without AttributeError.
    match_defender = CountingBlockDefender()
    match_logger = TrajectoryLogger(run_dir=tmp_path / "match", run_id="a1_v3")
    match_logger.run_dir.mkdir(parents=True, exist_ok=True)
    match_client = ScriptedLLMClient(responses=[response])
    match_attacker = OneShotLLMPlanner(
        experiment_seed=17,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V3,
        llm_client=match_client,
    )
    match = MatchOrchestrator().run_episode(
        match_attacker,
        MatchConfig(
            attacker_id="a1",
            anchor=starting_case,
            policy=governance_policy,
            budget=_qm_budget(5, 2),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=match_defender,
            seed=17,
            enabled_action_keys=enabled,
            logger=match_logger,
            reference_pool=reference_pool,
            require_reference_provenance=True,
        ),
    )
    assert match.attacker_id == "a1"
    assert match_attacker.call_record is not None
    assert match_attacker.call_record.q_used_before_freeze == 0
    assert match_attacker.call_record.d1_calls_before_freeze == 0
    assert match.q_used == match.scored_defender_queries
    assert match.q_used >= 1
    assert match_defender.calls == match.scored_defender_queries
    assert len(match_client.calls) == match_attacker.call_record.llm_call_count
