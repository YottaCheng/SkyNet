"""A1 V4 hard-contract regression tests (no DeepSeek)."""

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
    SUPPORTED_PROMPT_VERSIONS,
)
from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_contract import (
    build_v4_choice_catalog,
    build_v4_prompt_payload,
    build_v4_static_plan_options,
    parse_a1_v4_plan,
    parse_a1_v4_slot_replacements,
    static_locks_and_cost,
)
from attack_lab.budget import AttackBudget, compute_edit_metrics
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import InternalDefenceResult, PublicFeedback
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


def _catalog_and_plans(env, pool, *, m_max=2, q_max=5):
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
    return catalog, plans


def _pick_static_cost_plan(plans, cost: int):
    for plan in plans:
        if int(plan.static_edit_cost) == int(cost) and int(plan.residual_m) >= 1:
            return plan
    raise AssertionError(f"no static plan with cost={cost}")


def _distinct_query_ids(plan, n: int) -> list[str]:
    ids = list(plan.allowed_query_choice_ids)
    assert len(ids) >= n
    return ids[:n]


def test_v1_v2_v3_remain_supported() -> None:
    assert PROMPT_VERSION_V1 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V2 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V3 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V4 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V4 == "a1_oneshot_v4_hard_contract"


def test_v4_static_plan_feasibility_filtering(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans = _catalog_and_plans(env, reference_pool, m_max=2, q_max=5)
    assert plans
    costs = {int(p.static_edit_cost) for p in plans}
    residuals = {int(p.residual_m) for p in plans}
    assert 0 in costs
    assert 2 in residuals
    assert any(int(p.static_edit_cost) == 1 and int(p.residual_m) == 1 for p in plans)
    assert all(not (p.residual_m == 0 and p.static_edit_cost >= 2) for p in plans)
    for plan in plans:
        assert "ranking" not in plan.to_public_dict()
        if plan.residual_m >= 1:
            assert plan.n_distinct_feasible_candidates >= 5


def test_v4_query_slots_cannot_access_static_actions(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans = _catalog_and_plans(env, reference_pool)
    plan = _pick_static_cost_plan(plans, 1)
    static_ids = set(catalog.static_choice_ids)
    assert static_ids
    assert set(plan.allowed_query_choice_ids).isdisjoint(static_ids)
    for choice_id in plan.allowed_query_choice_ids:
        choice = catalog.get(choice_id)
        assert choice is not None
        assert choice.category == "per_attempt"
        rule = env.validator.policy.field_for_action(choice.action_key)
        assert rule is not None
        assert not rule.is_episode_locked


def test_v4_output_choice_id_only_contract() -> None:
    ok, status = parse_a1_v4_plan(
        json.dumps(
            {
                "static_plan_id": "static_plan_01",
                "candidates": [
                    {"strategy_label": "a", "choice_ids": ["choice_001"]},
                    {"strategy_label": "b", "choice_ids": ["choice_002"]},
                ],
            }
        ),
        q_max=2,
        allowed_static_plan_ids=["static_plan_01"],
    )
    assert status == "ok" and ok is not None
    bad, status = parse_a1_v4_plan(
        json.dumps(
            {
                "static_plan_id": "static_plan_01",
                "candidates": [
                    {
                        "strategy_label": "a",
                        "choice_ids": ["choice_001"],
                        "changes": {"income": {"reference_id": "ref_01"}},
                    },
                    {"strategy_label": "b", "choice_ids": ["choice_002"]},
                ],
            }
        ),
        q_max=2,
        allowed_static_plan_ids=["static_plan_01"],
    )
    assert bad is None
    assert status.startswith("forbidden_output_key")
    bad2, status2 = parse_a1_v4_slot_replacements(
        json.dumps(
            {
                "replacements": [
                    {
                        "candidate_index": 2,
                        "strategy_label": "x",
                        "choice_ids": ["choice_003"],
                        "reference_id": "ref_01",
                    }
                ]
            }
        ),
        requested_indices=[2],
    )
    assert bad2 is None
    assert "forbidden" in status2


def test_v4_prompt_hides_proxy_raw(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans = _catalog_and_plans(env, reference_pool)
    payload = build_v4_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=env.starting_case.case_id,
        catalog=catalog,
        static_plans=plans,
    )
    text = json.dumps(
        {
            "choice_catalogue": payload["choice_catalogue"],
            "anchor": payload["anchor"],
            "static_plan_options": payload["static_plan_options"],
        }
    )
    for name in PROXY_RAW_FEATURE_NAMES:
        assert name not in text


def test_v4_residual_m_blocks_two_choices_locally(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Reproduce the V3 lock+m failure: two query choices with residual_m=1."""
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    catalog, plans = _catalog_and_plans(env, reference_pool)
    plan = _pick_static_cost_plan(plans, 1)
    assert plan.residual_m == 1
    qids = _distinct_query_ids(plan, 5)
    bad = {
        "static_plan_id": plan.static_plan_id,
        "candidates": [
            {"strategy_label": "over", "choice_ids": [qids[0], qids[1]]},
            {"strategy_label": "c2", "choice_ids": [qids[1]]},
            {"strategy_label": "c3", "choice_ids": [qids[2]]},
            {"strategy_label": "c4", "choice_ids": [qids[3]]},
            {"strategy_label": "c5", "choice_ids": [qids[4]]},
        ],
    }
    good = {
        "replacements": [
            {"candidate_index": 1, "strategy_label": "fixed", "choice_ids": [qids[0]]}
        ]
    }
    client = ScriptedLLMClient([json.dumps(bad), json.dumps(good)])
    attacker = OneShotLLMPlanner(
        experiment_seed=21,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4,
        llm_client=client,
    )
    q_before = env.ledger.q_remaining
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.attempts_used == 0
    assert env.ledger.q_remaining == q_before == 5
    assert attacker.call_record is not None
    assert attacker.call_record.local_repair_count >= 1
    assert attacker.call_record.q_used_before_freeze == 0
    assert attacker.call_record.d1_calls_before_freeze == 0
    locks, cost, _ = static_locks_and_cost(
        validator=env.validator,
        anchor=env.starting_case.features,
        catalog=catalog,
        static_choice_ids=plan.static_choice_ids,
        m_max=2,
    )
    assert cost == 1 and locks is not None
    projected = env.validator.project_for_billing(
        env.starting_case.features, frozen[0], locked_values=locks
    )
    _, distance, _, _ = compute_edit_metrics(
        anchor=env.starting_case.features,
        candidate=projected,
        mutable_feature_names=env.validator.mutable_feature_names(),
        previous_candidate=None,
    )
    assert distance <= 2
    assert all(
        p.research_meta.get("static_plan_id") == plan.static_plan_id for p in frozen
    )


def test_v4_duplicate_slot_repair_preserves_valid_and_static_plan(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(starting_case, governance_policy, reference_pool, tmp_path)
    _catalog, plans = _catalog_and_plans(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0)
    qids = _distinct_query_ids(plan, 5)
    initial = {
        "static_plan_id": plan.static_plan_id,
        "candidates": [
            {"strategy_label": "A", "choice_ids": [qids[0]]},
            {"strategy_label": "B", "choice_ids": [qids[1]]},
            {"strategy_label": "dupA", "choice_ids": [qids[0]]},
            {"strategy_label": "C", "choice_ids": [qids[2]]},
            {"strategy_label": "dupB", "choice_ids": [qids[1]]},
        ],
    }
    repair = {
        "replacements": [
            {"candidate_index": 3, "strategy_label": "D", "choice_ids": [qids[3]]},
            {"candidate_index": 5, "strategy_label": "E", "choice_ids": [qids[4]]},
        ]
    }
    client = ScriptedLLMClient([json.dumps(initial), json.dumps(repair)])
    attacker = OneShotLLMPlanner(
        experiment_seed=22,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) == 5
    assert defender.calls == 0
    assert env.ledger.q_remaining == 5
    assert attacker._v4_selected_static_plan_id == plan.static_plan_id  # noqa: SLF001
    fps = [p.research_meta["candidate_fingerprint"] for p in frozen]
    assert len(set(fps)) == 5
    assert list(frozen[0].research_meta["choice_ids"]) == [qids[0]]
    assert list(frozen[1].research_meta["choice_ids"]) == [qids[1]]
    assert list(frozen[3].research_meta["choice_ids"]) == [qids[2]]
    assert list(frozen[2].research_meta["choice_ids"]) == [qids[3]]
    assert list(frozen[4].research_meta["choice_ids"]) == [qids[4]]
    assert attacker.call_record.retry_ledger[0].invalid_candidate_indices == (3, 5)


def test_v4_nonadaptive_orchestrator_freeze_wall(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, defender = _env(
        starting_case, governance_policy, reference_pool, tmp_path / "facade"
    )
    _catalog, plans = _catalog_and_plans(env, reference_pool)
    plan = next(p for p in plans if p.static_edit_cost == 0)
    qids = _distinct_query_ids(plan, 5)
    response = json.dumps(
        {
            "static_plan_id": plan.static_plan_id,
            "candidates": [
                {"strategy_label": f"c{i}", "choice_ids": [qids[i]]} for i in range(5)
            ],
        }
    )
    client = ScriptedLLMClient([response])
    attacker = OneShotLLMPlanner(
        experiment_seed=23,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4,
        llm_client=client,
    )
    facade = AttackerEpisode(env)
    assert not hasattr(facade, "defender")
    frozen = attacker.prepare_frozen_sequence(facade)
    assert len(frozen) == 5
    assert int(facade.attempts_used) == 0
    assert int(facade.budget.q_max) - int(facade.ledger.q_remaining) == 0
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

    class PassThenBlock:
        name = "pass_then_block"
        artefact_id = "test"
        threshold = 0.5
        calls = 0

        def score_application(self, features):
            self.calls += 1
            decision = "PASS" if self.calls >= 3 else "BLOCK"
            return InternalDefenceResult(
                risk_score=0.1 if decision == "PASS" else 0.9,
                threshold=self.threshold,
                decision=decision,
                runtime_ms=0.01,
                defender_name=self.name,
                artefact_id=self.artefact_id,
            )

    match_defender = PassThenBlock()
    match_logger = TrajectoryLogger(run_dir=tmp_path / "match", run_id="a1_v4")
    match_logger.run_dir.mkdir(parents=True, exist_ok=True)
    match_client = ScriptedLLMClient([response])
    match_attacker = OneShotLLMPlanner(
        experiment_seed=23,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V4,
        llm_client=match_client,
    )
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
    match = MatchOrchestrator().run_episode(
        match_attacker,
        MatchConfig(
            attacker_id="a1",
            anchor=starting_case,
            policy=governance_policy,
            budget=_qm_budget(5, 2),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=match_defender,
            seed=23,
            enabled_action_keys=enabled,
            logger=match_logger,
            reference_pool=reference_pool,
            require_reference_provenance=True,
        ),
    )
    assert match.success is True
    assert match_attacker.call_record is not None
    assert match_attacker.call_record.q_used_before_freeze == 0
    assert match_attacker.call_record.d1_calls_before_freeze == 0
    assert len(match_client.calls) == match_attacker.call_record.llm_call_count
    assert match.q_used == match.scored_defender_queries == 3


def test_historical_prompt_versions_still_constructible(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env, _ = _env(starting_case, governance_policy, reference_pool, tmp_path)
    for version in (PROMPT_VERSION_V1, PROMPT_VERSION_V2, PROMPT_VERSION_V3):
        attacker = OneShotLLMPlanner(
            experiment_seed=1,
            reference_pool=reference_pool,
            budget=AttackBudget(q_max=5, m_max=2),
            prompt_version=version,
            llm_client=ScriptedLLMClient([]),
        )
        assert attacker.prompt_version == version
