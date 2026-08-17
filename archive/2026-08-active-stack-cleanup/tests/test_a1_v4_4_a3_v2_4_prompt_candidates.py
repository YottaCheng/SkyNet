"""Minimal tests for final A1 V4.4 / A3 V2.4 prompt candidates and thinking wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack_lab.attackers.a1_planner import (
    A1ModelConfig,
    DeepSeekPlannerClient,
    PROMPT_VERSION_V4_3,
    PROMPT_VERSION_V4_4,
    SUPPORTED_PROMPT_VERSIONS,
    render_a1_messages,
)
from attack_lab.archive.contracts.a1_v4_1_contract import build_v4_1_action_slots
from attack_lab.attackers.a1_v4_3_contract import build_v4_3_prompt_payload
from attack_lab.attackers.a1_v4_4_contract import (
    FINAL_INTERNAL_VALIDATION_V4_4,
    FORBIDDEN_INFORMATION_BOUNDARY_V4_4,
    NON_ADAPTIVE_ONESHOT_PLANNING_V4_4,
    ROLE_AND_OBJECTIVE_V4_4,
    assert_v4_4_prompt_hard_contract,
    build_v4_4_prompt_payload,
)
from attack_lab.archive.contracts.a1_v4_contract import (
    PROXY_RAW_FEATURE_NAMES,
    build_v4_choice_catalog,
    build_v4_static_plan_options,
)
from attack_lab.attackers.a3_agent import A3ModelConfig, PROMPT_VERSION_A3_V2_4
from attack_lab.archive.contracts.a3_v2_1_contract import (
    public_slot_entries,
    writable_slots_from_episode_map,
)
from attack_lab.attackers.a3_v2_3_contract import (
    MAX_HYPOTHESIS_CHARS_V2_3,
    PROMPT_VERSION_A3_V2_3,
    build_a3_v2_3_episode_action_slots,
    build_a3_v2_3_prompt_payload,
)
from attack_lab.attackers.a3_v2_4_contract import (
    ADAPTATION_NOTE_LIMIT_V2_4,
    ADAPTIVE_EPISODIC_REASONING_V2_4,
    MAX_HYPOTHESIS_CHARS_V2_4,
    ROLE_AND_OBJECTIVE_V2_4,
    assert_a3_v2_4_prompt_hard_contract,
    build_a3_v2_4_episode_action_slots,
    build_a3_v2_4_prompt_payload,
    render_a3_v2_4_messages,
)
from attack_lab.budget import AttackBudget
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.validator import ConstraintValidator
from test_a1_planner import CountingBlockDefender, _qm_budget


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    ).get_pool(starting_case.case_id)


def _make_env(*, starting_case, governance_policy, reference_pool, tmp_path: Path):
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    return AttackEnvironment(
        starting_case=starting_case,
        defender=CountingBlockDefender(),
        validator=ConstraintValidator.from_policy(
            governance_policy,
            enabled_action_keys=tuple(governance_policy.available_action_keys),
            reference_pool=reference_pool,
            require_reference_provenance=True,
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=_qm_budget(5, 2),
    )


def test_v4_4_version_supported_and_persists(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    assert PROMPT_VERSION_V4_4 in SUPPORTED_PROMPT_VERSIONS
    assert PROMPT_VERSION_V4_4 != PROMPT_VERSION_V4_3
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "v44",
    )
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
    slots = build_v4_1_action_slots(catalog)
    payload = build_v4_4_prompt_payload(
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
    assert payload["prompt_version"] == PROMPT_VERSION_V4_4
    assert_v4_4_prompt_hard_contract(payload, pool=reference_pool)


def test_v4_4_contains_objective_and_nonadaptivity(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "v44b",
    )
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
    payload = build_v4_4_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=env.starting_case.case_id,
        catalog=catalog,
        static_plans=plans,
        action_slots=build_v4_1_action_slots(catalog),
    )
    assert payload["role_and_objective"] == ROLE_AND_OBJECTIVE_V4_4
    assert payload["forbidden_information_boundary"] == (
        FORBIDDEN_INFORMATION_BOUNDARY_V4_4
    )
    assert payload["final_internal_validation"] == FINAL_INTERNAL_VALIDATION_V4_4
    assert payload["non_adaptive_oneshot_planning"] == (
        NON_ADAPTIVE_ONESHOT_PLANNING_V4_4
    )
    assert "maximise the probability" in payload["role_and_objective"]
    assert "CRITICAL NON-ADAPTIVITY RULE" in payload["non_adaptive_oneshot_planning"]
    assert "SHAP values" in payload["forbidden_information_boundary"]
    rendered = "\n".join(m["content"] for m in render_a1_messages(payload))
    assert "maximise the probability" in rendered
    assert "frozen" in rendered.lower()


def test_v4_4_schema_and_public_reference_unchanged_vs_v4_3(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "cmp",
    )
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
    slots = build_v4_1_action_slots(catalog)
    kwargs = dict(
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
    v44 = build_v4_4_prompt_payload(**kwargs)
    v43 = build_v4_3_prompt_payload(**kwargs)
    assert v44["output_schema"] == v43["output_schema"]
    assert v44["choice_catalogue"] == v43["choice_catalogue"]
    assert v44["action_slots"] == v43["action_slots"]
    assert v44["public_reference_profiles"] == v43["public_reference_profiles"]
    text = json.dumps(v44)
    for raw in PROXY_RAW_FEATURE_NAMES:
        assert f'"{raw}"' not in text


def test_v2_4_prompt_objective_limit_and_schema(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    assert MAX_HYPOTHESIS_CHARS_V2_4 == MAX_HYPOTHESIS_CHARS_V2_3 == 512
    assert ADAPTATION_NOTE_LIMIT_V2_4 == "adaptation_note must be <= 512 characters."
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "a3",
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
    )
    slots = build_a3_v2_4_episode_action_slots(catalog, validator=env.validator)
    writable = writable_slots_from_episode_map(
        slots, validator=env.validator, include_static=True
    )
    payload = build_a3_v2_4_prompt_payload(
        case_id=env.starting_case.case_id,
        visible_anchor=env.validator.visible_fields(env.starting_case.features),
        current_application=env.observation().visible_fields,
        budget=AttackBudget(q_max=5, m_max=2),
        q_remaining=5,
        query_index=1,
        static_edit_cost=0,
        residual_m=2,
        locked_static_values={},
        slots=writable,
        slot_entries=public_slot_entries(writable, validator=env.validator),
        episodic_memory=[],
        pool=reference_pool,
        catalog=catalog,
        episode_slot_map=public_slot_entries(slots, validator=env.validator),
    )
    assert payload["prompt_version"] == PROMPT_VERSION_A3_V2_4
    assert PROMPT_VERSION_A3_V2_4 != PROMPT_VERSION_A3_V2_3
    assert_a3_v2_4_prompt_hard_contract(payload, pool=reference_pool)
    assert payload["role_and_objective"] == ROLE_AND_OBJECTIVE_V2_4
    assert payload["adaptive_episodic_adversarial_reasoning"] == (
        ADAPTIVE_EPISODIC_REASONING_V2_4
    )
    assert "SHAP values" in payload["forbidden_information_boundary"]
    assert f"<= {MAX_HYPOTHESIS_CHARS_V2_4} characters" in payload[
        "adaptation_note_limit"
    ]

    v23_slots = build_a3_v2_3_episode_action_slots(catalog, validator=env.validator)
    v23_writable = writable_slots_from_episode_map(
        v23_slots, validator=env.validator, include_static=True
    )
    v23 = build_a3_v2_3_prompt_payload(
        case_id=env.starting_case.case_id,
        visible_anchor=env.validator.visible_fields(env.starting_case.features),
        current_application=env.observation().visible_fields,
        budget=AttackBudget(q_max=5, m_max=2),
        q_remaining=5,
        query_index=1,
        static_edit_cost=0,
        residual_m=2,
        locked_static_values={},
        slots=v23_writable,
        slot_entries=public_slot_entries(v23_writable, validator=env.validator),
        episodic_memory=[],
        pool=reference_pool,
        catalog=catalog,
        episode_slot_map=public_slot_entries(v23_slots, validator=env.validator),
    )
    assert payload["output_schema"] == v23["output_schema"]
    assert payload["choice_catalogue"] == v23["choice_catalogue"]
    assert payload["public_reference_profiles"] == v23["public_reference_profiles"]
    rendered = "\n".join(m["content"] for m in render_a3_v2_4_messages(payload))
    assert ROLE_AND_OBJECTIVE_V2_4 in rendered
    assert "adaptation_note must be <= 512 characters" in rendered
    for raw in PROXY_RAW_FEATURE_NAMES:
        assert f'"{raw}"' not in json.dumps(payload["public_reference_profiles"])


def test_thinking_config_hashes_and_explicit_extra_body(monkeypatch) -> None:
    off = A1ModelConfig(
        model="deepseek-v4-pro",
        thinking_disabled=True,
        prompt_version=PROMPT_VERSION_V4_4,
    )
    on = A1ModelConfig(
        model="deepseek-v4-pro",
        thinking_disabled=False,
        reasoning_effort="max",
        prompt_version=PROMPT_VERSION_V4_4,
    )
    assert "reasoning_effort" not in off.to_dict()
    assert on.to_dict()["reasoning_effort"] == "max"
    assert off.config_hash() != on.config_hash()

    a3_off = A3ModelConfig(
        model="deepseek-v4-pro",
        thinking_disabled=True,
        prompt_version=PROMPT_VERSION_A3_V2_4,
    )
    a3_on = A3ModelConfig(
        model="deepseek-v4-pro",
        thinking_disabled=False,
        reasoning_effort="max",
        prompt_version=PROMPT_VERSION_A3_V2_4,
    )
    assert "reasoning_effort" not in a3_off.to_dict()
    assert a3_on.to_dict()["reasoning_effort"] == "max"

    captured: dict = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class _Msg:
                content = '{"ok": true}'

            class _Choice:
                message = _Msg()

            class _Usage:
                prompt_tokens = 1
                completion_tokens = 1
                total_tokens = 2
                prompt_tokens_details = None

            class _Resp:
                choices = [_Choice()]
                usage = _Usage()
                model = "deepseek-v4-pro"
                system_fingerprint = "fp_test"

            return _Resp()

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr("openai.OpenAI", _FakeClient)
    client = DeepSeekPlannerClient(api_key="x", base_url="http://example.invalid")
    client.complete(
        [{"role": "user", "content": "hi"}],
        model="deepseek-v4-pro",
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        timeout_seconds=5.0,
        thinking_disabled=True,
    )
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in captured

    captured.clear()
    out = client.complete(
        [{"role": "user", "content": "hi"}],
        model="deepseek-v4-pro",
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        timeout_seconds=5.0,
        thinking_disabled=False,
        reasoning_effort="max",
    )
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}
    assert captured["reasoning_effort"] == "max"
    assert out.thinking_disabled is False
    assert out.reasoning_effort == "max"
    assert out.requested_model == "deepseek-v4-pro"
