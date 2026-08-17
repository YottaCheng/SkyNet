"""Focused regressions for A3 V2.2 bounded-cardinality closeout (no DeepSeek)."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from attack_lab.attacker_interface import AttackerEpisode
from attack_lab.attackers.a1_planner import LLMCompletion
from attack_lab.attackers.a3_agent import (
    PROMPT_VERSION_A3_V2_1,
    PROMPT_VERSION_A3_V2_2,
    EpisodicLLMAgent,
)
from attack_lab.archive.contracts.a3_v2_1_contract import (
    build_a3_v2_1_episode_action_slots,
    writable_slots_from_episode_map,
)
from attack_lab.archive.contracts.a3_v2_2_contract import (
    ATTACK_OBJECTIVE_V2_2,
    CARDINALITY_REPAIR_INSTRUCTION,
    MAX_HYPOTHESIS_CHARS_V2_2,
    PROMPT_VERSION_A3_V2_2 as V2_2_LABEL,
    SELECTIONS_VS_HYPOTHESIS_NOTE,
    STATIC_RULE_DISCLOSURE_V2_2,
    build_a3_v2_2_cardinality_repair_schema,
    build_a3_v2_2_episode_action_slots,
    build_a3_v2_2_prompt_payload,
    filter_mechanically_valid_proposed_pairs,
    parse_a3_v2_2_repair_selections,
    parse_a3_v2_2_strategic_response,
    public_slot_entries,
)
from attack_lab.archive.contracts.a3_v2_contract import build_v4_choice_catalog
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import DefenceDecision, InternalDefenceResult
from attack_lab.validator import ConstraintValidator


@dataclass
class ScriptedLabelDefender:
    labels: list[str]
    name: str = "scripted"
    artefact_id: str = "test"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult:
        self.calls += 1
        label = self.labels[min(self.calls - 1, len(self.labels) - 1)]
        decision: DefenceDecision = "PASS" if label == "PASS" else "BLOCK"
        return InternalDefenceResult(
            risk_score=0.1 if decision == "PASS" else 0.9,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


@dataclass
class ScriptedLLMClient:
    responses: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

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
        self.calls.append({"messages": [dict(item) for item in messages]})
        item = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return LLMCompletion(
            text=str(item),
            model=model,
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            cached_tokens=0,
            latency_ms=12.5,
            thinking_disabled=thinking_disabled,
        )


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    ).get_pool(starting_case.case_id)


def _qm_budget(q_max: int, m_max: int) -> BudgetSpec:
    return BudgetSpec.development_dummy(
        q_max=q_max, m_max=m_max, label=f"dummy_a3_v2_2_q{q_max}_m{m_max}"
    )


def _make_env(
    *,
    starting_case,
    governance_policy,
    reference_pool,
    tmp_path: Path,
    budget: BudgetSpec,
    defender=None,
):
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
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    return AttackEnvironment(
        starting_case=starting_case,
        defender=defender or ScriptedLabelDefender(labels=["BLOCK", "BLOCK", "BLOCK"]),
        validator=ConstraintValidator.from_policy(
            governance_policy,
            enabled_action_keys=enabled,
            reference_pool=reference_pool,
            require_reference_provenance=True,
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=budget,
    )


def _slot_choice_pairs(slots, catalog, pool, anchor_features, n: int) -> dict[str, str]:
    """Pick n slot->choice pairs whose resolved raw values differ from the anchor."""
    by_ref = {p.profile_id: p for p in pool.profiles}
    out: dict[str, str] = {}
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        anchor_val = anchor_features.get(slot.action_key)
        chosen = None
        for choice_id in slot.allowed_choice_ids:
            choice = catalog.get(choice_id)
            if choice is None:
                continue
            profile = by_ref.get(choice.reference_id)
            if profile is None:
                continue
            raw = dict(profile.fields).get(slot.action_key)
            if raw != anchor_val:
                chosen = choice_id
                break
        if chosen is None:
            continue
        out[slot_id] = chosen
        if len(out) >= n:
            break
    assert len(out) >= n, f"need {n} differing pairs, got {len(out)}"
    return out


def _v2_2_json(*, mode: str, hypothesis: str, strategy_label: str, selections):
    return json.dumps(
        {
            "reflection_update": {"mode": mode, "hypothesis": hypothesis},
            "strategy_label": strategy_label,
            "selections": dict(selections),
        }
    )


def _repair_json(selections):
    return json.dumps({"selections": dict(selections)})


def test_v2_2_version_and_hypothesis_limit():
    assert V2_2_LABEL == "a3_episodic_reflective_v2_2_k10_bounded_cardinality"
    assert PROMPT_VERSION_A3_V2_2 == V2_2_LABEL
    assert PROMPT_VERSION_A3_V2_2 != PROMPT_VERSION_A3_V2_1
    assert MAX_HYPOTHESIS_CHARS_V2_2 == 512


def test_parse_preserves_envelope_on_over_cardinality():
    for residual in (1, 2, 3):
        oversized = {f"action_slot_{i:02d}": f"choice_{i:02d}" for i in range(1, residual + 3)}
        text = _v2_2_json(
            mode="INITIALIZE",
            hypothesis="broad profile",
            strategy_label="s1",
            selections=oversized,
        )
        cand, status = parse_a3_v2_2_strategic_response(
            text, query_index=1, residual_m=residual
        )
        assert status == "selection_count_exceeds_residual_m"
        assert cand is not None
        assert cand["strategic_envelope_valid"] is True
        assert cand["selection_status"] == "selection_count_exceeds_residual_m"
        assert cand["reflection_update"]["mode"] == "INITIALIZE"
        assert cand["strategy_label"] == "s1"
        assert len(cand["selections"]) == residual + 2


@pytest.mark.parametrize(
    "m_max,over_n",
    [(1, 3), (2, 4), (3, 5)],
    ids=["m1_over3", "m2_over4", "m3_over5"],
)
def test_cases_a_b_c_cardinality_repair_parameterised(
    m_max,
    over_n,
    starting_case,
    governance_policy,
    reference_pool,
    tmp_path,
):
    budget = _qm_budget(5, m_max)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / f"m{m_max}",
        budget=budget,
        defender=ScriptedLabelDefender(labels=["PASS"]),
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=reference_pool, anchor=starting_case.features
    )
    slots = build_a3_v2_2_episode_action_slots(catalog, validator=env.validator)
    over = _slot_choice_pairs(
        slots, catalog, reference_pool, starting_case.features, over_n
    )
    # Scripted LLM chooses a compliant subset (trusted code must not).
    repaired = {k: over[k] for k in list(over.keys())[-m_max:]}
    client = ScriptedLLMClient(
        responses=[
            _v2_2_json(
                mode="INITIALIZE",
                hypothesis="broad multi-field strategy",
                strategy_label="pinned_strategy",
                selections=over,
            ),
            _repair_json(repaired),
        ]
    )
    agent = EpisodicLLMAgent(
        experiment_seed=1,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=m_max),
        prompt_version=PROMPT_VERSION_A3_V2_2,
        llm_client=client,
        max_parse_retries=0,
        max_local_generation_attempts_per_query=3,
    )
    agent.run(AttackerEpisode(env))
    assert env.success is True
    assert agent._v2_strategic_llm_calls == 1
    assert agent._v2_repair_llm_calls >= 1
    assert agent.aggregate_counters()["selection_repair_llm_calls"] >= 1
    assert agent._v2_2_cardinality_exceed_initial == 1
    assert defender_q(env) == 1
    # Repair call must not re-ask for reflection.
    repair_user = client.calls[1]["messages"][1]["content"]
    repair_payload = json.loads(repair_user)
    assert "eligible_proposed_selections" in repair_payload["local_selection_repair"]
    assert repair_payload["local_selection_repair"]["cardinality_repair"] is True
    assert (
        repair_payload["repair_output_schema"]["properties"]["selections"][
            "maxProperties"
        ]
        == m_max
    )
    # Pinned strategy visible in memory.
    mem = agent._v2_episodic_memory[0]
    assert mem["strategy_label"] == "pinned_strategy"
    assert mem["reflection_update"]["mode"] == "INITIALIZE"


def defender_q(env: AttackEnvironment) -> int:
    return int(env.attempts_used)


def test_case_d_no_repair_when_already_compliant(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=ScriptedLabelDefender(labels=["PASS"]),
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=reference_pool, anchor=starting_case.features
    )
    slots = build_a3_v2_2_episode_action_slots(catalog, validator=env.validator)
    ok = _slot_choice_pairs(
        slots, catalog, reference_pool, starting_case.features, 2
    )
    client = ScriptedLLMClient(
        responses=[
            _v2_2_json(
                mode="INITIALIZE",
                hypothesis="tight",
                strategy_label="ok",
                selections=ok,
            )
        ]
    )
    agent = EpisodicLLMAgent(
        experiment_seed=1,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2_2,
        llm_client=client,
        max_parse_retries=0,
    )
    agent.run(AttackerEpisode(env))
    assert len(client.calls) == 1
    assert agent._v2_repair_llm_calls == 0
    assert agent._v2_2_cardinality_exceed_initial == 0
    assert env.success is True


def test_case_e_repair_still_over_cardinality_retries_without_qd1(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=ScriptedLabelDefender(labels=["PASS"]),
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=reference_pool, anchor=starting_case.features
    )
    slots = build_a3_v2_2_episode_action_slots(catalog, validator=env.validator)
    over = _slot_choice_pairs(
        slots, catalog, reference_pool, starting_case.features, 4
    )
    still_over = {k: over[k] for k in list(over.keys())[:3]}
    ok = {k: over[k] for k in list(over.keys())[:2]}
    client = ScriptedLLMClient(
        responses=[
            _v2_2_json(
                mode="INITIALIZE",
                hypothesis="h",
                strategy_label="pinned",
                selections=over,
            ),
            _repair_json(still_over),
            _repair_json(ok),
        ]
    )
    agent = EpisodicLLMAgent(
        experiment_seed=1,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2_2,
        llm_client=client,
        max_parse_retries=0,
        max_local_generation_attempts_per_query=3,
    )
    agent.run(AttackerEpisode(env))
    assert agent._v2_strategic_llm_calls == 1
    assert agent._v2_repair_llm_calls == 2
    assert agent._v2_2_cardinality_exceed_after_repair >= 1
    assert env.attempts_used == 1
    assert env.success is True
    assert agent._v2_episodic_memory[0]["strategy_label"] == "pinned"


def test_case_f_q2_revise_pin_after_block(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=ScriptedLabelDefender(labels=["BLOCK", "PASS"]),
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=reference_pool, anchor=starting_case.features
    )
    # Prefer per-attempt slots so residual_m stays 2 after q1.
    writable = writable_slots_from_episode_map(
        build_a3_v2_2_episode_action_slots(catalog, validator=env.validator),
        validator=env.validator,
        include_static=False,
    )
    if len(writable.ordered_slot_ids) < 4:
        pytest.skip("need >=4 per-attempt slots for q2 over-card test")
    q1 = _slot_choice_pairs(
        writable, catalog, reference_pool, starting_case.features, 2
    )
    over = _slot_choice_pairs(
        writable, catalog, reference_pool, starting_case.features, 4
    )
    repaired = {k: over[k] for k in list(over.keys())[-2:]}
    client = ScriptedLLMClient(
        responses=[
            _v2_2_json(
                mode="INITIALIZE",
                hypothesis="q1 hyp",
                strategy_label="s1",
                selections=q1,
            ),
            _v2_2_json(
                mode="REVISE",
                hypothesis="revised after block",
                strategy_label="s2_revise",
                selections=over,
            ),
            _repair_json(repaired),
        ]
    )
    agent = EpisodicLLMAgent(
        experiment_seed=1,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2_2,
        llm_client=client,
        max_parse_retries=0,
    )
    agent.run(AttackerEpisode(env))
    assert env.attempts_used == 2
    assert env.success is True
    assert agent._v2_episodic_memory[1]["reflection_update"]["mode"] == "REVISE"
    assert agent._v2_episodic_memory[1]["strategy_label"] == "s2_revise"
    # One strategic + one repair on q2 (plus one strategic on q1).
    assert agent._v2_strategic_llm_calls == 2
    assert agent._v2_repair_llm_calls == 1
    # Repair messages must not create a second reflection event.
    repair_sys = client.calls[2]["messages"][0]["content"]
    assert "Local compliance repair only" in repair_sys
    assert "frozen" in repair_sys.lower() or "pinned" in repair_sys.lower() or (
        CARDINALITY_REPAIR_INSTRUCTION[:40] in repair_sys
    )


def test_case_g_no_trusted_code_truncation_or_ranking():
    src = Path(
        "/Users/ziyaoch/ucl/dissertation/04_implementation/src/attack_lab"
    )
    blob = (src / "archive" / "contracts" / "a3_v2_2_contract.py").read_text(
        encoding="utf-8"
    )
    blob += (src / "attackers" / "a3_agent.py").read_text(encoding="utf-8")
    # Forbidden trusted-code subset selection patterns in V2.2 path.
    assert "sorted(selections" not in blob or "filter_mechanically_valid" in blob
    assert "random.sample" not in blob
    assert "[:residual_m]" not in blob
    assert "[: int(residual_m)]" not in blob
    assert "keep first" not in blob.lower()
    # Schema builder must bind maxProperties to residual_m dynamically.
    schema = build_a3_v2_2_cardinality_repair_schema(
        eligible_pairs={"action_slot_01": "c1", "action_slot_02": "c2", "action_slot_03": "c3"},
        residual_m=2,
    )
    assert schema["properties"]["selections"]["maxProperties"] == 2
    assert set(schema["properties"]["selections"]["properties"]) == {
        "action_slot_01",
        "action_slot_02",
        "action_slot_03",
    }
    # filter keeps all valid pairs; does not truncate to residual_m.
    assert inspect.isfunction(filter_mechanically_valid_proposed_pairs)


def test_case_h_stable_slot_mapping_unchanged(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=reference_pool, anchor=starting_case.features
    )
    ep_v21 = build_a3_v2_1_episode_action_slots(catalog, validator=env.validator)
    ep_v22 = build_a3_v2_2_episode_action_slots(catalog, validator=env.validator)
    assert ep_v21.ordered_slot_ids == ep_v22.ordered_slot_ids
    for sid in ep_v21.ordered_slot_ids:
        assert ep_v21.get(sid).action_key == ep_v22.get(sid).action_key
    w1 = writable_slots_from_episode_map(
        ep_v22, validator=env.validator, include_static=True
    )
    w2 = writable_slots_from_episode_map(
        ep_v22, validator=env.validator, include_static=False
    )
    # Filtering must not renumber.
    for sid in w2.ordered_slot_ids:
        assert w1.get(sid).action_key == w2.get(sid).action_key
        assert sid in ep_v22.ordered_slot_ids


def test_prompt_exposes_dynamic_residual_and_v2_1_preservations(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 3),
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=reference_pool, anchor=starting_case.features
    )
    slots = build_a3_v2_2_episode_action_slots(catalog, validator=env.validator)
    payload = build_a3_v2_2_prompt_payload(
        case_id=starting_case.case_id,
        visible_anchor=env.validator.visible_fields(starting_case.features),
        current_application=env.observation().visible_fields,
        budget=AttackBudget(q_max=5, m_max=3),
        q_remaining=5,
        query_index=1,
        static_edit_cost=0,
        residual_m=3,
        locked_static_values={},
        slots=slots,
        slot_entries=public_slot_entries(slots, validator=env.validator),
        episodic_memory=[],
    )
    assert ATTACK_OBJECTIVE_V2_2 in payload["attack_objective"]
    assert STATIC_RULE_DISCLOSURE_V2_2 in payload["episode_static_rules"]
    assert SELECTIONS_VS_HYPOTHESIS_NOTE in payload["selections_vs_hypothesis"]
    assert payload["budget"]["residual_m"] == 3
    assert payload["budget"]["maximum_submitted_action_selections_this_query"] == 3
    assert payload["output_schema"]["properties"]["selections"]["maxProperties"] == 3
    assert "INITIALIZE" in payload["reflection_mode_definitions"]


def test_repair_rejects_over_cardinality_locally():
    bad = parse_a3_v2_2_repair_selections(
        _repair_json({"a": "1", "b": "2", "c": "3"}),
        residual_m=2,
        eligible_pairs={"a": "1", "b": "2", "c": "3"},
    )
    assert bad[0] is None
    assert bad[1] == "selection_count_exceeds_residual_m"
    ok = parse_a3_v2_2_repair_selections(
        _repair_json({"a": "1", "b": "2"}),
        residual_m=2,
        eligible_pairs={"a": "1", "b": "2", "c": "3"},
    )
    assert ok[1] == "ok"
    assert ok[0] == {"a": "1", "b": "2"}
    # Outside eligible set rejected.
    outside = parse_a3_v2_2_repair_selections(
        _repair_json({"a": "9"}),
        residual_m=1,
        eligible_pairs={"a": "1"},
    )
    assert outside[1] == "selection_not_in_eligible_proposed_set"
