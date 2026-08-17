"""Focused regressions for A3 V2.1 construct-alignment (no DeepSeek)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from attack_lab.attacker_interface import AttackerEpisode
from attack_lab.attackers.a1_planner import LLMCompletion
from attack_lab.attackers.a3_agent import (
    PROMPT_VERSION_A3_V2,
    PROMPT_VERSION_A3_V2_1,
    EpisodicLLMAgent,
)
from attack_lab.archive.contracts.a3_v2_1_contract import (
    ATTACK_OBJECTIVE_V2_1,
    MAX_HYPOTHESIS_CHARS_V2_1,
    REFLECTION_MODE_DEFINITIONS_V2_1,
    STATIC_RULE_DISCLOSURE_V2_1,
    build_a3_v2_1_episode_action_slots,
    build_a3_v2_1_prompt_payload,
    parse_a3_v2_1_strategic_response,
    public_slot_entries,
    writable_slots_from_episode_map,
)
from attack_lab.archive.contracts.a3_v2_contract import (
    MAX_HYPOTHESIS_CHARS,
    build_v4_choice_catalog,
    parse_a3_v2_strategic_response,
)
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
        q_max=q_max, m_max=m_max, label="dummy_a3_v2_1_qm"
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


def _v2_1_json(*, mode: str, hypothesis: str, strategy_label: str, selections):
    return json.dumps(
        {
            "reflection_update": {"mode": mode, "hypothesis": hypothesis},
            "strategy_label": strategy_label,
            "selections": dict(selections),
        }
    )


def test_v2_1_version_distinct_from_v2():
    assert PROMPT_VERSION_A3_V2_1 == "a3_episodic_reflective_v2_1_k10_hard_contract"
    assert PROMPT_VERSION_A3_V2_1 != PROMPT_VERSION_A3_V2
    assert MAX_HYPOTHESIS_CHARS_V2_1 == 512
    assert MAX_HYPOTHESIS_CHARS == 240


def test_pass_oriented_objective_not_diagnostic(
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
    slots = build_a3_v2_1_episode_action_slots(catalog, validator=env.validator)
    payload = build_a3_v2_1_prompt_payload(
        case_id=starting_case.case_id,
        visible_anchor=env.validator.visible_fields(starting_case.features),
        current_application=env.observation().visible_fields,
        budget=AttackBudget(q_max=5, m_max=2),
        q_remaining=5,
        query_index=1,
        static_edit_cost=0,
        residual_m=2,
        locked_static_values={},
        slots=slots,
        slot_entries=public_slot_entries(slots, validator=env.validator),
        episodic_memory=[],
    )
    assert ATTACK_OBJECTIVE_V2_1 in payload["attack_objective"]
    assert "obtain PASS" in payload["task"]
    assert "most likely to obtain PASS" in payload["task"]
    assert "diagnosing the defender" in payload["reflection_purpose"]
    assert "find a blocking rule" not in payload["attack_objective"].lower()
    assert "key risk factor" not in payload["attack_objective"].lower()
    assert STATIC_RULE_DISCLOSURE_V2_1 == payload["episode_static_rules"]
    for key in REFLECTION_MODE_DEFINITIONS_V2_1:
        assert key in payload["reflection_mode_definitions"]
    schema = payload["output_schema"]["properties"]["reflection_update"]["properties"][
        "hypothesis"
    ]
    assert schema["maxLength"] == 512


def test_stable_action_slot_ids_across_queries(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=ScriptedLabelDefender(labels=["BLOCK", "BLOCK", "PASS"]),
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=reference_pool, anchor=starting_case.features
    )
    episode_slots = build_a3_v2_1_episode_action_slots(
        catalog, validator=env.validator
    )
    income_slot = None
    for sid in episode_slots.ordered_slot_ids:
        slot = episode_slots.get(sid)
        assert slot is not None
        if slot.action_key == "income":
            income_slot = slot
            break
    assert income_slot is not None
    income_id = income_slot.action_slot_id

    observed_maps: list[dict[str, str]] = []

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            payload = json.loads(messages[1]["content"])
            mapping = {
                e["action_slot_id"]: e["action_key"]
                for e in payload.get("episode_action_slot_map") or []
            }
            observed_maps.append(mapping)
            assert mapping.get(income_id) == "income"
            # Writable slots must not renumber income away
            writable = {
                e["action_slot_id"]: e["action_key"] for e in payload["action_slots"]
            }
            if income_id in writable:
                assert writable[income_id] == "income"
            preferred = [
                e for e in payload["action_slots"] if e.get("category") == "per_attempt"
            ] or payload["action_slots"]
            entry = preferred[min(len(self.calls) - 1, len(preferred) - 1)]
            q = int(payload["budget"]["query_index"])
            mode = "INITIALIZE" if q == 1 else "REVISE"
            return LLMCompletion(
                text=_v2_1_json(
                    mode=mode,
                    hypothesis=f"Hypothesis for query {q}.",
                    strategy_label=f"q{q}",
                    selections={
                        entry["action_slot_id"]: entry["allowed_choice_ids"][0]
                    },
                ),
                model=kwargs["model"],
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=10,
                cached_tokens=0,
                latency_ms=1.0,
                thinking_disabled=kwargs["thinking_disabled"],
            )

    attacker = EpisodicLLMAgent(
        experiment_seed=41,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2_1,
        llm_client=Client(responses=[]),
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert len(observed_maps) >= 2
    assert all(m.get(income_id) == "income" for m in observed_maps)
    # Ensure no renumbering of income to another key
    for m in observed_maps:
        for sid, key in m.items():
            if key == "income":
                assert sid == income_id


def test_writable_filter_preserves_ids(
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
    full = build_a3_v2_1_episode_action_slots(catalog, validator=env.validator)
    filtered = writable_slots_from_episode_map(
        full, validator=env.validator, include_static=False
    )
    for sid in filtered.ordered_slot_ids:
        assert full.get(sid) is not None
        assert filtered.get(sid).action_key == full.get(sid).action_key  # type: ignore[union-attr]
        rule = env.validator.policy.field_for_action(filtered.get(sid).action_key)  # type: ignore[union-attr]
        assert rule is not None and not rule.is_episode_locked


def test_hypothesis_512_accepted_over_rejected():
    hyp_ok = "x" * 512
    hyp_bad = "x" * 513
    body = {
        "reflection_update": {"mode": "INITIALIZE", "hypothesis": hyp_ok},
        "strategy_label": "s",
        "selections": {"action_slot_01": "choice_001"},
    }
    cand, status = parse_a3_v2_1_strategic_response(
        json.dumps(body), query_index=1, residual_m=2
    )
    assert status == "ok"
    assert cand is not None
    body["reflection_update"]["hypothesis"] = hyp_bad
    cand2, status2 = parse_a3_v2_1_strategic_response(
        json.dumps(body), query_index=1, residual_m=2
    )
    assert cand2 is None
    assert status2 == "hypothesis_too_long"
    # Historical V2 still rejects 241+
    body["reflection_update"]["hypothesis"] = "y" * 241
    _, v2_status = parse_a3_v2_strategic_response(
        json.dumps(body), query_index=1, residual_m=2
    )
    assert v2_status == "hypothesis_too_long"


def test_local_repair_pins_reflection_v2_1(
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
    slots = build_a3_v2_1_episode_action_slots(catalog, validator=env.validator)
    # prefer income
    good = None
    for sid in slots.ordered_slot_ids:
        slot = slots.get(sid)
        assert slot is not None
        if slot.action_key == "income":
            good = {sid: slot.allowed_choice_ids[0]}
            break
    assert good is not None
    hyp = "Pinned V2.1 hypothesis survives repair."

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            if len(self.calls) == 1:
                text = _v2_1_json(
                    mode="INITIALIZE",
                    hypothesis=hyp,
                    strategy_label="pinned",
                    selections={"action_slot_99": "choice_999"},
                )
            else:
                payload = json.loads(messages[1]["content"])
                assert "local_selection_repair" in payload
                assert payload["local_selection_repair"]["pinned_reflection_update"][
                    "hypothesis"
                ] == hyp
                text = json.dumps({"selections": good})
            return LLMCompletion(
                text=text,
                model=kwargs["model"],
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=10,
                cached_tokens=0,
                latency_ms=1.0,
                thinking_disabled=kwargs["thinking_disabled"],
            )

    attacker = EpisodicLLMAgent(
        experiment_seed=43,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2_1,
        llm_client=Client(responses=[]),
        max_parse_retries=0,
    )
    q_before = env.ledger.q_remaining
    attacker.run(AttackerEpisode(env))
    assert attacker.total_env_steps == 1
    assert env.ledger.q_remaining == q_before - 1
    assert attacker.query_records[0].changes["reflection_update"]["hypothesis"] == hyp
    assert len(attacker._v2_episodic_memory) == 1


def test_post_feedback_timing_v2_1(
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

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            payload = json.loads(messages[1]["content"])
            q = int(payload["budget"]["query_index"])
            if q > 1:
                assert payload["episodic_memory"]
                assert payload["episodic_memory"][0]["public_label"] == "BLOCK"
                assert payload["post_feedback_reflection_required"] is True
            preferred = [
                e for e in payload["action_slots"] if e.get("category") == "per_attempt"
            ] or payload["action_slots"]
            entry = preferred[min(q - 1, len(preferred) - 1)]
            choice = entry["allowed_choice_ids"][
                min(q - 1, len(entry["allowed_choice_ids"]) - 1)
            ]
            mode = "INITIALIZE" if q == 1 else "RETAIN"
            return LLMCompletion(
                text=_v2_1_json(
                    mode=mode,
                    hypothesis=f"q{q} hyp",
                    strategy_label=f"q{q}",
                    selections={entry["action_slot_id"]: choice},
                ),
                model=kwargs["model"],
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=10,
                cached_tokens=0,
                latency_ms=1.0,
                thinking_disabled=kwargs["thinking_disabled"],
            )

    attacker = EpisodicLLMAgent(
        experiment_seed=47,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2_1,
        llm_client=Client(responses=[]),
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    submitted = [r for r in attacker.query_records if r.submitted]
    assert len(submitted) >= 2
    assert submitted[0].changes["reflection_update"]["mode"] == "INITIALIZE"
    assert submitted[1].changes["reflection_update"]["mode"] == "RETAIN"
    assert env.success


def test_mechanical_gate_fails_on_local_generation_exhausted():
    """Smoke-audit helper contract: exhaustion is a mechanical failure."""

    def mechanical_gate(
        *,
        completed: int,
        n: int,
        reaching: int,
        local_generation_exhausted: int,
        non_ref: int,
        hidden: int,
        timing_ok: bool,
        errors: list,
        q_viol: int,
        m_viol: int,
    ) -> bool:
        return (
            completed == n
            and reaching == n
            and local_generation_exhausted == 0
            and non_ref == 0
            and hidden == 0
            and timing_ok
            and not errors
            and q_viol == 0
            and m_viol == 0
        )

    assert not mechanical_gate(
        completed=25,
        n=25,
        reaching=25,
        local_generation_exhausted=1,
        non_ref=0,
        hidden=0,
        timing_ok=True,
        errors=[],
        q_viol=0,
        m_viol=0,
    )
    assert mechanical_gate(
        completed=25,
        n=25,
        reaching=25,
        local_generation_exhausted=0,
        non_ref=0,
        hidden=0,
        timing_ok=True,
        errors=[],
        q_viol=0,
        m_viol=0,
    )
