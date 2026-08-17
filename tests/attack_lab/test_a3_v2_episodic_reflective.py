"""Focused unit tests for A3 V2 episodic reflective hard contract (no DeepSeek)."""

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
    PROMPT_VERSION_B2_GROUNDED_REFLECTION,
    EpisodicLLMAgent,
)
from attack_lab.archive.contracts.a3_v2_contract import (
    build_a3_v2_action_slots,
    build_a3_v2_prompt_payload,
    build_v4_choice_catalog,
    compute_static_cost_and_residual,
    parse_a3_v2_repair_selections,
    parse_a3_v2_strategic_response,
    public_slot_entries,
    resolve_a3_v2_selections,
)
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import DefenceDecision, InternalDefenceResult
from attack_lab.validator import ConstraintValidator


@dataclass
class CountingBlockDefender:
    name: str = "counting_block"
    artefact_id: str = "test"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult:
        self.calls += 1
        return InternalDefenceResult(
            risk_score=0.9,
            threshold=self.threshold,
            decision="BLOCK",
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


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
        self.calls.append(
            {
                "messages": [dict(item) for item in messages],
                "model": model,
            }
        )
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
        q_max=q_max, m_max=m_max, label="dummy_a3_v2_qm"
    )


def _make_env(
    *,
    starting_case,
    governance_policy,
    reference_pool,
    tmp_path: Path,
    budget: BudgetSpec,
    enabled: tuple[str, ...] | None = None,
    defender=None,
):
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
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    return AttackEnvironment(
        starting_case=starting_case,
        defender=defender or CountingBlockDefender(),
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


def _catalog_slots(env, pool):
    catalog = build_v4_choice_catalog(
        validator=env.validator, pool=pool, anchor=env.starting_case.features
    )
    slots = build_a3_v2_action_slots(
        catalog, validator=env.validator, include_static=True
    )
    return catalog, slots


def _slot_for(slots, action_key: str):
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        if slot.action_key == action_key:
            return slot
    raise AssertionError(f"missing slot for {action_key}")


def _pick_n_selections(
    slots, catalog, n: int, *, prefer_per_attempt: bool = True, validator=None
) -> dict[str, str]:
    out: dict[str, str] = {}
    ordered = list(slots.ordered_slot_ids)
    if prefer_per_attempt and validator is not None:
        per: list[str] = []
        static: list[str] = []
        for slot_id in ordered:
            slot = slots.get(slot_id)
            assert slot is not None
            rule = validator.policy.field_for_action(slot.action_key)
            if rule is not None and rule.is_episode_locked:
                static.append(slot_id)
            else:
                per.append(slot_id)
        ordered = per + static
    for slot_id in ordered:
        if len(out) >= n:
            break
        slot = slots.get(slot_id)
        assert slot is not None
        for choice_id in slot.allowed_choice_ids:
            choice = catalog.get(choice_id)
            if choice is None:
                continue
            out[slot_id] = choice_id
            break
    assert len(out) == n
    return out


def _adaptive_legal_client(first_selections: Mapping[str, str] | None = None):
    """Build responses from the live prompt catalogue (strategic + repair)."""

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            payload = json.loads(messages[1]["content"])
            entries = payload["action_slots"]
            assert entries
            residual = int(payload["budget"]["residual_m"])
            repair = payload.get("local_selection_repair")
            q = int(payload["budget"]["query_index"])
            # Prefer per-attempt slots when present.
            preferred = [
                e for e in entries if e.get("category") == "per_attempt"
            ] or list(entries)
            # Avoid previously rejected fingerprints when repairing.
            forbidden = set()
            if isinstance(repair, dict):
                for item in repair.get("do_not_repeat_selection_fingerprints") or []:
                    forbidden.add(str(item))
                priors = repair.get("prior_local_rejections") or []
                used_slots = {
                    next(iter(dict(p.get("selections") or {})), None) for p in priors
                }
                preferred = [
                    e
                    for e in preferred
                    if e["action_slot_id"] not in used_slots
                ] or preferred
            # Rotate by query / call index for distinctness.
            idx = (len(self.calls) - 1) % len(preferred)
            entry = preferred[idx]
            choice = entry["allowed_choice_ids"][
                min(len(self.calls) - 1, len(entry["allowed_choice_ids"]) - 1)
            ]
            selections = {entry["action_slot_id"]: choice}
            if first_selections is not None and len(self.calls) == 1 and repair is None:
                selections = dict(first_selections)
            # Ensure cardinality
            if len(selections) > residual:
                k = next(iter(selections))
                selections = {k: selections[k]}
            if repair is not None:
                return LLMCompletion(
                    text=_repair_json(selections),
                    model=kwargs["model"],
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cached_tokens=0,
                    latency_ms=1.0,
                    thinking_disabled=kwargs["thinking_disabled"],
                )
            mode = "INITIALIZE" if q == 1 else ("RETAIN" if q == 2 else "REVISE")
            return LLMCompletion(
                text=_v2_json(
                    mode=mode,
                    hypothesis=f"Observable strategy hypothesis for query {q}.",
                    strategy_label=f"q{q}",
                    selections=selections,
                ),
                model=kwargs["model"],
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=10,
                cached_tokens=0,
                latency_ms=1.0,
                thinking_disabled=kwargs["thinking_disabled"],
            )

    return Client(responses=[])


def _v2_json(
    *,
    mode: str,
    hypothesis: str,
    strategy_label: str,
    selections: Mapping[str, str],
) -> str:
    return json.dumps(
        {
            "reflection_update": {"mode": mode, "hypothesis": hypothesis},
            "strategy_label": strategy_label,
            "selections": dict(selections),
        }
    )


def _repair_json(selections: Mapping[str, str]) -> str:
    return json.dumps({"selections": dict(selections)})


def test_prompt_version_string():
    assert PROMPT_VERSION_A3_V2 == "a3_episodic_reflective_v2_k10_hard_contract"
    assert PROMPT_VERSION_B2_GROUNDED_REFLECTION != PROMPT_VERSION_A3_V2


def test_q1_contract_initialize_and_provenance(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
    )
    catalog, slots = _catalog_slots(env, reference_pool)
    sels = _pick_n_selections(slots, catalog, 1, validator=env.validator)
    client = _adaptive_legal_client(first_selections=sels)
    attacker = EpisodicLLMAgent(
        experiment_seed=7,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=client,
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert attacker.total_env_steps >= 1
    assert client.calls
    user = client.calls[0]["messages"][1]["content"]
    assert "INITIALIZE" in user
    rec = attacker.query_records[0]
    assert rec.submitted
    assert rec.changes["reflection_update"]["mode"] == "INITIALIZE"
    mem = attacker._v2_episodic_memory[0]
    assert mem["public_label"] in {"BLOCK", "PASS", "INVALID"}
    assert mem["selected_actions"]
    resolved, status = resolve_a3_v2_selections(sels, slots=slots, catalog=catalog)
    assert status == ""
    assert resolved is not None
    for sel in resolved.values():
        assert sel.reference_id


def test_q2_after_block_prompt_contains_prior_outcome(
    starting_case, governance_policy, reference_pool, tmp_path
):
    defender = ScriptedLabelDefender(labels=["BLOCK", "BLOCK"])
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=defender,
    )
    client = _adaptive_legal_client()
    attacker = EpisodicLLMAgent(
        experiment_seed=11,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=client,
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert len(client.calls) >= 2
    assert attacker.query_records[0].public_label == "BLOCK"
    assert attacker.query_records[1].changes["reflection_update"]["mode"] in {
        "RETAIN",
        "REVISE",
        "ABANDON",
    }
    q2_user = client.calls[1]["messages"][1]["content"]
    payload = json.loads(q2_user)
    assert payload["episodic_memory"][0]["public_label"] == "BLOCK"
    assert payload["post_feedback_reflection_required"] is True
    assert "You do not know which selected action" in json.dumps(payload)


def test_genuine_timing_not_copy_q1_hypothesis(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=ScriptedLabelDefender(labels=["BLOCK", "BLOCK"]),
    )
    q1_hyp = "Q1-ONLY-HYPOTHESIS-MARKER-AAAA"
    q2_hyp = "Q2-POST-BLOCK-HYPOTHESIS-MARKER-BBBB"

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            payload = json.loads(messages[1]["content"])
            if payload.get("local_selection_repair") is not None:
                entry = payload["action_slots"][0]
                return LLMCompletion(
                    text=_repair_json(
                        {entry["action_slot_id"]: entry["allowed_choice_ids"][0]}
                    ),
                    model=kwargs["model"],
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cached_tokens=0,
                    latency_ms=1.0,
                    thinking_disabled=kwargs["thinking_disabled"],
                )
            preferred = [
                e for e in payload["action_slots"] if e.get("category") == "per_attempt"
            ] or payload["action_slots"]
            entry = preferred[min(len(self.calls) - 1, len(preferred) - 1)]
            hyp = q1_hyp if int(payload["budget"]["query_index"]) == 1 else q2_hyp
            mode = (
                "INITIALIZE"
                if int(payload["budget"]["query_index"]) == 1
                else "ABANDON"
            )
            if int(payload["budget"]["query_index"]) > 1:
                assert payload["episodic_memory"][0]["reflection_update"][
                    "hypothesis"
                ] == q1_hyp
            return LLMCompletion(
                text=_v2_json(
                    mode=mode,
                    hypothesis=hyp,
                    strategy_label=f"q{payload['budget']['query_index']}",
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
        experiment_seed=13,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=Client(responses=[]),
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert attacker.query_records[1].changes["reflection_update"]["hypothesis"] == q2_hyp
    assert q1_hyp != q2_hyp


def test_retain_same_family_different_choice_allowed(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=ScriptedLabelDefender(labels=["BLOCK", "BLOCK"]),
    )
    catalog, slots = _catalog_slots(env, reference_pool)
    target = None
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        rule = env.validator.policy.field_for_action(slot.action_key)
        if rule is None or rule.is_episode_locked:
            continue
        if len(slot.allowed_choice_ids) >= 2:
            target = slot
            break
    if target is None:
        pytest.skip("need per-attempt slot with >=2 choices")

    c1, c2 = target.allowed_choice_ids[0], target.allowed_choice_ids[1]

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            payload = json.loads(messages[1]["content"])
            if payload.get("local_selection_repair") is not None:
                entry = [
                    e
                    for e in payload["action_slots"]
                    if e["action_key"] == target.action_key
                ][0]
                return LLMCompletion(
                    text=_repair_json(
                        {entry["action_slot_id"]: entry["allowed_choice_ids"][0]}
                    ),
                    model=kwargs["model"],
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cached_tokens=0,
                    latency_ms=1.0,
                    thinking_disabled=kwargs["thinking_disabled"],
                )
            if int(payload["budget"]["query_index"]) == 1:
                text = _v2_json(
                    mode="INITIALIZE",
                    hypothesis="Probe one family.",
                    strategy_label="fam",
                    selections={target.action_slot_id: c1},
                )
            else:
                matching = [
                    e
                    for e in payload["action_slots"]
                    if e["action_key"] == target.action_key
                ]
                assert matching, "code must not force a field switch"
                entry = matching[0]
                assert c2 in entry["allowed_choice_ids"]
                text = _v2_json(
                    mode="RETAIN",
                    hypothesis="Retain family; vary reference-backed detail.",
                    strategy_label="fam",
                    selections={entry["action_slot_id"]: c2},
                )
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
        experiment_seed=17,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=Client(responses=[]),
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert attacker.query_records[1].changes["reflection_update"]["mode"] == "RETAIN"
    assert attacker.query_records[1].submitted


def test_local_repair_pins_reflection(
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
    catalog, slots = _catalog_slots(env, reference_pool)
    good = _pick_n_selections(slots, catalog, 1, validator=env.validator)
    bad = {"action_slot_99": "choice_999"}
    hyp = "Pinned hypothesis must survive repair."

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            if len(self.calls) == 1:
                text = _v2_json(
                    mode="INITIALIZE",
                    hypothesis=hyp,
                    strategy_label="pinned_label",
                    selections=bad,
                )
            else:
                payload = json.loads(messages[1]["content"])
                assert "local_selection_repair" in payload
                assert payload["local_selection_repair"]["pinned_reflection_update"][
                    "hypothesis"
                ] == hyp
                text = _repair_json(good)
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

    client = Client(responses=[])
    attacker = EpisodicLLMAgent(
        experiment_seed=19,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=client,
        max_parse_retries=0,
    )
    q_before = env.ledger.q_remaining
    attacker.run(AttackerEpisode(env))
    assert attacker.total_env_steps == 1
    assert env.ledger.q_remaining == q_before - 1
    assert len(client.calls) == 2
    repair_user = client.calls[1]["messages"][1]["content"]
    assert "local_selection_repair" in repair_user
    assert "pinned_reflection_update" in repair_user
    assert attacker.query_records[0].changes["reflection_update"]["hypothesis"] == hyp
    assert attacker.query_records[0].strategy_label == "pinned_label"
    assert attacker._v2_repair_llm_calls >= 1
    assert len(attacker._v2_episodic_memory) == 1
    assert attacker._v2_episodic_memory[0]["reflection_update"]["hypothesis"] == hyp


def test_max_cardinality_local_reject(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
    )
    catalog, slots = _catalog_slots(env, reference_pool)
    over = _pick_n_selections(slots, catalog, 2, validator=env.validator)
    # Force residual_m=1 by using parse with residual check directly
    parsed, status = parse_a3_v2_strategic_response(
        _v2_json(
            mode="INITIALIZE",
            hypothesis="Too many selections.",
            strategy_label="x",
            selections=over,
        ),
        query_index=1,
        residual_m=1,
    )
    assert parsed is None
    assert status == "selection_count_exceeds_residual_m"

    legal2 = _pick_n_selections(slots, catalog, 2, validator=env.validator)
    parsed2, status2 = parse_a3_v2_strategic_response(
        _v2_json(
            mode="INITIALIZE",
            hypothesis="Two legal selections.",
            strategy_label="ok",
            selections=legal2,
        ),
        query_index=1,
        residual_m=2,
    )
    assert status2 == "ok"
    assert parsed2 is not None


def test_slot_choice_hard_contract_failures():
    # Unknown slot / choice / forbidden keys
    bad_cases = [
        (
            {
                "reflection_update": {"mode": "INITIALIZE", "hypothesis": "h" * 10},
                "strategy_label": "s",
                "selections": {"action_slot_99": "choice_001"},
            },
            None,  # resolve later
        ),
        (
            {
                "reflection_update": {"mode": "INITIALIZE", "hypothesis": "h" * 10},
                "strategy_label": "s",
                "changes": {"income": 0.2},
            },
            "forbidden_output_key:changes",
        ),
        (
            {
                "reflection_update": {"mode": "INITIALIZE", "hypothesis": "h" * 10},
                "strategy_label": "s",
                "selections": {"action_slot_01": "choice_001"},
                "reference_id": "p1",
            },
            "forbidden_output_key:reference_id",
        ),
        (
            {
                "reflection_update": {"mode": "INITIALIZE", "hypothesis": "h" * 10},
                "strategy_label": "s",
                "selections": {"income": "choice_001"},
            },
            None,
        ),
    ]
    for payload, expected in bad_cases:
        cand, status = parse_a3_v2_strategic_response(
            json.dumps(payload), query_index=1, residual_m=2
        )
        if expected:
            assert status == expected
            assert cand is None
        else:
            # May parse but fail resolve; if forbidden keys not present, ok parse
            if "changes" in payload or "reference_id" in payload:
                assert cand is None


def test_slot_choice_resolve_rejects(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
    )
    catalog, slots = _catalog_slots(env, reference_pool)
    slot_a = slots.get(slots.ordered_slot_ids[0])
    slot_b = slots.get(slots.ordered_slot_ids[1])
    assert slot_a and slot_b
    # Wrong pair: choice from slot_b used under slot_a
    wrong = {slot_a.action_slot_id: slot_b.allowed_choice_ids[0]}
    resolved, status = resolve_a3_v2_selections(wrong, slots=slots, catalog=catalog)
    assert resolved is None
    assert status == "choice_not_in_action_slot"
    unknown, status2 = resolve_a3_v2_selections(
        {"action_slot_99": slot_a.allowed_choice_ids[0]},
        slots=slots,
        catalog=catalog,
    )
    assert unknown is None
    assert status2 == "unknown_action_slot_id"


def test_static_lock_excludes_writable_static(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        defender=ScriptedLabelDefender(labels=["BLOCK", "BLOCK"]),
    )
    catalog, slots = _catalog_slots(env, reference_pool)
    static_slot = None
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        rule = env.validator.policy.field_for_action(slot.action_key)
        if rule is not None and rule.is_episode_locked:
            # Prefer customer_age / employment / housing over address relationship fields
            if slot.action_key in {
                "customer_age",
                "employment_status",
                "housing_status",
            }:
                static_slot = slot
                break
            if static_slot is None:
                static_slot = slot
    if static_slot is None:
        pytest.skip("no episode-static slot available")

    class Client(ScriptedLLMClient):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": [dict(m) for m in messages]})
            payload = json.loads(messages[1]["content"])
            if payload.get("local_selection_repair") is not None:
                entry = [
                    e
                    for e in payload["action_slots"]
                    if e.get("category") == "per_attempt"
                ][0]
                return LLMCompletion(
                    text=_repair_json(
                        {entry["action_slot_id"]: entry["allowed_choice_ids"][0]}
                    ),
                    model=kwargs["model"],
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cached_tokens=0,
                    latency_ms=1.0,
                    thinking_disabled=kwargs["thinking_disabled"],
                )
            if int(payload["budget"]["query_index"]) == 1:
                # Find current static slot id from live catalogue
                live = [
                    e
                    for e in payload["action_slots"]
                    if e["action_key"] == static_slot.action_key
                ][0]
                text = _v2_json(
                    mode="INITIALIZE",
                    hypothesis="Lock a static dimension.",
                    strategy_label="static",
                    selections={
                        live["action_slot_id"]: live["allowed_choice_ids"][0]
                    },
                )
            else:
                assert payload["budget"]["static_edit_cost"] >= 1
                assert payload["budget"]["residual_m"] <= 1
                for entry in payload["action_slots"]:
                    assert entry["action_key"] != static_slot.action_key
                    assert entry["category"] != "episode_static"
                entry = payload["action_slots"][0]
                text = _v2_json(
                    mode="REVISE",
                    hypothesis="Continue with residual per-attempt capacity.",
                    strategy_label="after_lock",
                    selections={
                        entry["action_slot_id"]: entry["allowed_choice_ids"][0]
                    },
                )
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
        experiment_seed=23,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=Client(responses=[]),
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert attacker._static_locked
    cost, residual = compute_static_cost_and_residual(
        validator=env.validator,
        anchor=starting_case.features,
        locked_static_values=attacker._locked_static_values,
        m_max=2,
    )
    assert cost >= 1
    assert residual == 2 - cost


def test_pass_stops_without_q2(
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
    client = _adaptive_legal_client()
    attacker = EpisodicLLMAgent(
        experiment_seed=29,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=client,
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert env.success
    assert len(client.calls) == 1
    assert attacker._v2_strategic_llm_calls == 1


def test_block_block_pass_timeline(
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
    client = _adaptive_legal_client()
    attacker = EpisodicLLMAgent(
        experiment_seed=31,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_A3_V2,
        llm_client=client,
        max_parse_retries=0,
    )
    attacker.run(AttackerEpisode(env))
    assert env.success
    assert len(attacker.query_records) == 3
    assert all(r.submitted for r in attacker.query_records)
    assert attacker.query_records[0].changes["reflection_update"]["mode"] == "INITIALIZE"
    assert attacker.query_records[1].changes["reflection_update"]["mode"] in {
        "RETAIN",
        "REVISE",
        "ABANDON",
    }
    assert attacker.query_records[2].changes["reflection_update"]["mode"] in {
        "RETAIN",
        "REVISE",
        "ABANDON",
    }
    assert len(client.calls) == 3
    assert attacker._v2_strategic_llm_calls == 3
    assert [m["query_index"] for m in attacker._v2_episodic_memory] == [1, 2, 3]
    assert [m["public_label"] for m in attacker._v2_episodic_memory] == [
        "BLOCK",
        "BLOCK",
        "PASS",
    ]


def test_repair_parse_rejects_new_reflection():
    text = json.dumps(
        {
            "reflection_update": {"mode": "REVISE", "hypothesis": "new"},
            "selections": {"action_slot_01": "choice_001"},
        }
    )
    sels, status = parse_a3_v2_repair_selections(text, residual_m=2)
    assert sels is None
    assert status == "reflection_immutable"


def test_prompt_payload_hard_contract(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
    )
    catalog, slots = _catalog_slots(env, reference_pool)
    payload = build_a3_v2_prompt_payload(
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
    blob = json.dumps(payload)
    assert "risk_score" not in blob
    assert "feature_importance" not in blob
    assert "name_email_similarity" not in blob
