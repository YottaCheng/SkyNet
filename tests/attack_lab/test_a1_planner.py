"""Unit tests for the A1 non-adaptive one-shot LLM planner."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from attack_lab.attackers.a1_planner import (
    DEFAULT_MAX_LOCAL_GENERATION_ATTEMPTS,
    DEFAULT_MAX_PARSE_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOP_P,
    DIVERSIFICATION_PRINCIPLE_V2,
    FORMAL_A1_MODEL_CONFIG,
    LLMCompletion,
    OneShotLLMPlanner,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V3,
    PROMPT_VERSION_V4,
    build_a1_prompt_payload,
    estimate_flash_cost_usd,
    format_a1_prompt_text,
    hash_a1_prompt_text,
    parse_a1_candidates,
    render_a1_messages,
)
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
from attack_lab.reference_actions import ReferenceSelection
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import (
    DefenceDecision,
    InternalDefenceResult,
    PublicFeedback,
)
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
        decision: DefenceDecision = "BLOCK"
        return InternalDefenceResult(
            risk_score=0.9,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


@dataclass
class ScriptedLLMClient:
    """Deterministic LLM double; sequence items may be str or Exception."""

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
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
                "thinking_disabled": thinking_disabled,
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
    config = ReferencePoolConfig.load()
    return ReferencePoolProvider.from_config(
        config, training_frame=train
    ).get_pool(starting_case.case_id)


def _qm_budget(q_max: int, m_max: int) -> BudgetSpec:
    return BudgetSpec.development_dummy(
        q_max=q_max, m_max=m_max, label="dummy_a1_qm"
    )


def _make_env(
    *,
    starting_case,
    governance_policy,
    tmp_path: Path,
    budget: BudgetSpec,
    enabled: tuple[str, ...] | None,
    defender: CountingBlockDefender | None = None,
    reference_pool=None,
    require_reference_provenance: bool = True,
):
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=defender or CountingBlockDefender(),
        validator=ConstraintValidator.from_policy(
            governance_policy,
            enabled_action_keys=enabled,
            reference_pool=reference_pool,
            require_reference_provenance=require_reference_provenance
            and reference_pool is not None,
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=budget,
    )
    return env


def _ref_change(action: str, reference_id: str) -> dict[str, Any]:
    return {action: {"reference_id": reference_id}}


def _pick_ref_differing(
    reference_pool, starting_case, feature: str
) -> str:
    anchor = starting_case.features[feature]
    for profile in reference_pool.profiles:
        if profile.fields.get(feature) != anchor:
            return profile.profile_id
    return reference_pool.profiles[0].profile_id


def _valid_plan_response(
    starting_case,
    reference_pool,
    *,
    prompt_version: str = PROMPT_VERSION_V2,
    enabled: Sequence[str] | None = None,
    governance_policy=None,
    q_max: int = 5,
) -> str:
    """Build a K10-backed plan that shares one episode-static lock plan.

    Episode-static fields are omitted (freeze to anchor) so diversification
    uses only per-attempt fields.  That mirrors A1 freeze validation, where
    the first candidate locks every episode-static field for the whole plan.
    Reference choices are deduplicated by resolved field value so binary
    fields do not emit duplicate fingerprints.
    """

    def _refs_for(feature: str) -> list[str]:
        anchor = starting_case.features[feature]
        seen_values: set[str] = set()
        refs: list[str] = []
        for profile in reference_pool.profiles:
            value = profile.fields.get(feature)
            if value == anchor:
                continue
            key = repr(value)
            if key in seen_values:
                continue
            seen_values.add(key)
            refs.append(profile.profile_id)
        return refs

    preferred = (
        "income",
        "keep_alive_session",
        "payment_type",
        "customer_age",
    )
    actions = list(enabled) if enabled is not None else list(preferred)

    def _is_episode_locked(action: str) -> bool:
        if governance_policy is not None:
            rule = governance_policy.field_for_action(action)
            return bool(rule is not None and rule.is_episode_locked)
        # Synthetic governance fixture: customer_age is episode-static.
        return action == "customer_age"

    free_actions = [action for action in actions if not _is_episode_locked(action)]
    # Fall back to a shared static edit only when no per-attempt field is enabled.
    plan_actions = free_actions or [
        action for action in actions if _is_episode_locked(action)
    ][:1]

    items: list[dict[str, Any]] = []
    labels: list[str] = []
    for action in plan_actions:
        refs = _refs_for(action)
        if not refs:
            continue
        for idx, ref_id in enumerate(refs):
            if len(items) >= int(q_max):
                break
            suffix = "a" if idx == 0 else "b" if idx == 1 else str(idx)
            items.append({"changes": _ref_change(action, ref_id)})
            labels.append(f"{action}_shift_{suffix}")
        if len(items) >= int(q_max):
            break
    assert items, "synthetic pool must expose at least one differing reference"
    if prompt_version in {PROMPT_VERSION_V2, PROMPT_VERSION_V3}:
        items = [
            {"strategy_label": label, **item}
            for label, item in zip(labels, items, strict=True)
        ]
    return json.dumps({"candidates": items})


def test_parse_a1_candidates_ok_and_errors() -> None:
    ok, status = parse_a1_candidates('{"candidates":[{"changes":{"income":0.2}}]}')
    assert status == "ok"
    assert ok == [{"changes": {"income": 0.2}, "strategy_label": None}]
    assert parse_a1_candidates("")[1] == "empty"
    assert parse_a1_candidates("not-json")[1] == "parse_error"
    assert parse_a1_candidates('{"candidates":"bad"}')[1] == "schema_error"
    v2_ok, v2_status = parse_a1_candidates(
        '{"candidates":[{"strategy_label":"x","changes":{"income":0.2}}]}',
        prompt_version=PROMPT_VERSION_V2,
    )
    assert v2_status == "ok"
    assert v2_ok[0]["strategy_label"] == "x"
    assert (
        parse_a1_candidates(
            '{"candidates":[{"changes":{"income":0.2}}]}',
            prompt_version=PROMPT_VERSION_V2,
            m_max=2,
        )[1]
        == "schema_error"
    )
    assert (
        parse_a1_candidates(
            '{"candidates":[{"strategy_label":"x","changes":{"a":1,"b":2,"c":3}}]}',
            prompt_version=PROMPT_VERSION_V2,
            m_max=2,
        )[1]
        == "schema_error"
    )
    assert (
        parse_a1_candidates(
            '{"candidates":[{"strategy_label":"x","changes":{"a":1},"extra":1}]}',
            prompt_version=PROMPT_VERSION_V2,
            m_max=2,
        )[1]
        == "schema_error"
    )


def test_estimate_flash_cost_usd() -> None:
    cost = estimate_flash_cost_usd(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cached_tokens=0,
    )
    assert cost == pytest.approx(0.14 + 0.28)


def test_default_max_parse_retries_is_two() -> None:
    assert DEFAULT_MAX_PARSE_RETRIES == 2


def test_prompt_excludes_d1_internals(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget_spec = _qm_budget(5, 2)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget_spec,
        enabled=("income", "customer_age", "keep_alive_session", "payment_type"),
    )
    payload = build_a1_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
    )
    messages = render_a1_messages(payload)
    rendered = "\n".join(item["content"] for item in messages)
    assert "risk_score" not in payload["anchor"]["visible_fields"]
    assert "threshold" not in payload["anchor"]["visible_fields"]
    assert "fraud_bool" not in payload["anchor"]["visible_fields"]
    assert "explicitly_unavailable" in payload
    assert '"threshold"' not in rendered
    assert "feature_importance_or_shap" in payload["explicitly_unavailable"]


def test_freeze_before_any_defender_call(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    defender = CountingBlockDefender()
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
        defender=defender,
    )
    client = ScriptedLLMClient(responses=[_valid_plan_response(starting_case, reference_pool)])
    attacker = OneShotLLMPlanner(
        experiment_seed=123,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        model=DEFAULT_MODEL,
        thinking_disabled=True,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert defender.calls == 0
    assert len(client.calls) == 1
    assert client.calls[0]["thinking_disabled"] is True
    assert attacker.call_record is not None
    assert attacker.call_record.selected_response_index == 0
    assert attacker.call_record.llm_call_count == 1
    assert attacker.call_record.retry_count == 0
    assert (env.logger.run_dir / "a1_raw_response_attempt_0.txt").exists()
    assert (env.logger.run_dir / "a1_parsed_plan.json").exists()
    assert (env.logger.run_dir / "a1_retry_ledger.json").exists()
    assert (env.logger.run_dir / "a1_llm_call.json").exists()


def test_invalid_json_causes_one_retry(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    client = ScriptedLLMClient(
        responses=["not-json-{", _valid_plan_response(starting_case, reference_pool)]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=1,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 2
    assert attacker.call_record is not None
    assert attacker.call_record.retry_count == 1
    assert attacker.call_record.selected_response_index == 1
    assert attacker.call_record.retry_ledger[0].retry_reason == "parse_error"
    assert attacker.call_record.retry_ledger[0].selected_for_plan is False
    assert attacker.call_record.retry_ledger[1].selected_for_plan is True
    assert (env.logger.run_dir / "a1_raw_response_attempt_0.txt").exists()
    assert (env.logger.run_dir / "a1_raw_response_attempt_1.txt").exists()


def test_empty_response_causes_one_retry(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    client = ScriptedLLMClient(
        responses=["", _valid_plan_response(starting_case, reference_pool)]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=2,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 2
    assert attacker.call_record is not None
    assert attacker.call_record.retry_ledger[0].retry_reason == "empty"
    assert attacker.call_record.selected_response_index == 1


def test_parseable_governance_invalid_plan_regenerates_without_consuming_q(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age")
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
    q_before = env.ledger.q_remaining
    response = json.dumps(
        {
            "candidates": [
                {"changes": {"not_a_field": 1}},
                {"changes": {"also_bad": 2}},
            ]
        }
    )
    good = _valid_plan_response(
        starting_case,
        reference_pool,
        prompt_version=PROMPT_VERSION_V1,
        enabled=enabled,
        governance_policy=governance_policy,
    )
    client = ScriptedLLMClient(responses=[response, good])
    attacker = OneShotLLMPlanner(
        experiment_seed=3,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V1,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 2
    assert attacker.call_record is not None
    assert attacker.call_record.llm_call_count == 2
    assert attacker.call_record.retry_count == 1
    assert attacker.call_record.selected_response_index == 1
    assert attacker.call_record.retry_ledger[0].retry_reason == (
        "local_validation_failed"
    )
    assert "unknown_action" in client.calls[1]["messages"][1]["content"]
    assert env.ledger.q_remaining == q_before
    assert defender.calls == 0


def test_parseable_local_failures_trigger_regeneration_without_consuming_q(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Duplicate local failure regenerates; Q untouched."""
    enabled = ("income",)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    q_before = env.ledger.q_remaining
    income_refs = [
        p.profile_id
        for p in reference_pool.profiles
        if p.fields.get("income") != starting_case.features["income"]
    ]
    assert income_refs
    # Two identical parseable candidates → duplicate under require-all freeze.
    bad = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "dup_a",
                    "changes": {"income": {"reference_id": income_refs[0]}},
                },
                {
                    "strategy_label": "dup_b",
                    "changes": {"income": {"reference_id": income_refs[0]}},
                },
            ]
        }
    )
    good = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "income_only",
                    "changes": {"income": {"reference_id": income_refs[0]}},
                }
            ]
        }
    )
    client = ScriptedLLMClient(responses=[bad, good])
    attacker = OneShotLLMPlanner(
        experiment_seed=4,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V2,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 2
    assert attacker.call_record is not None
    assert attacker.call_record.retry_ledger[0].retry_reason == "local_validation_failed"
    assert attacker.call_record.selected_response_index == 1
    assert "duplicate" in client.calls[1]["messages"][1]["content"]
    assert env.ledger.q_remaining == q_before
    assert getattr(env.defender, "calls", 0) == 0


def test_after_first_successful_parse_llm_call_count_is_exactly_one(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    client = ScriptedLLMClient(
        responses=[
            _valid_plan_response(starting_case, reference_pool),
            _valid_plan_response(starting_case, reference_pool),
            _valid_plan_response(starting_case, reference_pool),
        ]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=5,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        max_parse_retries=2,
        prompt_version=PROMPT_VERSION_V2,
    )
    attacker.prepare_frozen_sequence(env)
    assert len(client.calls) == 1
    assert attacker.call_record is not None
    assert attacker.call_record.llm_call_count == 1


def test_raw_attempts_persisted_separately_and_not_overwritten(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    client = ScriptedLLMClient(
        responses=["", "not-json", _valid_plan_response(starting_case, reference_pool)]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=6,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        max_parse_retries=2,
        prompt_version=PROMPT_VERSION_V2,
    )
    attacker.prepare_frozen_sequence(env)
    p0 = env.logger.run_dir / "a1_raw_response_attempt_0.txt"
    p1 = env.logger.run_dir / "a1_raw_response_attempt_1.txt"
    p2 = env.logger.run_dir / "a1_raw_response_attempt_2.txt"
    assert p0.read_text(encoding="utf-8") == ""
    assert p1.read_text(encoding="utf-8") == "not-json"
    assert "candidates" in p2.read_text(encoding="utf-8")
    ledger = json.loads(
        (env.logger.run_dir / "a1_retry_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["selected_response_index"] == 2
    assert ledger["attempts"][0]["retry_reason"] == "empty"
    assert ledger["attempts"][1]["retry_reason"] == "parse_error"
    assert ledger["attempts"][2]["selected_for_plan"] is True


def test_selected_response_index_null_when_all_parse_fail(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income",)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    client = ScriptedLLMClient(responses=["", "not-json", '{"candidates":"bad"}'])
    attacker = OneShotLLMPlanner(
        experiment_seed=8,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        max_parse_retries=2,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen == ()
    assert len(client.calls) == 3
    assert attacker.call_record is not None
    assert attacker.call_record.selected_response_index is None
    assert attacker.call_record.retry_count == 2


def test_feedback_does_not_change_frozen_sequence(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    response = _valid_plan_response(starting_case, reference_pool)

    def collect(poison: bool) -> list[dict[str, Any]]:
        env = _make_env(
            starting_case=starting_case,
            governance_policy=governance_policy,
            reference_pool=reference_pool,
            tmp_path=tmp_path / f"fb_{int(poison)}",
            budget=_qm_budget(5, 2),
            enabled=enabled,
        )
        attacker = OneShotLLMPlanner(
            experiment_seed=99,
            reference_pool=reference_pool,
            budget=AttackBudget(q_max=5, m_max=2),
            llm_client=ScriptedLLMClient(responses=[response]),
        prompt_version=PROMPT_VERSION_V2,
        )
        frozen = attacker.prepare_frozen_sequence(env)
        proposals = [dict(item.changes) for item in frozen]
        for proposal in frozen:
            if env.done:
                break
            env.step(proposal)
            if poison:
                env._last_feedback = PublicFeedback(  # noqa: SLF001
                    label="PASS",
                    message="poison",
                    attempt=env.attempts_used,
                    remaining_attempts=0,
                )
        env2 = _make_env(
            starting_case=starting_case,
            governance_policy=governance_policy,
            reference_pool=reference_pool,
            tmp_path=tmp_path / f"fb2_{int(poison)}",
            budget=_qm_budget(5, 2),
            enabled=enabled,
        )
        again = OneShotLLMPlanner(
            experiment_seed=99,
            reference_pool=reference_pool,
            budget=AttackBudget(q_max=5, m_max=2),
            llm_client=ScriptedLLMClient(responses=[response]),
        prompt_version=PROMPT_VERSION_V2,
        )
        assert [
            dict(item.changes) for item in again.prepare_frozen_sequence(env2)
        ] == proposals
        return proposals

    assert collect(False) == collect(True)


def test_run_episode_submits_frozen_only(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    defender = CountingBlockDefender()
    logger = TrajectoryLogger(run_dir=tmp_path / "match", run_id="a1")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    attacker = OneShotLLMPlanner(
        experiment_seed=42,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=3, m_max=2),
        llm_client=ScriptedLLMClient(responses=[_valid_plan_response(starting_case, reference_pool)]),
        stdout=io.StringIO(),
        prompt_version=PROMPT_VERSION_V2,
    )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a1",
            anchor=starting_case,
            policy=governance_policy,
            budget=_qm_budget(3, 2),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=defender,
            seed=42,
            enabled_action_keys=enabled,
            logger=logger,
            reference_pool=reference_pool,
            require_reference_provenance=True,
        ),
    )
    assert match.attacker_id == "a1"
    assert defender.calls == match.q_used
    assert match.q_used <= 3
    assert attacker.call_record is not None
    assert attacker.call_record.retry_count == 0
    assert attacker.call_record.llm_call_count == 1


def test_v2_prompt_version_and_diversification_clause(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    payload = build_a1_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        prompt_version=PROMPT_VERSION_V2,
    )
    assert payload["prompt_version"] == PROMPT_VERSION_V2
    assert payload["planning_principle"] == DIVERSIFICATION_PRINCIPLE_V2
    assert "materially different hypotheses" in payload["planning_principle"]
    item_schema = payload["output_schema"]["properties"]["candidates"]["items"]
    assert "strategy_label" in item_schema["required"]
    assert "changes" in item_schema["required"]

    messages = render_a1_messages(payload)
    text = format_a1_prompt_text(messages)
    assert PROMPT_VERSION_V2 in text
    assert DIVERSIFICATION_PRINCIPLE_V2 in text
    # New instructional text must not name preferred action fields or success lore.
    principle = payload["planning_principle"].lower()
    for banned in ("income", "proposed_credit_limit"):
        assert banned not in principle
    # May name hidden quantities only to forbid targeting them.
    assert "do not infer or target any hidden model score, threshold" in principle
    assert "risk_score" not in payload["anchor"]["visible_fields"]
    assert "threshold" not in payload["anchor"]["visible_fields"]
    assert "fraud_bool" not in payload["anchor"]["visible_fields"]


def test_v2_one_call_freeze_and_prompt_hash_persisted(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    defender = CountingBlockDefender()
    enabled = ("income", "customer_age", "keep_alive_session", "payment_type")
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
        defender=defender,
    )
    client = ScriptedLLMClient(
        responses=[
            _valid_plan_response(starting_case, reference_pool, prompt_version=PROMPT_VERSION_V2),
            _valid_plan_response(starting_case, reference_pool, prompt_version=PROMPT_VERSION_V2),
        ]
    )
    attacker = OneShotLLMPlanner(
        experiment_seed=77,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        prompt_version=PROMPT_VERSION_V2,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 1
    assert defender.calls == 0
    assert attacker.call_record is not None
    assert attacker.call_record.prompt_version == PROMPT_VERSION_V2
    assert attacker.call_record.llm_call_count == 1
    assert len(attacker.call_record.prompt_hash) == 64
    prompt_path = env.logger.run_dir / "a1_prompt_full.txt"
    hash_path = env.logger.run_dir / "a1_prompt_hash.txt"
    assert prompt_path.exists()
    assert hash_path.exists()
    assert hash_a1_prompt_text(prompt_path.read_text(encoding="utf-8")) == (
        hash_path.read_text(encoding="utf-8").strip()
    )
    assert all(
        item.research_meta.get("strategy_label") for item in frozen
    )
    assert all(
        item.research_meta.get("prompt_version") == PROMPT_VERSION_V2 for item in frozen
    )


def test_v1_default_unchanged_without_diversification_clause(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "customer_age"),
    )
    payload = build_a1_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        prompt_version=PROMPT_VERSION_V1,
    )
    assert payload["prompt_version"] == PROMPT_VERSION_V1
    assert "planning_principle" not in payload
    item_schema = payload["output_schema"]["properties"]["candidates"]["items"]
    assert item_schema["required"] == ["changes"]


def test_formal_model_config_frozen_values_and_hash_persisted(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    cfg = FORMAL_A1_MODEL_CONFIG
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.thinking_disabled is True
    assert cfg.temperature == 0.0
    assert cfg.top_p == 1.0
    assert cfg.max_tokens == 800
    assert cfg.max_parse_retries == 2
    assert cfg.timeout_seconds == 90.0
    assert cfg.prompt_version == PROMPT_VERSION_V4
    assert DEFAULT_MAX_TOKENS == 800
    assert DEFAULT_TOP_P == 1.0
    assert DEFAULT_TIMEOUT_SECONDS == 90.0
    assert DEFAULT_TEMPERATURE == 0.0

    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "customer_age", "keep_alive_session", "payment_type"),
    )
    # Formal default is V4; keep historical V3 explicitly selectable elsewhere.
    from attack_lab.archive.contracts.a1_v4_contract import build_v4_choice_catalog, build_v4_static_plan_options

    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=starting_case.features,
    )
    plans = build_v4_static_plan_options(
        validator=env.validator,
        pool=reference_pool,
        anchor=starting_case.features,
        catalog=catalog,
        m_max=2,
        q_max=5,
    )
    assert plans
    plan = plans[0]
    query_ids = list(plan.allowed_query_choice_ids)[:5]
    # Need 5 distinct single choices when residual_m >= 1
    assert len(query_ids) >= 5
    response = json.dumps(
        {
            "static_plan_id": plan.static_plan_id,
            "candidates": [
                {
                    "strategy_label": f"c{i}",
                    "choice_ids": [query_ids[i]],
                }
                for i in range(5)
            ],
        }
    )
    client = ScriptedLLMClient(responses=[response])
    attacker = OneShotLLMPlanner(
        experiment_seed=11,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V4,
    )
    attacker.prepare_frozen_sequence(env)
    assert attacker.model_config.to_dict() == cfg.to_dict()
    assert attacker.call_record is not None
    assert attacker.call_record.config_hash == cfg.config_hash()
    assert (env.logger.run_dir / "model_config.json").exists()
    assert (env.logger.run_dir / "a1_config_hash.txt").read_text(
        encoding="utf-8"
    ).strip() == cfg.config_hash()
    assert client.calls[0]["top_p"] == 1.0
    assert client.calls[0]["timeout_seconds"] == 90.0
    assert client.calls[0]["max_tokens"] == 800
    assert client.calls[0]["temperature"] == 0.0

    item_schema = build_a1_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        prompt_version=PROMPT_VERSION_V2,
    )["output_schema"]["properties"]["candidates"]["items"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["changes"]["minProperties"] == 1
    assert item_schema["properties"]["changes"]["maxProperties"] == 2

def test_default_max_local_generation_attempts_is_three() -> None:
    assert DEFAULT_MAX_LOCAL_GENERATION_ATTEMPTS == 3
    assert DEFAULT_MAX_LOCAL_GENERATION_ATTEMPTS == DEFAULT_MAX_PARSE_RETRIES + 1


def test_unknown_reference_id_regenerates_without_consuming_q(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income",)
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
    q_before = env.ledger.q_remaining
    income_refs = [
        p.profile_id
        for p in reference_pool.profiles
        if p.fields.get("income") != starting_case.features["income"]
    ]
    bad = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "bogus",
                    "changes": {"income": {"reference_id": "ref_99"}},
                }
            ]
        }
    )
    good = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "ok",
                    "changes": {"income": {"reference_id": income_refs[0]}},
                }
            ]
        }
    )
    client = ScriptedLLMClient(responses=[bad, good])
    attacker = OneShotLLMPlanner(
        experiment_seed=40,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 2
    assert "LOCAL RULE-COMPLIANCE REPAIR" in client.calls[1]["messages"][1]["content"]
    assert "unknown_reference_id" in client.calls[1]["messages"][1]["content"]
    assert env.ledger.q_remaining == q_before == 5
    assert defender.calls == 0


def test_provenance_invalid_literal_regenerates_without_consuming_q(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income",)
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
    q_before = env.ledger.q_remaining
    income_refs = [
        p.profile_id
        for p in reference_pool.profiles
        if p.fields.get("income") != starting_case.features["income"]
    ]
    bad = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "literal",
                    "changes": {"income": 0.123456789},
                }
            ]
        }
    )
    good = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "ok",
                    "changes": {"income": {"reference_id": income_refs[0]}},
                }
            ]
        }
    )
    client = ScriptedLLMClient(responses=[bad, good])
    attacker = OneShotLLMPlanner(
        experiment_seed=41,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 2
    assert "not_reference_selection" in client.calls[1]["messages"][1]["content"]
    assert env.ledger.q_remaining == q_before
    assert defender.calls == 0
    assert all(
        isinstance(v, ReferenceSelection) for v in frozen[0].changes.values()
    )


def test_m_budget_invalid_plan_regenerates_without_consuming_q(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "customer_age", "payment_type")
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 1),
        enabled=enabled,
        defender=defender,
    )
    q_before = env.ledger.q_remaining
    income_ref = _pick_ref_differing(reference_pool, starting_case, "income")
    age_ref = _pick_ref_differing(reference_pool, starting_case, "customer_age")
    payment_ref = _pick_ref_differing(reference_pool, starting_case, "payment_type")
    # m_max=1 but two edits -> budget_exceeded locally
    bad = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "too_many",
                    "changes": {
                        "income": {"reference_id": income_ref},
                        "customer_age": {"reference_id": age_ref},
                    },
                }
            ]
        }
    )
    good = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "ok",
                    "changes": {"payment_type": {"reference_id": payment_ref}},
                }
            ]
        }
    )
    client = ScriptedLLMClient(responses=[bad, good])
    attacker = OneShotLLMPlanner(
        experiment_seed=42,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=1),
        prompt_version=PROMPT_VERSION_V1,
        llm_client=client,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(client.calls) == 2
    assert "budget_exceeded" in client.calls[1]["messages"][1]["content"]
    assert env.ledger.q_remaining == q_before
    assert defender.calls == 0


def test_local_failure_never_reaches_d1_and_plan_frozen_before_first_query(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income",)
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
    income_refs = [
        p.profile_id
        for p in reference_pool.profiles
        if p.fields.get("income") != starting_case.features["income"]
    ]
    bad = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "bogus",
                    "changes": {"income": {"reference_id": "ref_99"}},
                }
            ]
        }
    )
    good = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "ok",
                    "changes": {"income": {"reference_id": income_refs[0]}},
                }
            ]
        }
    )
    client = ScriptedLLMClient(responses=[bad, good])
    attacker = OneShotLLMPlanner(
        experiment_seed=43,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert defender.calls == 0
    assert env.attempts_used == 0
    # Submit frozen plan: only then D1 may run.
    attacker.run(env)
    assert defender.calls >= 1
    assert attacker._sequence_prepared is True  # noqa: SLF001


def test_pass_block_feedback_never_replans_a1(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income",)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    income_refs = [
        p.profile_id
        for p in reference_pool.profiles
        if p.fields.get("income") != starting_case.features["income"]
    ][:3]
    plan = {
        "candidates": [
            {
                "strategy_label": f"c{i}",
                "changes": {"income": {"reference_id": rid}},
            }
            for i, rid in enumerate(income_refs)
        ]
    }
    client = ScriptedLLMClient(responses=[json.dumps(plan), json.dumps(plan)])
    attacker = OneShotLLMPlanner(
        experiment_seed=44,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = list(attacker.prepare_frozen_sequence(env))
    assert len(client.calls) == 1
    fingerprints = [p.research_meta["candidate_fingerprint"] for p in frozen]
    for proposal in frozen:
        if env.done:
            break
        env.step(proposal)
        # Poison feedback must not cause regeneration.
        env._last_feedback = PublicFeedback(  # noqa: SLF001
            label="PASS",
            message="poison",
            attempt=env.attempts_used,
            remaining_attempts=0,
        )
    again = OneShotLLMPlanner(
        experiment_seed=44,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        prompt_version=PROMPT_VERSION_V2,
    )
    env2 = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "again",
        budget=_qm_budget(5, 2),
        enabled=enabled,
    )
    frozen2 = again.prepare_frozen_sequence(env2)
    assert [p.research_meta["candidate_fingerprint"] for p in frozen2] == fingerprints
    # Original client should not be called again for the first attacker after freeze.
    assert len(client.calls) == 2  # second planner instance only


def test_local_generation_exhaustion_leaves_q_untouched(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income",)
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
    q_before = env.ledger.q_remaining
    bad = json.dumps(
        {
            "candidates": [
                {
                    "strategy_label": "bogus",
                    "changes": {"income": {"reference_id": "ref_99"}},
                }
            ]
        }
    )
    client = ScriptedLLMClient(responses=[bad, bad, bad, bad])
    attacker = OneShotLLMPlanner(
        experiment_seed=45,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
        max_parse_retries=2,
        max_local_generation_attempts=3,
        prompt_version=PROMPT_VERSION_V2,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen == ()
    assert len(client.calls) == 3
    assert attacker._pending_stop_reason == "local_generation_exhausted"  # noqa: SLF001
    assert env.ledger.q_remaining == q_before == 5
    assert defender.calls == 0


