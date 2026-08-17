"""Focused unit tests for A3 EpisodicLLMAgent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from attack_lab.attackers.a1_planner import LLMCompletion
from attack_lab.attackers.a3_agent import (
    FORMAL_A3_MODEL_CONFIG,
    PROMPT_VERSION,
    PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
    PROMPT_VERSION_B2_GROUNDED_REFLECTION,
    PROMPT_VERSION_P1_COMPACT,
    PROMPT_VERSION_P1_RANKED_PORTFOLIO,
    PROMPT_VERSION_P2_NOVELTY,
    RANKED_PORTFOLIO_CAP,
    A3AgentError,
    A3MemoryStep,
    A3ModelConfig,
    EpisodicLLMAgent,
    build_a3_prompt_payload,
    build_a3_rendered_prompt_context,
    compute_a3_edit_slot_accounting,
    parse_a3_candidate,
    parse_a3_ranked_portfolio,
    render_a3_messages,
)
from attack_lab.attacker_interface import AttackerEpisode
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import (
    AttackProposal,
    DefenceDecision,
    InternalDefenceResult,
    to_jsonable,
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
        return InternalDefenceResult(
            risk_score=0.9,
            threshold=self.threshold,
            decision="BLOCK",
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


@dataclass
class PassOnSecondDefender:
    name: str = "pass_on_second"
    artefact_id: str = "test"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult:
        self.calls += 1
        decision: DefenceDecision = "PASS" if self.calls >= 2 else "BLOCK"
        return InternalDefenceResult(
            risk_score=0.2 if decision == "PASS" else 0.9,
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
        q_max=q_max, m_max=m_max, label="dummy_a3_qm"
    )


def _make_env(
    *,
    starting_case,
    governance_policy,
    tmp_path: Path,
    budget: BudgetSpec,
    enabled: tuple[str, ...] | None,
    defender=None,
):
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    return AttackEnvironment(
        starting_case=starting_case,
        defender=defender or CountingBlockDefender(),
        validator=ConstraintValidator.from_policy(
            governance_policy, enabled_action_keys=enabled
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=budget,
    )


def _candidate_json(
    changes: Mapping[str, Any],
    *,
    strategy_label: str = "probe",
    adaptation_note: str = "try alternate fields",
) -> str:
    return json.dumps(
        {
            "strategy_label": strategy_label,
            "changes": dict(changes),
            "adaptation_note": adaptation_note,
        }
    )


def _portfolio_json(*candidates: Mapping[str, Any]) -> str:
    return json.dumps({"candidates": [dict(item) for item in candidates]})


def _candidate(
    changes: Mapping[str, Any],
    *,
    strategy_label: str,
    adaptation_note: str = "ranked local alternative",
) -> dict[str, Any]:
    return {
        "strategy_label": strategy_label,
        "changes": dict(changes),
        "adaptation_note": adaptation_note,
    }


def _different_income(starting_case) -> float:
    current = float(starting_case.features["income"])
    return 0.2 if abs(current - 0.2) > 1e-12 else 0.3


def _ranked_agent(*, reference_pool, budget: AttackBudget, client, seed: int = 101):
    return EpisodicLLMAgent(
        experiment_seed=seed,
        reference_pool=reference_pool,
        budget=budget,
        prompt_version=PROMPT_VERSION_P1_RANKED_PORTFOLIO,
        max_parse_retries=0,
        max_local_generation_attempts_per_query=1,
        portfolio_cap=RANKED_PORTFOLIO_CAP,
        llm_client=client,
    )


def test_parse_requires_single_candidate_schema() -> None:
    ok, status = parse_a3_candidate(
        _candidate_json({"income": 0.2}), m_max=2
    )
    assert status == "ok"
    assert ok is not None
    assert set(ok) == {"strategy_label", "changes", "adaptation_note"}
    assert (
        parse_a3_candidate(
            '{"candidates":[{"strategy_label":"x","changes":{"income":0.2},'
            '"adaptation_note":"n"}]}',
            m_max=2,
        )[1]
        == "schema_error"
    )
    assert (
        parse_a3_candidate(
            '{"strategy_label":"x","changes":{"a":1,"b":2,"c":3},'
            '"adaptation_note":"n"}',
            m_max=2,
        )[1]
        == "schema_error"
    )
    assert (
        parse_a3_candidate(
            '{"strategy_label":"x","changes":{"income":0.2},'
            '"adaptation_note":"n","extra":1}',
            m_max=2,
        )[1]
        == "schema_error"
    )


def test_one_llm_call_produces_one_candidate_only(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="income_shift",
            )
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "customer_age", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    # Force early stop after one BLOCK by using q_max=1
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path / "one",
        budget=_qm_budget(1, 2),
        enabled=("income", "customer_age", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=1,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 1
    assert len(attacker.query_records) == 1
    assert attacker.query_records[0].submitted is True
    prompt = client.calls[0]["messages"][1]["content"]
    assert "exactly one" in prompt.lower() or "exactly one JSON" in prompt
    assert '"candidates"' not in json.dumps(
        {"strategy_label": "income_shift", "changes": {"income": 0.1}}
    )  # sanity
    # Response schema is a single object — no portfolio key in parsed artefact
    parsed = json.loads(
        (
            env.logger.run_dir
            / "query_01"
            / "local_gen_01"
            / "a3_parsed_candidate.json"
        ).read_text()
    )
    assert "candidates" not in parsed
    assert set(parsed) == {"strategy_label", "changes", "adaptation_note"}


def test_block_permits_later_new_llm_call(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    keep = int(starting_case.features["keep_alive_session"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="income_first",
                adaptation_note="start with income",
            ),
            _candidate_json(
                {"keep_alive_session": 1 - keep},
                strategy_label="session_second",
                adaptation_note="switch after block",
            ),
        ]
    )
    defender = PassOnSecondDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session", "email_is_free"),
        defender=defender,
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=2,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 2
    assert env.success is True
    assert attacker.memory_steps[0].public_label == "BLOCK"
    assert attacker.memory_steps[1].public_label == "PASS"


def test_pass_stops_episode_immediately(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])

    @dataclass
    class AlwaysPass:
        name: str = "always_pass"
        artefact_id: str = "test"
        threshold: float = 0.5
        calls: int = field(default=0, init=False)

        def score_application(self, features: Mapping[str, Any]) -> InternalDefenceResult:
            self.calls += 1
            return InternalDefenceResult(
                risk_score=0.1,
                threshold=self.threshold,
                decision="PASS",
                runtime_ms=0.01,
                defender_name=self.name,
                artefact_id=self.artefact_id,
            )

    client = ScriptedLLMClient(
        responses=[
            _candidate_json({"income": min(0.99, income + 0.05)}),
            _candidate_json({"income": min(0.99, income + 0.10)}),
        ]
    )
    defender = AlwaysPass()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "customer_age"),
        defender=defender,
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=3,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert env.success is True
    assert defender.calls == 1
    assert len(client.calls) == 1
    assert env.result().stop_reason == "success"


def test_q_max_cap_respected(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    keep = int(starting_case.features["keep_alive_session"])
    email = int(starting_case.features["email_is_free"])
    responses = [
        _candidate_json(
            {"income": min(0.99, income + 0.05)},
            strategy_label="s0",
        ),
        _candidate_json(
            {"keep_alive_session": 1 - keep}, strategy_label="s1"
        ),
        _candidate_json(
            {"email_is_free": 1 - email}, strategy_label="s2"
        ),
    ]
    client = ScriptedLLMClient(responses=responses)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=("income", "keep_alive_session", "email_is_free"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=4,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 2
    assert env.ledger.q_used == 2
    assert env.done is True


def test_prior_block_and_strategy_appear_in_next_memory(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    keep = int(starting_case.features["keep_alive_session"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="income_probe",
                adaptation_note="first probe",
            ),
            _candidate_json(
                {"keep_alive_session": 1 - keep},
                strategy_label="session_probe",
                adaptation_note="adapt after block",
            ),
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=5,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 2
    second_user = client.calls[1]["messages"][1]["content"]
    assert "income_probe" in second_user
    assert '"public_label": "BLOCK"' in second_user or '"public_label":"BLOCK"' in second_user.replace(
        " ", ""
    )
    assert "income" in second_user
    mem = json.loads(
        (env.logger.run_dir / "query_02" / "a3_memory_before.json").read_text()
    )
    assert mem["episode_memory"][0]["strategy_label"] == "income_probe"
    assert mem["episode_memory"][0]["public_label"] == "BLOCK"


def test_no_d1_internals_in_prompts(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[_candidate_json({"income": min(0.99, income + 0.05)})]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=("income", "customer_age"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=6,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    blob = json.dumps(client.calls[0]["messages"]).lower()
    for forbidden in (
        "risk_score",
        "threshold",
        "feature_importance",
        "shap",
        "gradient",
        "fraud_bool",
        "d1_risk_score",
    ):
        # explicitly_unavailable may list names; ensure they are not values/inputs
        assert f'"{forbidden}":' not in blob.replace(" ", "")


def test_local_reject_then_valid_same_query_consumes_q_once(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": income},
                strategy_label="same",
                adaptation_note="zero edit",
            ),
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="repaired",
                adaptation_note="pre-submission repair",
            ),
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=("income", "keep_alive_session"),
        defender=defender,
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=7,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 2
    assert defender.calls == 1
    assert env.ledger.q_used == 1
    assert attacker.total_env_steps == 1
    assert attacker.total_local_rejections == 1
    assert attacker.total_local_regenerations == 1
    assert attacker.query_records[0].submitted is True
    assert attacker.query_records[0].public_label == "BLOCK"
    assert all(step.public_label != "LOCAL_REJECT" for step in attacker.memory_steps)
    assert all(step.public_label != "BLOCK" or step.submitted for step in attacker.memory_steps)
    # Repair context must not be framed as D1 BLOCK.
    second_prompt = client.calls[1]["messages"][1]["content"]
    assert "local_proposal_repair" in second_prompt
    assert "not a D1 BLOCK" in second_prompt
    assert '"public_label": "BLOCK"' not in second_prompt.split("episode_memory")[0]


def test_two_local_rejects_then_valid_three_llm_one_step(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json({"income": income}, strategy_label="r1", adaptation_note="a"),
            _candidate_json({"income": income}, strategy_label="r2", adaptation_note="b"),
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="ok",
                adaptation_note="c",
            ),
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=("income", "keep_alive_session"),
        defender=defender,
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=71,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 3
    assert defender.calls == 1
    assert env.ledger.q_used == 1
    assert attacker.total_env_steps == 1
    assert attacker.total_local_rejections == 2
    assert attacker.total_local_regenerations == 2


def test_all_three_local_rejects_exhaust_without_step(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json({"income": income}, strategy_label="r1", adaptation_note="a"),
            _candidate_json({"income": income}, strategy_label="r2", adaptation_note="b"),
            _candidate_json({"income": income}, strategy_label="r3", adaptation_note="c"),
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="should_not_run",
                adaptation_note="d",
            ),
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session"),
        defender=defender,
    )
    q_before = env.ledger.q_remaining
    attacker = EpisodicLLMAgent(
        experiment_seed=72,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 3
    assert defender.calls == 0
    assert attacker.total_env_steps == 0
    assert env.ledger.q_remaining == q_before
    assert env.ledger.q_used == 0
    assert env.result().stop_reason == "local_generation_exhausted"
    assert attacker.total_regeneration_exhaustions == 1
    assert attacker.memory_steps == ()
    assert attacker.query_records[0].regeneration_exhausted is True
    assert attacker.query_records[0].public_label is None


def test_block_does_not_grant_free_local_regeneration(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    keep = int(starting_case.features["keep_alive_session"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="q1",
                adaptation_note="first",
            ),
            _candidate_json(
                {"keep_alive_session": 1 - keep},
                strategy_label="q2",
                adaptation_note="after block",
            ),
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=73,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 2
    assert env.ledger.q_used == 2
    assert attacker.total_env_steps == 2
    assert attacker.total_local_regenerations == 0
    assert attacker.query_records[0].query_index == 1
    assert attacker.query_records[1].query_index == 2
    assert attacker.memory_steps[0].public_label == "BLOCK"
    assert attacker.memory_steps[0].submitted is True


def test_submitted_invalid_charges_q(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Force a post-step INVALID; local validation is bypassed only in this test."""

    class ForceInvalidSubmitAgent(EpisodicLLMAgent):
        def _build_proposal(  # type: ignore[override]
            self,
            env,
            *,
            raw_changes,
            strategy_label,
            adaptation_note,
            query_index,
            prompt_hash,
        ):
            proposal = AttackProposal(
                changes={"income": 999.0},
                raw_command="force-invalid",
                research_meta={
                    "strategy_label": strategy_label,
                    "adaptation_note": adaptation_note,
                    "prompt_hash": prompt_hash,
                    "query_index": query_index,
                },
            )
            return proposal, None, ("income",), "forced-fingerprint"

    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": 0.2},
                strategy_label="force",
                adaptation_note="will be invalid at env",
            )
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = ForceInvalidSubmitAgent(
        experiment_seed=74,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert attacker.total_env_steps == 1
    assert env.ledger.q_used == 1
    assert attacker.query_records[0].public_label == "INVALID"
    assert attacker.memory_steps[0].public_label == "INVALID"
    assert attacker.memory_steps[0].submitted is True


def test_api_parse_failure_may_retry_and_is_capped(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[
            "not-json",
            "",
            _candidate_json({"income": min(0.99, income + 0.05)}),
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=("income", "customer_age"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=8,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        max_parse_retries=2,
        llm_client=client,
    )
    attacker.run(env)
    assert len(client.calls) == 3  # initial + 2 retries
    assert attacker.query_records[0].retry_count == 2
    assert attacker.query_records[0].parse_status == "ok"

    client2 = ScriptedLLMClient(responses=["bad", "also-bad", "still-bad", "extra"])
    env2 = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path / "cap",
        budget=_qm_budget(1, 2),
        enabled=("income", "customer_age"),
        defender=CountingBlockDefender(),
    )
    attacker2 = EpisodicLLMAgent(
        experiment_seed=9,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        max_parse_retries=2,
        max_local_generation_attempts_per_query=1,
        llm_client=client2,
    )
    attacker2.run(env2)
    assert len(client2.calls) == 3  # capped at max_parse_retries+1 within one local gen
    assert attacker2.query_records[0].parse_status in {
        "parse_error",
        "schema_error",
        "empty",
    }


def test_adaptation_does_not_alter_episode_static_locks(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    age = int(starting_case.features["customer_age"])
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"customer_age": age + 1},
                strategy_label="lock_age",
                adaptation_note="set static age",
            ),
            _candidate_json(
                {"customer_age": age + 2, "income": min(0.99, income + 0.05)},
                strategy_label="break_lock",
                adaptation_note="illegal static change",
            ),
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "customer_age", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=10,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    # q1 submits BLOCK; q2 tries up to 3 local repairs, all static_field_changed.
    assert len(client.calls) == 1 + 3
    assert len(attacker.memory_steps) == 1
    assert attacker.memory_steps[0].public_label == "BLOCK"
    assert attacker.memory_steps[0].submitted is True
    assert attacker.query_records[1].submitted is False
    assert attacker.query_records[1].regeneration_exhausted is True
    assert attacker.total_local_rejections == 3
    assert all(
        rec.local_rejection_reason == "static_field_changed"
        for rec in attacker.query_records[1].local_generation_records
    )
    # Local rejection must never appear as a D1 BLOCK in episodic memory.
    assert not any(step.public_label == "LOCAL_REJECT" for step in attacker.memory_steps)


def test_formal_config_defaults_and_orchestrator(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    cfg = FORMAL_A3_MODEL_CONFIG
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.thinking_disabled is True
    assert cfg.temperature == 0.0
    assert cfg.top_p == 1.0
    assert cfg.max_tokens == 800
    assert cfg.max_parse_retries == 2
    assert cfg.timeout_seconds == 90.0
    assert cfg.prompt_version == PROMPT_VERSION
    assert cfg.max_local_generation_attempts_per_query == 3
    assert cfg.prompt_version == "a3_episodic_v2"
    # State-representation / repair semantics must not reuse older config hashes.
    assert cfg.config_hash() not in {
        "43f6612a13c891352737b75b2125b3d4040d2a94d365ad0d250e6735aef808fb",
        "fec474c00678a52c0cb21da173ca019a9c19b19e914e3ae7140a2a30f223f190",
    }

    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[_candidate_json({"income": min(0.99, income + 0.05)})]
    )
    logger = TrajectoryLogger(run_dir=tmp_path / "match", run_id="a3")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    attacker = EpisodicLLMAgent(
        experiment_seed=11,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        llm_client=client,
    )
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a3",
            anchor=starting_case,
            policy=governance_policy,
            budget=AttackBudget(q_max=1, m_max=2).to_budget_spec(label="a3"),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=CountingBlockDefender(),
            seed=11,
            enabled_action_keys=("income", "customer_age"),
            logger=logger,
            reference_pool=reference_pool,
        ),
    )
    assert match.stop_reason in {
        "q_exhausted",
        "success",
        "local_generation_exhausted",
    }
    assert attacker.config_hash == cfg.config_hash()
    assert (logger.run_dir / "query_01" / "model_config.json").exists()
    researcher_only = json.loads(
        logger.researcher_only_diagnostics_path.read_text(encoding="utf-8")
    )
    assert researcher_only["access"] == "RESEARCHER_ONLY"
    assert researcher_only["case_id"] == starting_case.case_id
    assert researcher_only["steps"][0]["public_decision"] == "BLOCK"
    assert researcher_only["steps"][0]["internal_defence"]["risk_score"] == 0.9
    public_text = logger.public_transcript_path.read_text(encoding="utf-8")
    assert "risk_score" not in public_text


@pytest.mark.parametrize(
    ("proxy_action_key", "hidden_target"),
    (
        ("name_email_alignment", "name_email_similarity"),
        ("home_phone_configuration", "phone_home_valid"),
        ("mobile_phone_configuration", "phone_mobile_valid"),
    ),
)
def test_formal_path_opaque_assessment_executes_changed_proxy_actions(
    proxy_action_key,
    hidden_target,
    starting_case,
    governance_policy,
    reference_pool,
    tmp_path,
) -> None:
    rule = governance_policy.field_for_action(proxy_action_key)
    assert rule is not None
    action_name = next(
        name
        for name, resolved in rule.resolved_proxy_actions.items()
        if resolved != starting_case.features[hidden_target]
    )
    defender = CountingBlockDefender()
    attacker = EpisodicLLMAgent(
        experiment_seed=11,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        prompt_version=PROMPT_VERSION_P1_COMPACT,
        llm_client=ScriptedLLMClient(
            responses=[_candidate_json({proxy_action_key: action_name})]
        ),
    )
    logger = TrajectoryLogger(
        run_dir=tmp_path / f"formal_changed_{proxy_action_key}",
        run_id="a3_proxy_changed",
    )
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a3",
            anchor=starting_case,
            policy=governance_policy,
            budget=_qm_budget(1, 2),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=defender,
            seed=11,
            enabled_action_keys=(proxy_action_key,),
            logger=logger,
            reference_pool=reference_pool,
        ),
    )

    assert defender.calls == 1
    assert match.q_used == 1
    assert match.scored_defender_queries == 1
    assert attacker.memory_steps[0].edited_fields == (proxy_action_key,)
    assert attacker.query_records[0].local_rejections == 0
    assert attacker.query_records[0].env_step_called is True
    assert hidden_target not in json.dumps(
        to_jsonable(attacker.memory_steps[0]), sort_keys=True
    )


@pytest.mark.parametrize(
    ("proxy_action_key", "hidden_target"),
    (
        ("name_email_alignment", "name_email_similarity"),
        ("home_phone_configuration", "phone_home_valid"),
        ("mobile_phone_configuration", "phone_mobile_valid"),
    ),
)
def test_formal_path_opaque_assessment_rejects_unchanged_proxy_without_side_effects(
    proxy_action_key,
    hidden_target,
    starting_case,
    governance_policy,
    reference_pool,
    tmp_path,
) -> None:
    rule = governance_policy.field_for_action(proxy_action_key)
    assert rule is not None
    action_name, resolved_value = next(iter(rule.resolved_proxy_actions.items()))
    unchanged_case = replace(
        starting_case,
        features={**starting_case.features, hidden_target: resolved_value},
    )
    defender = CountingBlockDefender()
    client = ScriptedLLMClient(
        responses=[_candidate_json({proxy_action_key: action_name})]
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=12,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        prompt_version=PROMPT_VERSION_P1_COMPACT,
        max_local_generation_attempts_per_query=3,
        llm_client=client,
    )
    logger = TrajectoryLogger(
        run_dir=tmp_path / f"formal_unchanged_{proxy_action_key}",
        run_id="a3_proxy_unchanged",
    )
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    match = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a3",
            anchor=unchanged_case,
            policy=governance_policy,
            budget=_qm_budget(1, 2),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=defender,
            seed=12,
            enabled_action_keys=(proxy_action_key,),
            logger=logger,
            reference_pool=reference_pool,
        ),
    )

    assert defender.calls == 0
    assert match.q_used == 0
    assert match.scored_defender_queries == 0
    assert attacker.memory_steps == ()
    assert len(client.calls) == 3
    assert [
        item.local_rejection_reason
        for item in attacker.query_records[0].local_generation_records
    ] == ["same_as_anchor", "same_as_anchor", "same_as_anchor"]
    assert match.stop_reason == "local_generation_exhausted"


def test_opaque_assessment_object_cannot_serialize_hidden_proxy_state(
    starting_case, governance_policy, tmp_path
) -> None:
    proxy_action_key = "name_email_alignment"
    hidden_target = "name_email_similarity"
    rule = governance_policy.field_for_action(proxy_action_key)
    assert rule is not None
    action_name = next(
        name
        for name, resolved in rule.resolved_proxy_actions.items()
        if resolved != starting_case.features[hidden_target]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=(proxy_action_key,),
    )
    facade = AttackerEpisode(env)
    assessment = facade.validator.assess_candidate(
        facade.starting_case.features,
        AttackProposal(changes={proxy_action_key: action_name}),
        locked_values={},
        anchor_id=facade.starting_case.case_id,
        m_max=2,
    )
    serialised = json.dumps(to_jsonable(assessment), sort_keys=True)

    assert assessment.is_valid is True
    assert assessment.edited_action_dimensions == (proxy_action_key,)
    assert hidden_target not in serialised
    assert "candidate_features" not in serialised
    assert "projected_candidate" not in serialised
    assert not hasattr(assessment, "candidate_features")
    assert not hasattr(assessment, "projected_candidate")
    assert set(to_jsonable(assessment)) == {
        "assessment_version",
        "is_valid",
        "error_codes",
        "edit_distance",
        "edited_action_dimensions",
        "canonical_fingerprint",
    }


def test_b1_neutral_affordances_are_complete_stable_and_proxy_safe(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=None,
    )
    payload = build_a3_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        memory_steps=(),
        locked_static_values={},
        query_index=1,
        prompt_version=PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
        outbound_episode_id="dev-anchor-grounded-test",
    )
    affordance = payload["neutral_affordance_view"]
    actions = {item["action_key"]: item for item in affordance["actions"]}
    assert tuple(actions) == env.validator.enabled_action_keys
    assert set(payload["output_schema"]["properties"]["changes"]["properties"]) == set(
        env.validator.enabled_action_keys
    )

    for action_key in env.validator.enabled_action_keys:
        rule = governance_policy.field_for_action(action_key)
        assert rule is not None
        item = actions[action_key]
        if rule.agent_action_mode == "proxy_action":
            assert item["choices"] == list(rule.resolved_proxy_actions)
            assert item["raw_proxy_target_exposed"] is False
            assert rule.feature not in json.dumps(item, sort_keys=True)
        elif rule.data_type in {"categorical", "binary"}:
            assert item["choices"] == list(rule.allowed_values)
            assert item["choice_semantics"] == (
                "complete_governance_legal_categories"
            )
        else:
            expected = []
            seen = set()
            for profile in payload["reference_pool"]["profiles"]:
                fields = profile["fields"]
                if rule.feature not in fields:
                    continue
                value = fields[rule.feature]
                key = json.dumps(value, sort_keys=True, separators=(",", ":"))
                if key not in seen:
                    seen.add(key)
                    expected.append(value)
            assert item["reference_backed_examples"] == expected
            assert item["examples_are_exclusive"] is False
            assert item["domain_rule"]["lower_bound"] == rule.lower_bound
            assert item["domain_rule"]["upper_bound"] == rule.upper_bound

    text = json.dumps(affordance, sort_keys=True).lower()
    assert "gower" not in text
    assert "normality" not in text
    assert "success_rank" not in text
    assert "frequency_order" not in text
    assert affordance["neutrality_contract"] == {
        "selection_guidance": "none",
        "researcher_diagnostics_included": False,
        "d1_information": False,
        "example_order": "fixed_pool_order_only",
    }


def test_b1_b2_share_action_schema_and_only_b2_changes_memory_representation(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=None,
    )
    memory = (
        A3MemoryStep(
            query_index=1,
            strategy_label="field_family_shift",
            changes={"income": 0.4},
            edited_fields=("income",),
            public_label="BLOCK",
            adaptation_note="Change field family after the observed BLOCK.",
            governance_reject_reason=None,
            q_remaining_after=4,
            submitted=True,
        ),
    )
    common = dict(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        memory_steps=memory,
        locked_static_values={},
        query_index=2,
        outbound_episode_id="dev-anchor-grounded-memory-test",
    )
    b1 = build_a3_prompt_payload(
        **common, prompt_version=PROMPT_VERSION_B1_NEUTRAL_GROUNDED
    )
    b2 = build_a3_prompt_payload(
        **common, prompt_version=PROMPT_VERSION_B2_GROUNDED_REFLECTION
    )

    assert b1["output_schema"] == b2["output_schema"]
    assert b1["neutral_affordance_view"] == b2["neutral_affordance_view"]
    assert b1["episode_memory"][0] == memory[0].to_public_dict()
    assert set(b2["episode_memory"][0]) == {
        "query_index",
        "public_label",
        "edited_action_dimensions",
        "strategy_family",
        "exact_chosen_actions",
        "lock_state",
        "remaining_edit_slots",
        "next_strategy_hypothesis",
    }
    assert b2["episode_memory"][0]["public_label"] == "BLOCK"
    assert b2["episode_memory"][0]["next_strategy_hypothesis"] == (
        memory[0].adaptation_note
    )
    assert "governance_reject_reason" not in b2["episode_memory"][0]


def test_b1_cannot_bypass_proxy_action_key_or_governance_choices(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    action_key = "home_phone_configuration"
    hidden_target = "phone_home_valid"
    rule = governance_policy.field_for_action(action_key)
    assert rule is not None
    valid_action = next(
        name
        for name, resolved in rule.resolved_proxy_actions.items()
        if resolved != starting_case.features[hidden_target]
    )
    client = ScriptedLLMClient(
        responses=[
            _candidate_json({hidden_target: 1}),
            _candidate_json({action_key: "forged_proxy_choice"}),
            _candidate_json({action_key: valid_action}),
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=(action_key,),
        defender=defender,
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=14,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        prompt_version=PROMPT_VERSION_B1_NEUTRAL_GROUNDED,
        max_local_generation_attempts_per_query=3,
        llm_client=client,
    )
    attacker.run(AttackerEpisode(env))

    records = attacker.query_records[0].local_generation_records
    assert [item.local_rejection_reason for item in records] == [
        "unknown_action",
        "type_error",
        None,
    ]
    assert [item.env_step_called for item in records] == [False, False, True]
    assert defender.calls == 1
    assert env.ledger.q_used == 1
    assert len(attacker.memory_steps) == 1
    assert attacker.memory_steps[0].changes == {action_key: valid_action}


def test_b2_structured_reflection_uses_no_extra_llm_call(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    foreign = 1 - int(starting_case.features["foreign_request"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": min(0.99, income + 0.05)},
                strategy_label="income",
                adaptation_note="If BLOCK, switch from income to credit limit.",
            ),
            _candidate_json(
                {"foreign_request": foreign},
                strategy_label="foreign_request",
                adaptation_note="Changed field family after the observed BLOCK.",
            ),
        ]
    )
    defender = PassOnSecondDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=("income", "foreign_request"),
        defender=defender,
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=13,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        prompt_version=PROMPT_VERSION_B2_GROUNDED_REFLECTION,
        max_local_generation_attempts_per_query=3,
        llm_client=client,
    )
    attacker.run(AttackerEpisode(env))

    assert defender.calls == 2
    assert len(client.calls) == 2
    assert attacker.total_llm_calls == 2
    assert env.ledger.q_used == 2
    assert len(attacker.memory_steps) == 2
    second_prompt = client.calls[1]["messages"][1]["content"]
    assert "STRUCTURED_EPISODIC_MEMORY" in second_prompt
    assert "If BLOCK, switch from income to credit limit." in second_prompt
    assert "chain-of-thought" in second_prompt
    assert "risk_score" not in second_prompt
    assert "d1_threshold" not in second_prompt


def test_b0_p1_payload_has_no_grounded_representation_change(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=None,
    )
    payload = build_a3_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        memory_steps=(),
        locked_static_values={},
        query_index=1,
        prompt_version=PROMPT_VERSION_P1_COMPACT,
        outbound_episode_id="dev-anchor-b0-regression",
    )
    assert "neutral_affordance_view" not in payload
    assert "per_attempt_fields" in payload["field_roles"]
    assert "per_attempt_action_keys" not in payload["field_roles"]
    assert payload["action_catalogue"]
    assert set(payload["output_schema"]["properties"]["changes"]) == {
        "type",
        "description",
        "minProperties",
        "maxProperties",
    }


def test_original_anchor_equals_current_application_before_first_submission(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session"),
    )
    payload = build_a3_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        memory_steps=(),
        locked_static_values={},
        query_index=1,
    )
    assert "anchor" not in payload
    assert payload["original_anchor"]["case_id"].startswith("dev-anchor-")
    assert payload["original_anchor"]["case_id"] != starting_case.case_id
    assert payload["current_application"]["case_id"] == payload["original_anchor"]["case_id"]
    assert payload["reference_pool"]["anchor_id"] == payload["original_anchor"]["case_id"]
    assert all(
        profile["profile_id"].startswith("ref-")
        for profile in payload["reference_pool"]["profiles"]
    )
    assert "generation_seed" not in json.dumps(payload["reference_pool"])
    assert "pool_fingerprint" not in json.dumps(payload["reference_pool"])
    assert (
        payload["original_anchor"]["visible_fields"]
        == payload["current_application"]["visible_fields"]
    )
    assert (
        payload["original_anchor"]["state_hash"]
        == payload["current_application"]["state_hash"]
    )


def test_original_anchor_immutable_current_application_updates_after_submission(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    new_income = min(0.99, income + 0.05)
    keep = int(starting_case.features["keep_alive_session"])
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": new_income},
                strategy_label="q1",
                adaptation_note="first",
            ),
            _candidate_json(
                {"keep_alive_session": 1 - keep},
                strategy_label="q2",
                adaptation_note="second",
            ),
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=21,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        llm_client=client,
    )
    attacker.run(env)

    q1_state = json.loads(
        (env.logger.run_dir / "query_01" / "a3_state_representation.json").read_text()
    )
    q2_state = json.loads(
        (env.logger.run_dir / "query_02" / "a3_state_representation.json").read_text()
    )
    q1_prompt = json.loads(
        (
            env.logger.run_dir
            / "query_01"
            / "local_gen_01"
            / "a3_prompt_payload.json"
        ).read_text()
    )
    q2_prompt = json.loads(
        (
            env.logger.run_dir
            / "query_02"
            / "local_gen_01"
            / "a3_prompt_payload.json"
        ).read_text()
    )

    assert q1_state["original_anchor"]["state_hash"] == q2_state["original_anchor"][
        "state_hash"
    ]
    assert (
        q1_prompt["original_anchor"]["state_hash"]
        == q2_prompt["original_anchor"]["state_hash"]
    )
    assert (
        q1_prompt["original_anchor"]["visible_fields"]
        == q2_prompt["original_anchor"]["visible_fields"]
    )
    assert (
        q1_state["current_application"]["state_hash"]
        != q2_state["current_application"]["state_hash"]
    )
    assert q2_prompt["current_application"]["visible_fields"]["income"] == new_income
    assert (
        q2_prompt["original_anchor"]["visible_fields"]["income"]
        == float(starting_case.features["income"])
    )
    assert "anchor" not in q1_prompt and "anchor" not in q2_prompt


def test_same_as_anchor_uses_original_and_prompt_exposes_original_values(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    new_income = min(0.99, income + 0.05)
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": new_income},
                strategy_label="q1",
                adaptation_note="leave original",
            ),
            # Revert to original income after q1 — must be same_as_anchor.
            _candidate_json(
                {"income": income},
                strategy_label="revert_original",
                adaptation_note="equals original_anchor",
            ),
            _candidate_json(
                {"income": min(0.99, income + 0.10)},
                strategy_label="repair",
                adaptation_note="true edit vs original",
            ),
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=22,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    q2 = attacker.query_records[1]
    assert q2.local_rejections >= 1
    assert any(
        rec.local_rejection_reason == "same_as_anchor"
        for rec in q2.local_generation_records
    )
    repair_prompt = json.loads(
        (
            env.logger.run_dir
            / "query_02"
            / "local_gen_02"
            / "a3_prompt_payload.json"
        ).read_text()
    )
    assert repair_prompt["original_anchor"]["visible_fields"]["income"] == income
    assert "differ from original_anchor" in repair_prompt["local_proposal_repair"]["note"]
    assert "not the edit-distance reference" in repair_prompt["local_proposal_repair"][
        "note"
    ]


def test_duplicate_of_prior_submission_still_detected_with_split_state(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    new_income = min(0.99, income + 0.05)
    client = ScriptedLLMClient(
        responses=[
            _candidate_json(
                {"income": new_income},
                strategy_label="q1",
                adaptation_note="first",
            ),
            _candidate_json(
                {"income": new_income},
                strategy_label="dup",
                adaptation_note="repeat prior submission",
            ),
            _candidate_json(
                {"income": min(0.99, income + 0.10)},
                strategy_label="unique",
                adaptation_note="new value",
            ),
        ]
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker = EpisodicLLMAgent(
        experiment_seed=23,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        llm_client=client,
    )
    attacker.run(env)
    assert any(
        rec.local_rejection_reason == "duplicate_candidate"
        for rec in attacker.query_records[1].local_generation_records
    )
    assert attacker.total_env_steps == 2
    assert attacker.total_local_regenerations >= 1


def test_prompt_variants_p0_p1_p2_preserve_boundary_and_differ(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session", "customer_age"),
    )
    memory = (
        A3MemoryStep(
            query_index=1,
            strategy_label="s1",
            changes={"income": 0.5},
            edited_fields=("income",),
            public_label="BLOCK",
            adaptation_note="first try",
            governance_reject_reason=None,
            q_remaining_after=4,
            submitted=True,
        ),
    )
    repair = [
        {
            "local_generation_attempt": 1,
            "local_rejection_reason": "duplicate_candidate",
            "changes": {"income": 0.55},
            "edited_fields": ["income"],
        }
    ]
    payloads = {}
    messages = {}
    for version in (PROMPT_VERSION, PROMPT_VERSION_P1_COMPACT, PROMPT_VERSION_P2_NOVELTY):
        payload = build_a3_prompt_payload(
            env=env,
            reference_pool=reference_pool,
            budget=AttackBudget(q_max=5, m_max=2),
            memory_steps=memory,
            locked_static_values={},
            query_index=2,
            prompt_version=version,
            local_proposal_repair=repair,
            local_generation_attempt=2,
        )
        payloads[version] = payload
        messages[version] = render_a3_messages(payload)

    p0_text = messages[PROMPT_VERSION][1]["content"]
    p1_text = messages[PROMPT_VERSION_P1_COMPACT][1]["content"]
    p2_text = messages[PROMPT_VERSION_P2_NOVELTY][1]["content"]
    assert len(p1_text) < len(p0_text)
    assert len(p2_text) < len(p0_text)
    assert "DO_NOT_REPEAT" in p1_text
    assert "PREVIOUS_SUBMISSIONS" in p1_text
    assert "pre_output_self_check" in p2_text
    assert "LOCAL PROPOSAL REJECTED" in p2_text
    assert "adaptation_note" in p1_text  # schema retained
    # Compact variants must not invent D1 leakage fields.
    for text in (p1_text, p2_text):
        assert '"risk_score":' not in text.replace(" ", "")
        assert '"y_score":' not in text.replace(" ", "")
        assert '"threshold":' not in text.replace(" ", "")

    p1_ctx = build_a3_rendered_prompt_context(payloads[PROMPT_VERSION_P1_COMPACT])
    assert p1_ctx["PREVIOUS_SUBMISSIONS"][0]["outcome"] == "BLOCK"
    assert any(item["source"] == "submitted" for item in p1_ctx["DO_NOT_REPEAT"])
    assert any(item["source"] == "local_reject" for item in p1_ctx["DO_NOT_REPEAT"])

    hashes = {
        A3ModelConfig(
            **{**FORMAL_A3_MODEL_CONFIG.to_dict(), "prompt_version": version}
        ).config_hash()
        for version in (
            PROMPT_VERSION,
            PROMPT_VERSION_P1_COMPACT,
            PROMPT_VERSION_P2_NOVELTY,
        )
    }
    assert len(hashes) == 3
    assert FORMAL_A3_MODEL_CONFIG.prompt_version == PROMPT_VERSION


def test_p1_stable_prefix_places_static_context_before_episode_delta(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session", "customer_age"),
    )
    first = build_a3_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        memory_steps=(),
        locked_static_values={},
        query_index=1,
        prompt_version=PROMPT_VERSION_P1_COMPACT,
    )
    second = build_a3_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        memory_steps=(
            A3MemoryStep(
                query_index=1,
                strategy_label="prior",
                changes={"income": 0.5},
                edited_fields=("income",),
                public_label="BLOCK",
                adaptation_note="blocked",
                governance_reject_reason=None,
                q_remaining_after=4,
                submitted=True,
            ),
        ),
        locked_static_values={},
        query_index=2,
        prompt_version=PROMPT_VERSION_P1_COMPACT,
    )
    first_user = render_a3_messages(first)[1]["content"]
    second_user = render_a3_messages(second)[1]["content"]
    marker = "\nCURRENT_APPLICATION\n"
    first_prefix = first_user.split(marker, 1)[0]
    second_prefix = second_user.split(marker, 1)[0]
    assert first_prefix == second_prefix
    assert "GOVERNED_ACTIONS" in first_prefix
    assert "REFERENCE_POOL" in first_prefix
    assert "\nPREVIOUS_SUBMISSIONS\n" not in first_prefix


def test_ranked_portfolio_parser_enforces_cap_and_preserves_order() -> None:
    candidates = [
        _candidate({"income": 0.2}, strategy_label="first"),
        _candidate({"keep_alive_session": 1}, strategy_label="second"),
        _candidate({"payment_type": "AA"}, strategy_label="third"),
    ]
    parsed, status = parse_a3_ranked_portfolio(
        _portfolio_json(*candidates), m_max=2
    )
    assert status == "ok"
    assert parsed is not None
    assert [item["strategy_label"] for item in parsed] == [
        "first",
        "second",
        "third",
    ]
    assert parse_a3_ranked_portfolio('{"candidates":[]}', m_max=2)[1] == (
        "schema_error"
    )
    assert parse_a3_ranked_portfolio(
        _portfolio_json(*candidates, candidates[0]), m_max=2
    )[1] == "schema_error"
    assert parse_a3_ranked_portfolio(
        _candidate_json({"income": 0.2}), m_max=2
    )[1] == "schema_error"


def test_ranked_portfolio_submits_first_legal_unique_candidate_once(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = _different_income(starting_case)
    client = ScriptedLLMClient(
        responses=[
            _portfolio_json(
                _candidate(
                    {"income": float(starting_case.features["income"])},
                    strategy_label="same_anchor",
                ),
                _candidate({"income": income}, strategy_label="selected"),
                _candidate({"income": income}, strategy_label="duplicate_unselected"),
            )
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=("income", "keep_alive_session"),
        defender=defender,
    )
    attacker = _ranked_agent(
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        client=client,
    )
    attacker.run(env)

    assert len(client.calls) == 1
    assert defender.calls == 1
    assert env.ledger.q_used == 1
    assert len(attacker.memory_steps) == 1
    assert attacker.memory_steps[0].strategy_label == "selected"
    query = attacker.query_records[0]
    assert query.selected_portfolio_rank == 2
    assert len(query.local_generation_records) == 3
    first, second, third = query.local_generation_records
    assert first.local_rejection_reason == "same_as_anchor"
    assert first.env_step_called is False
    assert second.local_validation_ok is True
    assert second.selected_for_submission is True
    assert second.env_step_called is True
    assert third.local_rejection_reason == "duplicate_candidate"
    assert third.env_step_called is False


def test_ranked_portfolio_static_lock_accounting_and_budget_fallback(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    age = int(starting_case.features["customer_age"])
    new_age = age + 1 if age < 90 else age - 1
    income = _different_income(starting_case)
    keep = 1 - int(starting_case.features["keep_alive_session"])
    email = 1 - int(starting_case.features["email_is_free"])
    client = ScriptedLLMClient(
        responses=[
            _portfolio_json(
                _candidate({"customer_age": new_age}, strategy_label="lock_age")
            ),
            _portfolio_json(
                _candidate(
                    {"income": income, "keep_alive_session": keep},
                    strategy_label="too_many_after_lock",
                ),
                _candidate(
                    {"keep_alive_session": keep}, strategy_label="fits_one_slot"
                ),
                _candidate({"email_is_free": email}, strategy_label="other_one_slot"),
            ),
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(2, 2),
        enabled=(
            "customer_age",
            "income",
            "keep_alive_session",
            "email_is_free",
        ),
        defender=defender,
    )
    attacker = _ranked_agent(
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=2, m_max=2),
        client=client,
        seed=102,
    )
    attacker.run(env)

    assert len(client.calls) == 2
    assert defender.calls == 2
    assert env.ledger.q_used == 2
    q2 = attacker.query_records[1]
    assert q2.selected_portfolio_rank == 2
    assert q2.local_generation_records[0].local_rejection_reason == "budget_exceeded"
    accounting = json.loads(
        (
            env.logger.run_dir
            / "query_02"
            / "a3_ranked_portfolio_input_summary.json"
        ).read_text()
    )["edit_slot_accounting"]
    assert accounting["locked_static_edit_slots_occupied"] == 1
    assert accounting["remaining_edit_slots_after_static_locks"] == 1
    direct = compute_a3_edit_slot_accounting(
        env=env,
        locked_static_values=env.locked_static_values,
        budget=AttackBudget(q_max=2, m_max=2),
    )
    assert direct["locked_static_edit_slots_occupied"] == 1
    assert direct["remaining_edit_slots_after_static_locks"] == 1


def test_ranked_portfolio_skips_proxy_and_domain_failures(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    client = ScriptedLLMClient(
        responses=[
            _portfolio_json(
                _candidate(
                    {"name_email_similarity": 0.9},
                    strategy_label="forbidden_direct_proxy",
                ),
                _candidate({"income": 999.0}, strategy_label="outside_domain"),
                _candidate(
                    {"name_email_alignment": "high_alignment"},
                    strategy_label="legal_proxy_action",
                ),
            )
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=("income", "name_email_alignment"),
        defender=defender,
    )
    attacker = _ranked_agent(
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        client=client,
        seed=103,
    )
    attacker.run(env)

    records = attacker.query_records[0].local_generation_records
    assert [item.local_rejection_reason for item in records] == [
        "unknown_action",
        "out_of_domain",
        None,
    ]
    assert attacker.query_records[0].selected_portfolio_rank == 3
    assert defender.calls == 1
    assert env.ledger.q_used == 1


def test_ranked_portfolio_skips_relationship_failure(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    keep = 1 - int(starting_case.features["keep_alive_session"])
    client = ScriptedLLMClient(
        responses=[
            _portfolio_json(
                _candidate(
                    {"customer_age": 90, "current_address_months_count": 400},
                    strategy_label="unsupported_relation",
                ),
                _candidate(
                    {"keep_alive_session": keep}, strategy_label="legal_fallback"
                ),
                _candidate(
                    {"income": _different_income(starting_case)},
                    strategy_label="unselected_fallback",
                ),
            )
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(1, 2),
        enabled=(
            "customer_age",
            "current_address_months_count",
            "keep_alive_session",
            "income",
        ),
        defender=defender,
    )
    attacker = _ranked_agent(
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        client=client,
        seed=104,
    )
    attacker.run(env)

    records = attacker.query_records[0].local_generation_records
    assert records[0].local_rejection_reason == "constraint_failed"
    assert attacker.query_records[0].selected_portfolio_rank == 2
    assert defender.calls == 1


def test_ranked_portfolio_all_fail_exhausts_without_q_or_feedback(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    income = float(starting_case.features["income"])
    client = ScriptedLLMClient(
        responses=[
            _portfolio_json(
                _candidate({"income": income}, strategy_label="same_1"),
                _candidate({"income": income}, strategy_label="same_2"),
                _candidate({"income": income}, strategy_label="same_3"),
            )
        ]
    )
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session"),
        defender=defender,
    )
    attacker = _ranked_agent(
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        client=client,
        seed=105,
    )
    attacker.run(env)

    assert len(client.calls) == 1
    assert defender.calls == 0
    assert env.ledger.q_used == 0
    assert attacker.memory_steps == ()
    assert attacker.total_env_steps == 0
    assert env.result().stop_reason == "local_generation_exhausted"
    query = attacker.query_records[0]
    assert query.regeneration_exhausted is True
    assert query.selected_portfolio_rank is None
    assert len(query.local_generation_records) == 3
    assert all(
        item.local_rejection_reason == "same_as_anchor"
        for item in query.local_generation_records
    )


def test_ranked_portfolio_freezes_one_call_budget_and_resets_between_anchors(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    with pytest.raises(A3AgentError, match="exactly one generation batch"):
        EpisodicLLMAgent(
            experiment_seed=106,
            reference_pool=reference_pool,
            budget=AttackBudget(q_max=1, m_max=2),
            prompt_version=PROMPT_VERSION_P1_RANKED_PORTFOLIO,
            max_parse_retries=0,
            max_local_generation_attempts_per_query=2,
        )
    with pytest.raises(A3AgentError, match="max_parse_retries must be 0"):
        EpisodicLLMAgent(
            experiment_seed=106,
            reference_pool=reference_pool,
            budget=AttackBudget(q_max=1, m_max=2),
            prompt_version=PROMPT_VERSION_P1_RANKED_PORTFOLIO,
            max_parse_retries=1,
            max_local_generation_attempts_per_query=1,
        )

    income = _different_income(starting_case)
    keep = 1 - int(starting_case.features["keep_alive_session"])
    client = ScriptedLLMClient(
        responses=[
            _portfolio_json(
                _candidate({"income": income}, strategy_label="anchor_one")
            ),
            _portfolio_json(
                _candidate(
                    {"keep_alive_session": keep}, strategy_label="anchor_two"
                )
            ),
        ]
    )
    attacker = _ranked_agent(
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=1, m_max=2),
        client=client,
        seed=106,
    )
    env1 = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path / "anchor_one",
        budget=_qm_budget(1, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker.run(env1)
    second_case = replace(
        starting_case,
        case_id="900002",
        source_row_id=900002,
    )
    env2 = _make_env(
        starting_case=second_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path / "anchor_two",
        budget=_qm_budget(1, 2),
        enabled=("income", "keep_alive_session"),
        defender=CountingBlockDefender(),
    )
    attacker.run(env2)

    assert len(client.calls) == 2
    assert len(attacker.memory_steps) == 1
    assert attacker.memory_steps[0].strategy_label == "anchor_two"
    second_prompt = client.calls[1]["messages"][1]["content"]
    assert "anchor_one" not in second_prompt
    assert attacker.total_llm_calls == 1
    assert attacker.total_env_steps == 1


def test_ranked_prompt_exposes_only_public_budget_accounting(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        budget=_qm_budget(5, 2),
        enabled=("income", "keep_alive_session", "customer_age"),
    )
    payload = build_a3_prompt_payload(
        env=env,
        reference_pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        memory_steps=(),
        locked_static_values={},
        query_index=1,
        prompt_version=PROMPT_VERSION_P1_RANKED_PORTFOLIO,
        max_local_generation_attempts=1,
    )
    assert payload["portfolio_cap"] == 3
    assert payload["edit_slot_accounting"]["locked_static_edit_slots_occupied"] == 0
    assert payload["edit_slot_accounting"][
        "remaining_edit_slots_after_static_locks"
    ] == 2
    assert payload["output_schema"]["properties"]["candidates"]["maxItems"] == 3
    blob = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        '"risk_score":',
        '"threshold":',
        '"fraud_bool":',
        '"feature_importance":',
        '"shap":',
        '"gradient":',
    ):
        assert forbidden not in blob
