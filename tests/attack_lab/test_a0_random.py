"""Tests for the frozen Q,m A0 constrained-random baseline."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker, derive_episode_seed
from attack_lab.budget import BudgetSpec
from attack_lab.cli import build_parser
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
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


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    config = ReferencePoolConfig.load()
    return ReferencePoolProvider.from_config(
        config, training_frame=train
    ).get_pool(starting_case.case_id)


def _qm_budget(q_max: int, m_max: int) -> BudgetSpec:
    return BudgetSpec.development_dummy(
        q_max=q_max, m_max=m_max, label="dummy_qm_protocol"
    )


def _make_env(
    *,
    starting_case,
    governance_policy,
    tmp_path: Path,
    budget: BudgetSpec,
    enabled: tuple[str, ...] | None,
    reference_pool=None,
    require_reference_provenance: bool = True,
):
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=CountingBlockDefender(),
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


def test_cli_accepts_a0_seed_and_m_max() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--attacker",
            "a0",
            "--case-id",
            "795076",
            "--max-attempts",
            "3",
            "--seed",
            "42",
            "--m-max",
            "5",
        ]
    )
    assert args.attacker == "a0"
    assert args.seed == 42
    assert args.m_max == 5


def test_feedback_does_not_change_frozen_sequence(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(3, 5)
    enabled = ("income", "keep_alive_session", "payment_type", "customer_age")

    def collect(poison: bool) -> list[dict[str, Any]]:
        env = _make_env(
            starting_case=starting_case,
            governance_policy=governance_policy,
            reference_pool=reference_pool,
            tmp_path=tmp_path / f"fb_{int(poison)}",
            budget=budget,
            enabled=enabled,
        )
        attacker = ConstrainedRandomAttacker(
            seed=123,
            reference_pool=reference_pool,
            m_max=5,
            attacker_id="a0",
        )
        frozen = attacker.prepare_frozen_sequence(env)
        assert frozen
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
        # Re-prepare must keep the same frozen sequence object content.
        again = ConstrainedRandomAttacker(
            seed=123,
            reference_pool=reference_pool,
            m_max=5,
            attacker_id="a0",
        )
        env2 = _make_env(
            starting_case=starting_case,
            governance_policy=governance_policy,
            reference_pool=reference_pool,
            tmp_path=tmp_path / f"fb2_{int(poison)}",
            budget=budget,
            enabled=enabled,
        )
        assert [dict(p.changes) for p in again.prepare_frozen_sequence(env2)] == proposals
        return proposals

    assert collect(False) == collect(True)


def test_same_anchor_seed_reproducible_sequence(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(4, 5)
    enabled = ("income", "keep_alive_session", "payment_type", "customer_age")

    sequence_index = 0

    def sequence() -> list[dict[str, Any]]:
        nonlocal sequence_index
        sequence_index += 1
        env = _make_env(
            starting_case=starting_case,
            governance_policy=governance_policy,
            reference_pool=reference_pool,
            tmp_path=tmp_path / f"rep_{sequence_index}",
            budget=budget,
            enabled=enabled,
        )
        attacker = ConstrainedRandomAttacker(
            seed=7,
            reference_pool=reference_pool,
            m_max=5,
            attacker_id="a0",
        )
        return [
            {
                "changes": dict(p.changes),
                "fingerprint": p.research_meta["candidate_fingerprint"],
            }
            for p in attacker.prepare_frozen_sequence(env)
        ]

    assert sequence() == sequence()


def test_different_anchors_different_episode_seeds(
    starting_case, synthetic_frame, governance_policy, tmp_path
) -> None:
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    provider = ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    )
    other = starting_case.__class__(
        case_id="anchor_other_999",
        source_row_id=999,
        label=1,
        features=dict(starting_case.features),
        initial_score=starting_case.initial_score,
        initial_decision="BLOCK",
        data_split="dev_month6",
    )
    seed = 20260803
    assert derive_episode_seed(seed, starting_case.case_id, "a0") != derive_episode_seed(
        seed, other.case_id, "a0"
    )

    budget = _qm_budget(3, 4)
    enabled = ("income", "keep_alive_session", "payment_type")
    seqs = []
    for case in (starting_case, other):
        pool = provider.get_pool(case.case_id, seed=seed)
        env = _make_env(
            starting_case=case,
            governance_policy=governance_policy,
            reference_pool=pool,
            tmp_path=tmp_path / f"anch_{case.case_id}",
            budget=budget,
            enabled=enabled,
        )
        attacker = ConstrainedRandomAttacker(
            seed=seed, reference_pool=pool, m_max=4, attacker_id="a0"
        )
        seqs.append(
            [p.research_meta["candidate_fingerprint"] for p in attacker.prepare_frozen_sequence(env)]
        )
    assert seqs[0] != seqs[1]


def test_every_candidate_respects_m_distance(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    m_max = 3
    budget = _qm_budget(5, m_max)
    enabled = (
        "income",
        "keep_alive_session",
        "payment_type",
        "customer_age",
        "employment_status",
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=enabled,
    )
    attacker = ConstrainedRandomAttacker(
        seed=11, reference_pool=reference_pool, m_max=m_max, attacker_id="a0"
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    for proposal in frozen:
        distance = int(proposal.research_meta["edit_distance_from_anchor"])
        assert 1 <= distance <= m_max
        assert len(proposal.research_meta["edited_fields"]) == distance


def test_generation_never_exceeds_q_or_m(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    q_max, m_max = 4, 2
    budget = _qm_budget(q_max, m_max)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "keep_alive_session", "payment_type", "customer_age"),
    )
    attacker = ConstrainedRandomAttacker(
        seed=19, reference_pool=reference_pool, m_max=m_max, attacker_id="a0"
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert len(frozen) <= q_max
    assert all(
        int(p.research_meta["edit_distance_from_anchor"]) <= m_max for p in frozen
    )


def test_candidates_generated_before_any_execution(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(3, 5)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "keep_alive_session", "payment_type"),
    )
    defender = env.defender
    attacker = ConstrainedRandomAttacker(
        seed=29, reference_pool=reference_pool, m_max=5, attacker_id="a0"
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert len(frozen) == budget.q_max or len(frozen) >= 1
    # No D1 calls during generation.
    inner = getattr(defender, "_inner", defender)
    assert getattr(inner, "calls", 0) == 0
    assert env.attempts_used == 0
    assert all(p.research_meta.get("candidate_index") for p in frozen)


def test_research_meta_not_in_observation(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(2, 4)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "keep_alive_session", "payment_type"),
    )
    attacker = ConstrainedRandomAttacker(
        seed=5, reference_pool=reference_pool, m_max=4, attacker_id="a0"
    )
    proposal = attacker.propose(env)
    assert proposal is not None
    assert proposal.research_meta.get("generation_seed") is not None
    obs = env.observation()
    blob = str(obs.__dict__)
    assert "candidate_fingerprint" not in blob
    assert "reference_ids_used" not in blob
    assert "generation_seed" not in blob


def test_enum_fallback_when_random_retries_disabled(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    """Random exhaustion must not emit false no_feasible when enum is non-empty."""
    budget = _qm_budget(3, 2)
    enabled = (
        "income",
        "keep_alive_session",
        "payment_type",
        "employment_status",
        "customer_age",
    )
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=enabled,
    )
    attacker = ConstrainedRandomAttacker(
        seed=101,
        reference_pool=reference_pool,
        m_max=2,
        attacker_id="a0",
        max_local_resamples=0,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    assert attacker.sampling_diagnostics["undersample_events"] >= 1
    assert attacker.sampling_diagnostics["enum_fallback_picks"] == len(frozen)
    assert all(
        p.research_meta.get("generation_method") == "enum_fallback" for p in frozen
    )
    assert all(
        1 <= int(p.research_meta["edit_distance_from_anchor"]) <= 2 for p in frozen
    )


def test_enum_fallback_pick_is_stable_across_runs(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(2, 2)
    enabled = ("income", "keep_alive_session", "payment_type", "employment_status")

    def freeze(label: str):
        env = _make_env(
            starting_case=starting_case,
            governance_policy=governance_policy,
            reference_pool=reference_pool,
            tmp_path=tmp_path / label,
            budget=budget,
            enabled=enabled,
        )
        attacker = ConstrainedRandomAttacker(
            seed=20260804,
            reference_pool=reference_pool,
            m_max=2,
            attacker_id="a0",
            max_local_resamples=0,
        )
        frozen = attacker.prepare_frozen_sequence(env)
        return [
            {
                "changes": dict(p.changes),
                "fingerprint": p.research_meta["candidate_fingerprint"],
            }
            for p in frozen
        ]

    assert freeze("a") == freeze("b")


def test_no_feasible_only_when_enumeration_empty(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(2, 0)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "keep_alive_session"),
    )
    attacker = ConstrainedRandomAttacker(
        seed=3,
        reference_pool=reference_pool,
        m_max=0,
        attacker_id="a0",
        max_local_resamples=0,
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen == ()
    assert attacker._pending_stop_reason == "insufficient_edit_budget"  # noqa: SLF001
    audit = attacker.sampling_diagnostics["termination_audit"]
    assert audit is not None
    assert audit["enum_remaining_count"] == 0
    assert audit["undersample_confirmed"] is False


def test_termination_audit_records_lock_pool_and_rejects(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(2, 3)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=("income", "keep_alive_session", "payment_type", "employment_status"),
    )
    attacker = ConstrainedRandomAttacker(
        seed=17,
        reference_pool=reference_pool,
        m_max=3,
        attacker_id="a0",
        stdout=io.StringIO(),
        max_local_resamples=4,
    )
    attacker.run(env)
    diag = attacker.sampling_diagnostics
    audit = diag["termination_audit"]
    assert audit is not None
    assert audit["anchor_id"] == starting_case.case_id
    assert audit["m_max"] == 3
    assert audit["reference_pool_fingerprint"] == reference_pool.pool_fingerprint
    assert "lock_plan" in audit
    assert "submitted_fingerprints" in audit
    assert "random_reject_counts" in audit
    assert isinstance(audit["random_reject_counts"], dict)


def test_orchestrated_episode_submits_frozen_sequence(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = _qm_budget(3, 5)
    logger = TrajectoryLogger(run_dir=tmp_path / "match", run_id="match")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    attacker = ConstrainedRandomAttacker(
        seed=41,
        reference_pool=reference_pool,
        m_max=5,
        attacker_id="a0",
        stdout=io.StringIO(),
    )
    result = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a0",
            anchor=starting_case,
            policy=governance_policy,
            budget=budget,
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=CountingBlockDefender(),
            seed=41,
            enabled_action_keys=("income", "keep_alive_session", "payment_type"),
            logger=logger,
            reference_pool=reference_pool,
            require_reference_provenance=True,
        ),
    )
    assert len(attacker.frozen_proposals) >= 1
    assert len(result.trajectory) <= budget.q_max
    for step in result.trajectory:
        if step.research_meta:
            assert step.research_meta["edit_distance_from_anchor"] <= 5
