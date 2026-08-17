"""Tests for A2 surrogate-guided adaptive searcher (mechanism verification)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
from attack_lab.budget import AttackBudget
from attack_lab.cli import build_parser
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import DefenceDecision, InternalDefenceResult
from attack_lab.validator import ConstraintValidator


@dataclass
class CountingBlockDefender:
    name: str = "counting_block"
    artefact_id: str = "test"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)
    scores_seen: list[float] = field(default_factory=list, init=False)

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult:
        self.calls += 1
        score = 0.9
        self.scores_seen.append(score)
        decision: DefenceDecision = "BLOCK"
        return InternalDefenceResult(
            risk_score=score,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


@dataclass
class PassOnSecondDefender:
    name: str = "pass_second"
    artefact_id: str = "test"
    threshold: float = 0.5
    calls: int = field(default=0, init=False)

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult:
        self.calls += 1
        decision: DefenceDecision = "PASS" if self.calls >= 2 else "BLOCK"
        return InternalDefenceResult(
            risk_score=0.1 if decision == "PASS" else 0.9,
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


ENABLED = (
    "income",
    "keep_alive_session",
    "payment_type",
    "employment_status",
    "customer_age",
)


def _make_env(
    *,
    starting_case,
    governance_policy,
    tmp_path: Path,
    budget: AttackBudget,
    defender=None,
    enabled=ENABLED,
    reference_pool=None,
    require_reference_provenance: bool = True,
):
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    return AttackEnvironment(
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
        budget=budget.to_budget_spec(label="test_a2_budget"),
    )


def test_cli_accepts_a2_m_q_experiment_seed() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--attacker",
            "a2",
            "--case-id",
            "795076",
            "--m",
            "2",
            "--q",
            "5",
            "--experiment-seed",
            "20260804",
            "--n-anchors",
            "30",
        ]
    )
    assert args.attacker == "a2"
    assert args.m_max == 2
    assert args.q_max == 5
    assert args.experiment_seed == 20260804
    assert args.n_anchors == 30


def test_governance_view_exposes_roles_not_model_internals(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=11,
        attacker_id="a2",
    )
    attacker.run(env)
    view = attacker.governance_view
    assert view is not None
    # Roles (compiled synthetic governance may differ from production counts).
    assert set(view.per_attempt_fields).isdisjoint(view.episode_static_fields)
    assert "income" in view.per_attempt_fields or "income" in {
        r.feature for r in view.action_field_rules
    }
    public = view.to_public_dict()
    hidden = set(public["explicitly_hidden"])
    assert "d1_risk_score" in hidden
    assert "d1_threshold" in hidden
    assert "feature_importance_or_shap" in hidden
    assert "fraud_bool" in hidden
    assert "month7_data" in hidden
    # No live model internals in episode_state / action rules.
    assert "risk_score" not in public["episode_state"]
    assert "threshold" not in public["episode_state"]


def test_forbidden_and_readonly_not_in_action_rules(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=2, m_max=2)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        enabled=None,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget, reference_pool=reference_pool, experiment_seed=3
    )
    # Build view without running full enum-heavy episode when enabled=None on
    # synthetic policy — instead construct from policy via a short run with
    # restricted fields, and check pool read-only against action rules.
    env2 = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "b",
        budget=budget,
        enabled=ENABLED,
    )
    attacker.run(env2)
    view = attacker.governance_view
    assert view is not None
    action_features = {rule.feature for rule in view.action_field_rules}
    for name in view.read_only_context_fields:
        assert name not in action_features
    for name in view.forbidden_fields:
        assert name not in action_features


@pytest.mark.parametrize("m_max", [1, 2, 3])
def test_m_interface_constrains_candidate_distance(
    m_max, starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=4, m_max=m_max)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / f"m{m_max}",
        budget=budget,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=21,
        attacker_id="a2",
    )
    attacker.run(env)
    assert env.attempts_used >= 1
    for step in env.result().steps:
        assert step.submitted_edit_cost <= m_max
        meta = step.research_meta or {}
        if "edit_distance_from_anchor" in meta:
            assert int(meta["edit_distance_from_anchor"]) <= m_max


def test_static_fields_lock_after_first_submission(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        defender=CountingBlockDefender(),
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget, reference_pool=reference_pool, experiment_seed=31
    )
    attacker.run(env)
    steps = env.result().steps
    assert len(steps) >= 1
    first_static = {
        k: v
        for k, v in steps[0].proposed_changes.items()
        if k in {"employment_status", "customer_age"}
    }
    # After first submission, later proposals must not change locked static values.
    locks = dict(env._episode_locks)  # noqa: SLF001
    for name in ("employment_status", "customer_age"):
        if name in locks:
            for step in steps[1:]:
                if name in step.proposed_changes:
                    assert step.proposed_changes[name] == first_static.get(
                        name, locks[name]
                    ) or step.proposed_changes[name] == locks[name]


def test_static_edits_consume_m(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=2, m_max=1)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget, reference_pool=reference_pool, experiment_seed=41
    )
    attacker.run(env)
    for step in env.result().steps:
        meta = step.research_meta or {}
        locked = int(meta.get("locked_edit_count", 0))
        dynamic = int(meta.get("dynamic_edit_count", 0))
        assert locked + dynamic == int(meta.get("edit_distance_from_anchor", 0))
        assert locked + dynamic <= 1


def test_no_duplicate_candidates(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=4, m_max=2)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget, reference_pool=reference_pool, experiment_seed=51
    )
    attacker.run(env)
    hashes = [
        step.research_meta.get("candidate_hash")
        for step in env.result().steps
        if step.research_meta
    ]
    assert len(hashes) == len(set(hashes))


def test_does_not_stop_while_legal_candidates_remain(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget, reference_pool=reference_pool, experiment_seed=61
    )
    attacker.run(env)
    result = env.result()
    if result.stop_reason == "action_space_exhaustion":
        # Last log must show remaining 0 before stop.
        assert attacker.submission_logs
        assert attacker.submission_logs[-1]["legal_unique_candidates_remaining"] == 0
    else:
        assert result.stop_reason in {"q_exhausted", "success", "m_exceeded"}


def test_block_changes_ranking(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        defender=CountingBlockDefender(),
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget, reference_pool=reference_pool, experiment_seed=71
    )
    attacker.run(env)
    logs = attacker.submission_logs
    if len(logs) >= 2:
        assert logs[0]["candidate_hash"] != logs[1]["candidate_hash"]
        # After first BLOCK, diversification metrics become active.
        assert logs[1]["failure_field_value_overlap"] is not None


def test_reproducible_with_same_seed(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)

    def run_once(label: str) -> list[str]:
        env = _make_env(
            starting_case=starting_case,
            governance_policy=governance_policy,
            reference_pool=reference_pool,
            tmp_path=tmp_path / label,
            budget=budget,
        )
        attacker = SurrogateGuidedSearcher(
            budget=budget, reference_pool=reference_pool, experiment_seed=81
        )
        attacker.run(env)
        return [
            str(step.research_meta.get("candidate_hash"))
            for step in env.result().steps
        ]

    assert run_once("a") == run_once("b")


def test_no_cross_anchor_memory(
    starting_case, governance_policy, reference_pool, tmp_path, synthetic_frame
) -> None:
    budget = AttackBudget(q_max=2, m_max=2)
    other = starting_case.__class__(
        case_id="anchor_other_a2",
        source_row_id=998,
        label=1,
        features=dict(starting_case.features),
        initial_score=starting_case.initial_score,
        initial_decision="BLOCK",
        data_split="dev_month6",
    )
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    provider = ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=provider.get_pool(starting_case.case_id),
        experiment_seed=91,
    )
    env1 = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "one",
        budget=budget,
    )
    attacker.run(env1)
    assert attacker.governance_view is not None
    assert attacker.governance_view.submission_history == ()
    assert attacker._history == []  # noqa: SLF001

    attacker.reference_pool = provider.get_pool(other.case_id)
    env2 = _make_env(
        starting_case=other,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "two",
        budget=budget,
    )
    attacker.run(env2)
    assert attacker.governance_view.submission_history == ()


def test_does_not_read_month7_or_d1_internals(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=2, m_max=2)
    defender = CountingBlockDefender()
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path,
        budget=budget,
        defender=defender,
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=101,
        stdout=io.StringIO(),
    )
    attacker.run(env)
    # A2 only reaches D1 through env.step; scores must not appear in A2 logs.
    joined = " ".join(str(item) for item in attacker.submission_logs)
    assert "risk_score" not in joined
    assert "threshold" not in joined
    assert "shap" not in joined
    assert "month" not in joined or "month7" not in joined.lower()
    # Pool source months are 0-5 only (provider invariant).
    assert reference_pool.generation_seed is not None


def test_orchestrated_episode_works(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    budget = AttackBudget(q_max=3, m_max=2)
    logger = TrajectoryLogger(run_dir=tmp_path / "match", run_id="match")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=111,
        stdout=io.StringIO(),
    )
    result = MatchOrchestrator().run_episode(
        attacker,
        MatchConfig(
            attacker_id="a2",
            anchor=starting_case,
            policy=governance_policy,
            budget=budget.to_budget_spec(),
            feedback_policy=FeedbackPolicy(mode="label_only"),
            defender=PassOnSecondDefender(),
            seed=111,
            enabled_action_keys=ENABLED,
            logger=logger,
            reference_pool=reference_pool,
            require_reference_provenance=True,
        ),
    )
    assert result.attacker_id == "a2"
    assert result.q_used >= 1
    assert result.success is True
    assert result.attempts_to_success == 2
