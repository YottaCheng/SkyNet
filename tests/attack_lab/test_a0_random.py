"""Tests for A0 constrained random attacker."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from attack_lab.a0_random import ConstrainedRandomAttacker
from attack_lab.budget import BudgetSpec
from attack_lab.cli import build_parser
from attack_lab.domains import (
    AttackLabDomainError,
    ProposalDomainSet,
    build_proposal_domains,
)
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.types import AttackProposal
from attack_lab.validator import ConstraintValidator


def _domains_for(mutable: tuple[str, ...]) -> ProposalDomainSet:
    return build_proposal_domains(
        mutable,
        categorical_vocabularies={
            "payment_type": ("AA", "AB"),
            "employment_status": ("CA", "CB"),
            "housing_status": ("BA", "BB"),
            "source": ("INTERNET", "TELEAPP"),
            "device_os": ("linux", "windows"),
        },
        numeric_domains_config={
            "label": "test_domains",
            "numeric_domains": {
                "income": {"kind": "float", "low": 0.0, "high": 1.0},
                "name_email_similarity": {"kind": "float", "low": 0.0, "high": 1.0},
                "customer_age": {"kind": "integer", "low": 10, "high": 90},
            },
        },
    )


class _DefaultBlockDefender:
    name = "mock"
    artefact_id = "mock"
    threshold = 0.5

    def score_application(self, features):
        from attack_lab.types import InternalDefenceResult

        return InternalDefenceResult(
            risk_score=0.9,
            threshold=self.threshold,
            decision="BLOCK",
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


def _env(
    starting_case,
    mutable,
    tmp_path,
    governance_policy,
    *,
    max_attempts=3,
    defender=None,
):
    defender = defender or _DefaultBlockDefender()
    validator = ConstraintValidator.from_policy(
        governance_policy,
        enabled_action_keys=mutable,
    )
    logger = TrajectoryLogger(run_dir=tmp_path / "a0run", run_id="a0run")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=defender,
        validator=validator,
        feedback_policy=FeedbackPolicy(mode="label_only"),
        max_attempts=max_attempts,
        logger=logger,
        budget=BudgetSpec.development_dummy(
            q_max=max_attempts,
            e_max=1_000_000,
            label="test_dummy_budget_not_final",
        ),
    )
    return env, defender, logger


def test_cli_accepts_a0_and_requires_seed() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--attacker",
            "a0",
            "--case-id",
            "795076",
            "--mutable-fields",
            "income",
            "--max-attempts",
            "3",
            "--seed",
            "42",
        ]
    )
    assert args.attacker == "a0"
    assert args.seed == 42
    assert args.defence == "d1"


def test_missing_numeric_domain_is_blocker() -> None:
    with pytest.raises(AttackLabDomainError, match="Missing sampling domain|numeric domain"):
        build_proposal_domains(
            ("income", "zip_count_4w"),
            categorical_vocabularies={},
            numeric_domains_config={
                "numeric_domains": {
                    "income": {"kind": "float", "low": 0.0, "high": 1.0},
                }
            },
        )


def test_categorical_domains_use_fitted_vocabulary() -> None:
    domains = build_proposal_domains(
        ("payment_type",),
        categorical_vocabularies={"payment_type": ("AA", "AB", "AC")},
        numeric_domains_config={"numeric_domains": {}},
    )
    assert domains.domains["payment_type"].values == ("AA", "AB", "AC")
    assert domains.domains["payment_type"].source == "fitted_c1_onehot_vocabulary"


def test_proposals_only_touch_mutable_fields(
    starting_case, tmp_path, governance_policy
) -> None:
    mutable = ("income", "customer_age")
    env, _defender, _logger = _env(
        starting_case, mutable, tmp_path, governance_policy
    )
    attacker = ConstrainedRandomAttacker(
        env=env,
        domains=_domains_for(mutable),
        seed=0,
        stdout=io.StringIO(),
    )
    proposal = attacker.propose()
    assert set(proposal.changes) == set(mutable)
    assert "payment_type" not in proposal.changes


def test_proposals_are_independent_of_previous_candidate(
    starting_case, tmp_path, governance_policy
) -> None:
    mutable = ("income",)
    env, _defender, _logger = _env(
        starting_case,
        mutable,
        tmp_path,
        governance_policy,
        max_attempts=5,
    )
    attacker = ConstrainedRandomAttacker(
        env=env,
        domains=_domains_for(mutable),
        seed=1,
        stdout=io.StringIO(),
    )
    # Force a known failed candidate into the environment's current features.
    env._current_features["income"] = 0.123456  # noqa: SLF001
    proposal = attacker.propose()
    # Proposal must be generated from original starting case, not current features.
    assert "income" in proposal.changes
    # Sampling uses baseline only via domains.sample_changes(baseline=starting).
    baseline = env.starting_case.features
    validity = env.validator.validate(baseline, proposal)
    assert validity.is_valid


def test_a0_does_not_adapt_from_block_feedback(
    starting_case, tmp_path, governance_policy
) -> None:
    mutable = ("income",)

    class RecordingDefender:
        name = "mock"
        artefact_id = "mock"
        threshold = 0.5

        def score_application(self, features):
            from attack_lab.types import InternalDefenceResult

            return InternalDefenceResult(
                risk_score=0.9,
                threshold=0.5,
                decision="BLOCK",
                runtime_ms=0.01,
                defender_name=self.name,
                artefact_id=self.artefact_id,
            )

    env, _d, _logger = _env(
        starting_case,
        mutable,
        tmp_path,
        governance_policy,
        max_attempts=3,
        defender=RecordingDefender(),
    )
    # Two attackers with same seed must produce identical proposal sequences
    # even if the first one observes BLOCK feedback between proposes.
    domains = _domains_for(mutable)
    out = io.StringIO()
    a1 = ConstrainedRandomAttacker(env=env, domains=domains, seed=99, stdout=out)
    p1 = a1.propose().changes["income"]
    env.step(AttackProposal(changes={"income": p1}))
    assert env._last_feedback is not None  # noqa: SLF001
    assert env._last_feedback.label == "BLOCK"  # noqa: SLF001
    p2 = a1.propose().changes["income"]

    # Fresh attacker, same seed: first two proposes match without any feedback.
    env2, _d2, _logger2 = _env(
        starting_case,
        mutable,
        tmp_path / "b",
        governance_policy,
        max_attempts=3,
        defender=RecordingDefender(),
    )
    a2 = ConstrainedRandomAttacker(
        env=env2, domains=domains, seed=99, stdout=io.StringIO()
    )
    q1 = a2.propose().changes["income"]
    q2 = a2.propose().changes["income"]
    assert p1 == q1
    assert p2 == q2


def test_a0_receives_only_public_labels_in_loop(
    starting_case, tmp_path, governance_policy
) -> None:
    mutable = ("income",)

    class PassOnLowIncome:
        name = "mock"
        artefact_id = "mock"
        threshold = 0.5
        seen_keys: list[tuple[str, ...]] = []

        def score_application(self, features):
            from attack_lab.types import InternalDefenceResult

            self.seen_keys.append(tuple(sorted(features.keys())))
            decision = "PASS" if float(features["income"]) < 0.22 else "BLOCK"
            return InternalDefenceResult(
                risk_score=0.1 if decision == "PASS" else 0.9,
                threshold=self.threshold,
                decision=decision,
                runtime_ms=0.01,
                defender_name=self.name,
                artefact_id=self.artefact_id,
            )

    defender = PassOnLowIncome()
    env, _d, logger = _env(
        starting_case,
        mutable,
        tmp_path,
        governance_policy,
        max_attempts=8,
        defender=defender,
    )
    # Force a domain that can hit PASS quickly with a fixed seed search.
    domains = build_proposal_domains(
        mutable,
        categorical_vocabularies={},
        numeric_domains_config={
            "numeric_domains": {
                "income": {"kind": "float", "low": 0.2, "high": 0.21},
            }
        },
    )
    out = io.StringIO()
    result = ConstrainedRandomAttacker(
        env=env, domains=domains, seed=0, stdout=out
    ).run()
    text = out.getvalue()
    assert "public_feedback=" in text
    assert "risk_score" not in text
    assert "threshold" not in text.lower() or "frozen threshold" not in text
    assert "0.9" not in text
    assert result.success is True
    assert set(defender.seen_keys[0]) == set(starting_case.features)
    public = logger.public_transcript_path.read_text(encoding="utf-8")
    traj = logger.trajectory_path.read_text(encoding="utf-8")
    assert "risk_score" in traj
    assert "risk_score" not in public


def test_seeded_reproducibility(
    starting_case, tmp_path, governance_policy
) -> None:
    mutable = ("income", "customer_age")
    domains = _domains_for(mutable)

    def collect(seed: int, root: Path) -> list[dict]:
        env, _d, _logger = _env(
            starting_case,
            mutable,
            root,
            governance_policy,
            max_attempts=3,
        )
        attacker = ConstrainedRandomAttacker(
            env=env, domains=domains, seed=seed, stdout=io.StringIO()
        )
        return [attacker.propose().changes for _ in range(3)]

    assert collect(7, tmp_path / "r1") == collect(7, tmp_path / "r2")
    assert collect(7, tmp_path / "r3") != collect(8, tmp_path / "r4")


def test_complete_a0_episode_budget_exhaustion(
    starting_case, tmp_path, governance_policy
) -> None:
    mutable = ("income",)

    class AlwaysBlock:
        name = "mock"
        artefact_id = "mock"
        threshold = 0.5

        def score_application(self, features):
            from attack_lab.types import InternalDefenceResult

            return InternalDefenceResult(
                risk_score=0.99,
                threshold=0.5,
                decision="BLOCK",
                runtime_ms=0.01,
                defender_name=self.name,
                artefact_id=self.artefact_id,
            )

    env, _d, logger = _env(
        starting_case,
        mutable,
        tmp_path,
        governance_policy,
        max_attempts=2,
        defender=AlwaysBlock(),
    )
    result = ConstrainedRandomAttacker(
        env=env,
        domains=_domains_for(mutable),
        seed=3,
        stdout=io.StringIO(),
    ).run()
    assert result.success is False
    assert result.stop_reason == "q_exhausted"
    assert result.attempts_used == 2
    assert logger.trajectory_path.is_file()
