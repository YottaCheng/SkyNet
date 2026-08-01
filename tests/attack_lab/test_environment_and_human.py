"""Episode rules, logging separation, and human parsing tests."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.budget import BudgetSpec
from attack_lab.environment import AttackEnvironment
from attack_lab.human import HumanAttacker, HumanAttackerError
from attack_lab.logger import TrajectoryLogger
from attack_lab.types import AttackProposal, DefenceDecision, InternalDefenceResult


@dataclass
class MockDefender:
    """Deterministic defender for unit tests (no frozen artefact required)."""

    name: str = "mock_d1"
    artefact_id: str = "mock_artefact"
    threshold: float = 0.5
    block_unless: Mapping[str, Any] | None = None
    last_feature_keys: tuple[str, ...] = field(default_factory=tuple, init=False)
    last_features: dict[str, Any] = field(default_factory=dict, init=False)

    def fit(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        raise RuntimeError("MockDefender has no training/refit path.")

    def score_application(self, features: Mapping[str, Any]) -> InternalDefenceResult:
        self.last_feature_keys = tuple(sorted(features.keys()))
        self.last_features = dict(features)
        decision: DefenceDecision = "BLOCK"
        score = 0.9
        if self.block_unless is not None:
            if all(features.get(k) == v for k, v in self.block_unless.items()):
                decision = "PASS"
                score = 0.1
        return InternalDefenceResult(
            risk_score=score,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=0.01,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )


def _make_env(
    starting_case,
    validator,
    feedback_policy,
    tmp_path: Path,
    *,
    max_attempts: int = 3,
    defender: MockDefender | None = None,
) -> tuple[AttackEnvironment, MockDefender, TrajectoryLogger]:
    defender = defender or MockDefender(block_unless={"income": 0.2})
    # Isolate logger under tmp via monkeypatched root in caller.
    logger = TrajectoryLogger(run_dir=tmp_path / "run1", run_id="run1")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    env = AttackEnvironment(
        starting_case=starting_case,
        defender=defender,
        validator=validator,
        feedback_policy=feedback_policy,
        max_attempts=max_attempts,
        logger=logger,
        budget=BudgetSpec.development_dummy(
            q_max=max_attempts,
            e_max=1_000_000,
            label="test_dummy_budget_not_final",
        ),
    )
    return env, defender, logger


def test_attacker_metadata_not_passed_into_model_features(
    starting_case, validator, feedback_policy, tmp_path
) -> None:
    env, defender, _logger = _make_env(
        starting_case, validator, feedback_policy, tmp_path
    )
    env.step(AttackProposal(changes={"income": 0.2}))
    assert set(defender.last_feature_keys) == set(starting_case.features)
    assert "attacker" not in defender.last_features
    assert "attempt" not in defender.last_features
    assert "case_id" not in defender.last_features
    assert "remaining_budget" not in defender.last_features


def test_success_stopping_rule(
    starting_case, validator, feedback_policy, tmp_path
) -> None:
    env, _defender, _logger = _make_env(
        starting_case, validator, feedback_policy, tmp_path, max_attempts=5
    )
    record = env.step(AttackProposal(changes={"income": 0.2}))
    assert record.success is True
    assert record.public_feedback.label == "PASS"
    assert env.done is True
    result = env.result()
    assert result.success is True
    assert result.stop_reason == "success"


def test_budget_exhaustion_stopping_rule(
    starting_case, validator, feedback_policy, tmp_path
) -> None:
    defender = MockDefender(block_unless={"income": 999.0})  # never matches
    env, _d, _logger = _make_env(
        starting_case,
        validator,
        feedback_policy,
        tmp_path,
        max_attempts=2,
        defender=defender,
    )
    env.step(AttackProposal(changes={"income": 0.2}))
    env.step(AttackProposal(changes={"income": 0.3}))
    assert env.done is True
    result = env.result()
    assert result.success is False
    assert result.stop_reason == "q_exhausted"
    assert result.attempts_used == 2


def test_invalid_attempt_logged_not_scored(
    starting_case, validator, feedback_policy, tmp_path
) -> None:
    env, defender, logger = _make_env(
        starting_case, validator, feedback_policy, tmp_path
    )
    record = env.step(AttackProposal(changes={"customer_age": 40}))
    assert record.public_feedback.label == "INVALID"
    assert record.internal_defence is None
    assert not hasattr(defender, "last_features") or "customer_age" not in getattr(
        defender, "last_features", {}
    )
    # Invalid still consumes an attempt in this development environment.
    assert env.attempts_used == 1
    text = logger.trajectory_path.read_text(encoding="utf-8")
    assert "INVALID" in text or "immutable" in text


def test_trajectory_and_public_transcript_differ(
    starting_case, validator, feedback_policy, tmp_path
) -> None:
    env, _defender, logger = _make_env(
        starting_case, validator, feedback_policy, tmp_path
    )
    env.step(AttackProposal(changes={"income": 0.2}))
    env.result()
    trajectory = logger.trajectory_path.read_text(encoding="utf-8")
    public = logger.public_transcript_path.read_text(encoding="utf-8")
    assert "risk_score" in trajectory
    assert "threshold" in trajectory
    assert "risk_score" not in public
    assert "threshold" not in public
    assert "PASS" in public


def test_human_attacker_input_parsing(starting_case, validator, feedback_policy, tmp_path) -> None:
    env, _d, _logger = _make_env(starting_case, validator, feedback_policy, tmp_path)
    attacker = HumanAttacker(
        env=env, stdin=io.StringIO(""), stdout=io.StringIO()
    )
    assert attacker.parse_line("quit") == "quit"
    assert attacker.parse_line("show") == "show"
    assert attacker.parse_line("reset-current-proposal") == "reset"
    assert attacker.parse_line("income=0.25") is None
    assert attacker.pending_changes["income"] == 0.25
    proposal = attacker.parse_line("submit")
    assert isinstance(proposal, AttackProposal)
    assert proposal.changes["income"] == 0.25
    with pytest.raises(HumanAttackerError):
        attacker.parse_line("not_a_command")


def test_complete_episode_with_mock_defender(
    starting_case, validator, feedback_policy, tmp_path
) -> None:
    env, _defender, logger = _make_env(
        starting_case, validator, feedback_policy, tmp_path, max_attempts=3
    )
    stdin = io.StringIO("income=0.2\nsubmit\n")
    stdout = io.StringIO()
    HumanAttacker(env=env, stdin=stdin, stdout=stdout).run()
    out = stdout.getvalue()
    assert "PASS" in out
    assert "0.9" not in out  # mock internal score must not appear in attacker view
    assert "threshold" not in out.lower()
    assert logger.trajectory_path.is_file()
    assert logger.public_transcript_path.is_file()
    assert (logger.run_dir / "episode_result.json").is_file()
