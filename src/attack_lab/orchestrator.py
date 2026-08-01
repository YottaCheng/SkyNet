"""Minimal “electronic cricket-fight” match orchestrator skeleton.

This module provides a uniform episode harness shared by future A0–A3
attackers.  It does not implement A1–A3 search algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from attack_lab.budget import BudgetSpec
from attack_lab.cases import StartingCase
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.governance import CompiledGovernancePolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.types import EpisodeResult, StepRecord, to_jsonable
from attack_lab.validator import ConstraintValidator


class MatchAttacker(Protocol):
    """Minimal attacker interface used by the orchestrator."""

    attacker_id: str

    def run(self, env: AttackEnvironment) -> None:
        """Drive submissions exclusively through ``env.step`` until stop."""


@dataclass(frozen=True)
class MatchConfig:
    """Shared match conditions for one attacker-versus-defence episode."""

    attacker_id: str
    anchor: StartingCase
    policy: CompiledGovernancePolicy
    budget: BudgetSpec
    feedback_policy: FeedbackPolicy
    defender: Any
    seed: int
    enabled_action_keys: tuple[str, ...] | None = None
    logger: TrajectoryLogger | None = None


@dataclass(frozen=True)
class MatchResult:
    """Uniform match outcome consumed by later comparative analysis."""

    attacker_id: str
    anchor_id: str
    success: bool
    stop_reason: str
    q_used: int
    e_used: int
    scored_defender_queries: int
    attempts_to_success: int | None
    invalid_submissions: int
    unique_fields_ever_manipulated: tuple[str, ...]
    trajectory: tuple[StepRecord, ...]
    policy_fingerprint: str
    budget_spec: dict[str, Any]
    seed: int
    episode: EpisodeResult

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "attacker_id": self.attacker_id,
                "anchor_id": self.anchor_id,
                "success": self.success,
                "stop_reason": self.stop_reason,
                "q_used": self.q_used,
                "e_used": self.e_used,
                "scored_defender_queries": self.scored_defender_queries,
                "attempts_to_success": self.attempts_to_success,
                "invalid_submissions": self.invalid_submissions,
                "unique_fields_ever_manipulated": list(
                    self.unique_fields_ever_manipulated
                ),
                "trajectory": self.trajectory,
                "policy_fingerprint": self.policy_fingerprint,
                "budget_spec": self.budget_spec,
                "seed": self.seed,
            }
        )


@dataclass
class MatchOrchestrator:
    """Run one fair episode under identical shared conditions."""

    def run_episode(
        self,
        attacker: MatchAttacker,
        config: MatchConfig,
    ) -> MatchResult:
        if attacker.attacker_id != config.attacker_id:
            raise ValueError(
                f"Attacker id mismatch: attacker={attacker.attacker_id!r}, "
                f"config={config.attacker_id!r}."
            )
        if config.anchor.initial_decision != "BLOCK":
            return MatchResult(
                attacker_id=config.attacker_id,
                anchor_id=config.anchor.case_id,
                success=False,
                stop_reason="invalid_environment",
                q_used=0,
                e_used=0,
                scored_defender_queries=0,
                attempts_to_success=None,
                invalid_submissions=0,
                unique_fields_ever_manipulated=(),
                trajectory=(),
                policy_fingerprint=config.policy.policy_fingerprint,
                budget_spec=config.budget.to_dict(),
                seed=config.seed,
                episode=EpisodeResult(
                    case_id=config.anchor.case_id,
                    success=False,
                    attempts_used=0,
                    max_attempts=config.budget.q_max,
                    stop_reason="invalid_environment",
                    budget_spec=config.budget.to_dict(),
                ),
            )

        try:
            validator = ConstraintValidator.from_policy(
                config.policy,
                enabled_action_keys=config.enabled_action_keys,
            )
        except Exception:  # noqa: BLE001
            return MatchResult(
                attacker_id=config.attacker_id,
                anchor_id=config.anchor.case_id,
                success=False,
                stop_reason="policy_error",
                q_used=0,
                e_used=0,
                scored_defender_queries=0,
                attempts_to_success=None,
                invalid_submissions=0,
                unique_fields_ever_manipulated=(),
                trajectory=(),
                policy_fingerprint=config.policy.policy_fingerprint,
                budget_spec=config.budget.to_dict(),
                seed=config.seed,
                episode=EpisodeResult(
                    case_id=config.anchor.case_id,
                    success=False,
                    attempts_used=0,
                    max_attempts=config.budget.q_max,
                    stop_reason="policy_error",
                    budget_spec=config.budget.to_dict(),
                ),
            )

        logger = config.logger
        if logger is None:
            logger = TrajectoryLogger.create(
                run_id=(
                    f"match_{config.attacker_id}_"
                    f"{config.anchor.case_id}_seed{config.seed}"
                )
            )

        env = AttackEnvironment(
            starting_case=config.anchor,
            defender=config.defender,
            validator=validator,
            feedback_policy=config.feedback_policy,
            logger=logger,
            budget=config.budget,
        )
        attacker.run(env)
        if not env.done:
            env.abort(reason="attacker_stopped")
        episode = env.result()
        return MatchResult(
            attacker_id=config.attacker_id,
            anchor_id=config.anchor.case_id,
            success=episode.success,
            stop_reason=episode.stop_reason,
            q_used=episode.q_used,
            e_used=episode.e_used,
            scored_defender_queries=episode.scored_defender_queries,
            attempts_to_success=episode.attempts_to_success,
            invalid_submissions=episode.invalid_submissions,
            unique_fields_ever_manipulated=episode.unique_fields_ever_manipulated,
            trajectory=episode.steps,
            policy_fingerprint=config.policy.policy_fingerprint,
            budget_spec=config.budget.to_dict(),
            seed=config.seed,
            episode=episode,
        )


@dataclass
class ScriptedAttacker:
    """Test/development attacker that replays fixed proposals via env.step."""

    attacker_id: str
    proposals: tuple[Any, ...] = field(default_factory=tuple)

    def run(self, env: AttackEnvironment) -> None:
        for proposal in self.proposals:
            if env.done:
                break
            env.step(proposal)
