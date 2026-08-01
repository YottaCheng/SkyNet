"""A0 — constrained random search attacker (non-LLM, non-adaptive)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TextIO

import numpy as np

from attack_lab.domains import ProposalDomainSet
from attack_lab.environment import AttackEnvironment
from attack_lab.types import AttackProposal, EpisodeResult, PublicFeedback


@dataclass
class ConstrainedRandomAttacker:
    """Non-adaptive constrained random baseline (A0).

    Scientific role: estimate bypass success available from repeated valid
    random proposals without language reasoning, optimisation, memory,
    reflection, model internals, risk scores or previous-attempt learning.

    Development implementation decisions (not a final scientific freeze):
    - each attempt is sampled independently from the original starting case;
    - BLOCK / INVALID feedback is observed only for stopping/logging and is
      never used to bias later proposals;
    - early stop on public PASS is allowed.
    """

    env: AttackEnvironment
    domains: ProposalDomainSet
    seed: int
    stdout: TextIO
    rng: np.random.Generator = field(init=False)
    name: str = "a0_constrained_random"
    _public_feedback_seen: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def propose(self) -> AttackProposal:
        """Generate one independent proposal from the original starting case."""
        baseline = self.env.starting_case.features
        mutable = self.env.validator.mutable_fields
        changes = self.domains.sample_changes(mutable, baseline, self.rng)
        return AttackProposal(
            changes=changes,
            raw_command=f"{self.name}:seed={self.seed}:independent_redraw",
        )

    def _consume_public_feedback(self, feedback: PublicFeedback) -> None:
        """Record public labels only; never adapt the proposal policy."""
        self._public_feedback_seen.append(feedback.label)
        # Intentionally ignore BLOCK/INVALID content for search adaptation.
        # Remaining-attempt information may be used only for loop control via
        # the shared environment; proposal generation stays independent.

    def run(self) -> EpisodeResult:
        """Drive an A0 episode until PASS or budget exhaustion."""
        self.stdout.write(
            f"\n=== A0 ConstrainedRandomAttacker "
            f"(seed={self.seed}, case={self.env.starting_case.case_id}) ===\n"
            "Development rule: independent redraws from the original starting "
            "case; public label_only feedback only; no BLOCK-guided search.\n"
        )
        self.stdout.flush()

        while not self.env.done:
            obs = self.env.observation()
            # Attacker-visible observation is available, but A0 proposal
            # generation does not use last_feedback or prior candidates.
            _ = obs.last_feedback
            proposal = self.propose()
            self.stdout.write(
                f"attempt {obs.attempt}/{obs.max_attempts}: "
                f"submitting independent random proposal over "
                f"{list(proposal.changes)}\n"
            )
            self.stdout.flush()
            record = self.env.step(proposal)
            public = record.public_feedback
            self._consume_public_feedback(public)
            self.stdout.write(
                f"  public_feedback={public.label} "
                f"remaining_attempts={public.remaining_attempts}\n"
            )
            self.stdout.flush()
            # Early stop is handled by the environment on PASS.
            if public.label == "PASS":
                break

        result = self.env.result()
        self.stdout.write(
            f"Episode finished: success={result.success}, "
            f"reason={result.stop_reason}, attempts={result.attempts_used}\n"
        )
        self.stdout.flush()
        return result
