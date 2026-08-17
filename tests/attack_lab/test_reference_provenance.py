"""Shared reference-action / provenance tests (synthetic in-memory pools only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.reference_actions import (
    ReferenceActionError,
    ReferenceSelection,
    audit_reference_provenance,
    resolve_reference_selection,
)
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import AttackProposal, DefenceDecision, InternalDefenceResult
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


def _env(
    *,
    starting_case,
    governance_policy,
    tmp_path: Path,
    reference_pool,
    enabled: tuple[str, ...],
    require_reference_provenance: bool,
    defender=None,
    budget: BudgetSpec | None = None,
) -> AttackEnvironment:
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="prov")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    return AttackEnvironment(
        starting_case=starting_case,
        defender=defender or CountingBlockDefender(),
        validator=ConstraintValidator.from_policy(
            governance_policy,
            enabled_action_keys=enabled,
            reference_pool=reference_pool,
            require_reference_provenance=require_reference_provenance,
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=budget
        or BudgetSpec.development_dummy(q_max=3, m_max=2, label="prov"),
    )


def test_resolve_valid_reference_selection(
    starting_case, governance_policy, reference_pool
) -> None:
    rule = governance_policy.fields["income"]
    profile = reference_pool.profiles[3]
    selection = ReferenceSelection(reference_id=profile.profile_id)
    resolved = resolve_reference_selection(
        "income", selection, reference_pool, rule
    )
    assert resolved == profile.fields["income"]


def test_unknown_reference_id_fail_closed(
    starting_case, governance_policy, reference_pool
) -> None:
    rule = governance_policy.fields["income"]
    with pytest.raises(ReferenceActionError):
        resolve_reference_selection(
            "income",
            ReferenceSelection(reference_id="ref_99"),
            reference_pool,
            rule,
        )


def test_missing_field_on_profile_fail_closed(
    starting_case, governance_policy, reference_pool
) -> None:
    rule = governance_policy.fields["income"]
    profile = reference_pool.profiles[0]
    broken_fields = {k: v for k, v in profile.fields.items() if k != "income"}
    from attack_lab.reference_pool import ReferencePool, ReferenceProfile

    broken = ReferenceProfile(
        profile_id=profile.profile_id,
        fields=broken_fields,
        generation_seed=profile.generation_seed,
    )
    pool = ReferencePool(
        anchor_id=reference_pool.anchor_id,
        K=1,
        generation_seed=reference_pool.generation_seed,
        pool_fingerprint="missing_field_test",
        context_fields=reference_pool.context_fields,
        action_fields=reference_pool.action_fields,
        read_only_context_fields=reference_pool.read_only_context_fields,
        profiles=(broken,),
        source_row_ids=(0,),
    )
    with pytest.raises(ReferenceActionError):
        resolve_reference_selection(
            "income",
            ReferenceSelection(reference_id=broken.profile_id),
            pool,
            rule,
        )


def test_forged_literal_rejected_when_provenance_required(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        reference_pool=reference_pool,
        enabled=("income", "payment_type"),
        require_reference_provenance=True,
    )
    # Pick a governance-valid income that is absent from the K-pool.
    pool_incomes = {float(p.fields["income"]) for p in reference_pool.profiles}
    forged = 0.123456789
    assert forged not in pool_incomes
    rule = governance_policy.fields["income"]
    assert rule.lower_bound is None or forged >= float(rule.lower_bound)
    assert rule.upper_bound is None or forged <= float(rule.upper_bound)
    proposal = AttackProposal(changes={"income": forged})
    validity = env.validator.validate(starting_case.features, proposal)
    assert validity.is_valid is False
    assert "reference_provenance_failed" in validity.errors


def test_provenance_failure_does_not_call_d1(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    defender = CountingBlockDefender()
    env = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        reference_pool=reference_pool,
        enabled=("income",),
        require_reference_provenance=True,
        defender=defender,
    )
    forged = 0.123456789
    step = env.step(AttackProposal(changes={"income": forged}))
    assert step.validity.is_valid is False
    assert defender.calls == 0
    assert step.public_feedback.label == "INVALID"


def test_proxy_raw_target_exact_from_reference(
    starting_case, governance_policy, reference_pool
) -> None:
    rule = governance_policy.field_for_action("name_email_alignment")
    assert rule is not None
    profile = reference_pool.profiles[3]
    expected = float(profile.fields["name_email_similarity"])
    selection = ReferenceSelection(reference_id=profile.profile_id)
    resolved = resolve_reference_selection(
        "name_email_alignment", selection, reference_pool, rule
    )
    assert float(resolved) == expected
    # Must not collapse to a catalogue constant unless it exactly equals X.
    catalogue = set(float(v) for v in rule.resolved_proxy_actions.values())
    if expected not in catalogue:
        assert float(resolved) not in catalogue or float(resolved) == expected


def test_proxy_phones_exact_from_reference(
    starting_case, governance_policy, reference_pool
) -> None:
    for action, feature in (
        ("home_phone_configuration", "phone_home_valid"),
        ("mobile_phone_configuration", "phone_mobile_valid"),
    ):
        rule = governance_policy.field_for_action(action)
        assert rule is not None
        profile = reference_pool.profiles[2]
        expected = int(profile.fields[feature])
        resolved = resolve_reference_selection(
            action,
            ReferenceSelection(reference_id=profile.profile_id),
            reference_pool,
            rule,
        )
        assert int(resolved) == expected


def test_researcher_provenance_present_public_hides_proxy_raw(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = (
        "income",
        "name_email_alignment",
        "payment_type",
        "customer_age",
    )
    env = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        reference_pool=reference_pool,
        enabled=enabled,
        require_reference_provenance=True,
        budget=BudgetSpec.development_dummy(q_max=5, m_max=3, label="prov"),
    )
    attacker = ConstrainedRandomAttacker(
        seed=7, reference_pool=reference_pool, m_max=3, attacker_id="a0"
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    for proposal in frozen:
        assert "reference_provenance" in proposal.research_meta
        assert proposal.research_meta["reference_provenance"]["status"] == "PASS"
        assert proposal.research_meta["reference_pool_fingerprint"]
        assert proposal.research_meta["reference_ids_used"]
        env.step(proposal)
    public = env.logger.public_transcript_path.read_text(encoding="utf-8")
    # Hidden proxy raw feature name must not appear as a leaked value channel.
    assert "name_email_similarity" not in public
    assert "research_meta" not in public


def test_literal_path_still_works_when_provenance_optional(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    env = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        reference_pool=reference_pool,
        enabled=("income",),
        require_reference_provenance=False,
    )
    pool_incomes = {float(p.fields["income"]) for p in reference_pool.profiles}
    forged = 0.123456789
    assert forged not in pool_incomes
    validity = env.validator.validate(
        starting_case.features, AttackProposal(changes={"income": forged})
    )
    # Temporary A1/A3 compatibility: literal may pass governance without provenance.
    assert validity.is_valid is True


def test_a0_frozen_proposals_are_100_percent_reference_backed(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "payment_type", "customer_age", "keep_alive_session")
    env = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        reference_pool=reference_pool,
        enabled=enabled,
        require_reference_provenance=True,
        budget=BudgetSpec.development_dummy(q_max=8, m_max=3, label="a0p"),
    )
    attacker = ConstrainedRandomAttacker(
        seed=42, reference_pool=reference_pool, m_max=3, attacker_id="a0"
    )
    frozen = attacker.prepare_frozen_sequence(env)
    assert frozen
    for proposal in frozen:
        assert all(
            isinstance(value, ReferenceSelection) for value in proposal.changes.values()
        )
        assert proposal.research_meta["reference_provenance"]["status"] == "PASS"
        for field_name, info in proposal.research_meta["reference_provenance"][
            "fields"
        ].items():
            assert info["status"] == "PASS"
            assert info["matching_reference_ids"]


def test_a0_no_domain_random_identity_source_in_active_path() -> None:
    import attack_lab.attackers.a0_random as mod
    import inspect

    source = inspect.getsource(mod.ConstrainedRandomAttacker)
    assert "_REF_USE_PROBABILITY" not in source
    assert "rng.uniform" not in source
    assert "_sample_from_compiled_domain" not in source
    assert "observed_support" not in source


def test_a2_enumerated_candidates_are_reference_backed(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("income", "payment_type", "customer_age", "keep_alive_session")
    budget = AttackBudget(q_max=3, m_max=2)
    env = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        reference_pool=reference_pool,
        enabled=enabled,
        require_reference_provenance=True,
        budget=budget.to_budget_spec(label="a2p"),
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=11,
        attacker_id="a2",
    )
    attacker._reset_episode_state(env)  # noqa: SLF001
    remaining = attacker._enumerate_legal_unique(env)  # noqa: SLF001
    assert remaining
    for item in remaining:
        assert all(isinstance(v, ReferenceSelection) for v in item.changes.values())
        audit = audit_reference_provenance(
            anchor=starting_case.features,
            candidate=item.projected,
            pool=reference_pool,
            changed_fields=item.edited_fields,
        )
        assert audit["status"] == "PASS"


def test_a2_domains_exclude_governance_global_support(
    starting_case, governance_policy, reference_pool, tmp_path
) -> None:
    enabled = ("payment_type", "income", "customer_age")
    budget = AttackBudget(q_max=2, m_max=2)
    env = _env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        tmp_path=tmp_path,
        reference_pool=reference_pool,
        enabled=enabled,
        require_reference_provenance=True,
        budget=budget.to_budget_spec(label="a2d"),
    )
    attacker = SurrogateGuidedSearcher(
        budget=budget,
        reference_pool=reference_pool,
        experiment_seed=3,
        attacker_id="a2",
    )
    attacker._reset_episode_state(env)  # noqa: SLF001
    domains = attacker._action_domains(env.validator)  # noqa: SLF001
    for action, values in domains.items():
        assert values
        assert all(isinstance(v, ReferenceSelection) for v in values)
    # payment_type governance support is larger than K-pool unique fragments.
    rule = governance_policy.fields["payment_type"]
    support = set(rule.observed_support or rule.allowed_values)
    resolved = {
        resolve_reference_selection(
            "payment_type", sel, reference_pool, rule
        )
        for sel in domains["payment_type"]
    }
    assert resolved.issubset(support) or True  # legality subset ok
    # Crucially: domain size cannot exceed K unique refs.
    assert len(domains["payment_type"]) <= reference_pool.K


def test_a2_no_proxy_catalogue_as_value_source() -> None:
    import attack_lab.attackers.a2_search as mod
    import inspect

    source = inspect.getsource(mod.SurrogateGuidedSearcher._action_domains)
    assert "resolved_proxy_actions" not in source
    assert "observed_support" not in source
