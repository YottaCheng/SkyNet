"""Fairness fix: abstract proxy actions in A1 V4.3 / A3 V2.3 catalogues.

Restores governance-approved abstract keys without exposing trusted proxy
raw feature values or weakening exact K-pool provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack_lab.archive.contracts.a1_v4_1_contract import build_v4_1_action_slots
from attack_lab.attackers.a1_v4_3_contract import build_v4_3_prompt_payload
from attack_lab.archive.contracts.a1_v4_contract import (
    PROXY_RAW_FEATURE_NAMES,
    _enabled_actions_by_category,
    build_v4_choice_catalog,
    build_v4_static_plan_options,
    resolve_choice_ids_to_changes,
)
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
from attack_lab.archive.contracts.a3_v2_1_contract import (
    public_slot_entries,
    writable_slots_from_episode_map,
)
from attack_lab.attackers.a3_v2_3_contract import (
    build_a3_v2_3_episode_action_slots,
    build_a3_v2_3_prompt_payload,
)
from attack_lab.archive.contracts.a3_v2_contract import resolve_a3_v2_selections
from attack_lab.budget import AttackBudget
from attack_lab.environment import AttackEnvironment
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.public_reference_view import (
    build_canonical_public_reference_view,
    choice_public_value_lookup,
    public_safe_reference_field_names,
)
from attack_lab.reference_actions import (
    ReferenceSelection,
    audit_reference_provenance,
    provenance_audit_counts,
    raw_feature_for_provenance_field,
    resolve_reference_selection,
)
from attack_lab.reference_pool import ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import AttackProposal
from attack_lab.validator import ConstraintValidator
from test_a1_planner import CountingBlockDefender, _qm_budget

ABSTRACT_PROXY_ACTIONS = frozenset(
    {
        "name_email_alignment",
        "home_phone_configuration",
        "mobile_phone_configuration",
    }
)
PROXY_RAWS = frozenset(PROXY_RAW_FEATURE_NAMES)


@pytest.fixture()
def reference_pool(synthetic_frame, starting_case):
    train = synthetic_frame.loc[synthetic_frame["month"].between(0, 5)].copy()
    return ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=train
    ).get_pool(starting_case.case_id)


def _full_enabled(governance_policy) -> tuple[str, ...]:
    return tuple(governance_policy.available_action_keys)


def _make_env(
    *,
    starting_case,
    governance_policy,
    reference_pool,
    tmp_path: Path,
    enabled: tuple[str, ...] | None = None,
    require_reference_provenance: bool = True,
):
    keys = enabled if enabled is not None else _full_enabled(governance_policy)
    logger = TrajectoryLogger(run_dir=tmp_path / "env", run_id="env")
    logger.run_dir.mkdir(parents=True, exist_ok=True)
    return AttackEnvironment(
        starting_case=starting_case,
        defender=CountingBlockDefender(),
        validator=ConstraintValidator.from_policy(
            governance_policy,
            enabled_action_keys=keys,
            reference_pool=reference_pool,
            require_reference_provenance=require_reference_provenance,
        ),
        feedback_policy=FeedbackPolicy(mode="label_only"),
        logger=logger,
        budget=_qm_budget(5, 2),
    )


def test_shared_intended_abstract_action_keys_include_proxy(
    starting_case, governance_policy, reference_pool, tmp_path
):
    enabled = _full_enabled(governance_policy)
    assert ABSTRACT_PROXY_ACTIONS.issubset(set(enabled))
    assert not (PROXY_RAWS & set(enabled))

    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "shared",
        enabled=enabled,
    )
    a0_keys = set(env.validator.enabled_action_keys)

    a2 = SurrogateGuidedSearcher(
        budget=AttackBudget(q_max=5, m_max=2),
        reference_pool=reference_pool,
        experiment_seed=1,
    )
    a2._reset_episode_state(env)
    a2_keys = set(env.validator.enabled_action_keys)
    assert a2_keys == a0_keys
    # A2 domains are keyed by the same enabled abstract actions.
    domains = a2._action_domains(env.validator)
    assert ABSTRACT_PROXY_ACTIONS.issubset(set(domains))

    static, per_attempt = _enabled_actions_by_category(env.validator)
    a1_keys = set(static) | set(per_attempt)
    assert a1_keys == a0_keys
    assert ABSTRACT_PROXY_ACTIONS.issubset(a1_keys)
    assert not (a1_keys & PROXY_RAWS)

    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
    )
    episode_slots = build_a3_v2_3_episode_action_slots(
        catalog, validator=env.validator
    )
    a3_keys = {
        episode_slots.get(sid).action_key  # type: ignore[union-attr]
        for sid in episode_slots.ordered_slot_ids
    }
    # Slot map covers every catalogue action that has ≥1 legal ref-backed choice.
    catalog_keys = {
        catalog.get(cid).action_key  # type: ignore[union-attr]
        for cid in list(catalog.static_choice_ids) + list(catalog.per_attempt_choice_ids)
    }
    assert catalog_keys == a3_keys
    assert ABSTRACT_PROXY_ACTIONS.issubset(catalog_keys)
    assert not (catalog_keys & PROXY_RAWS)
    # Intended governance set equals A0/A1/A2 enabled keys; catalogue may omit a
    # key only when the current pool has no change-from-anchor selection.
    missing = a0_keys - catalog_keys
    for action in missing:
        rule = env.validator.policy.field_for_action(action)
        assert rule is not None
        from attack_lab.reference_actions import reference_backed_selections_for_action

        sels = reference_backed_selections_for_action(
            action_key=action,
            pool=reference_pool,
            rule=rule,
            anchor_value=env.starting_case.features.get(rule.feature),
            require_change_from_anchor=True,
        )
        assert sels == ()


def test_a1_v4_3_and_a3_v2_3_include_abstract_proxy_without_raw_exposure(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "payloads",
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
    )
    catalog_keys = {
        catalog.get(cid).action_key  # type: ignore[union-attr]
        for cid in list(catalog.static_choice_ids) + list(catalog.per_attempt_choice_ids)
    }
    assert ABSTRACT_PROXY_ACTIONS.issubset(catalog_keys)

    plans = build_v4_static_plan_options(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
        catalog=catalog,
        m_max=2,
        q_max=5,
    )
    slots = build_v4_1_action_slots(catalog)
    a1_payload = build_v4_3_prompt_payload(
        validator=env.validator,
        pool=reference_pool,
        budget=AttackBudget(q_max=5, m_max=2),
        q_max=5,
        visible_anchor=env.observation().visible_fields,
        case_id=env.starting_case.case_id,
        catalog=catalog,
        static_plans=plans,
        action_slots=slots,
    )
    a1_blob = json.dumps(a1_payload, sort_keys=True)
    for raw in PROXY_RAWS:
        assert f'"{raw}"' not in a1_blob
    a1_actions = {item["action_key"] for item in a1_payload["choice_catalogue"]}
    assert ABSTRACT_PROXY_ACTIONS.issubset(a1_actions)

    episode_slots = build_a3_v2_3_episode_action_slots(
        catalog, validator=env.validator
    )
    writable = writable_slots_from_episode_map(
        episode_slots, validator=env.validator, include_static=True
    )
    a3_payload = build_a3_v2_3_prompt_payload(
        case_id=env.starting_case.case_id,
        visible_anchor=env.validator.visible_fields(env.starting_case.features),
        current_application=env.observation().visible_fields,
        budget=AttackBudget(q_max=5, m_max=2),
        q_remaining=5,
        query_index=1,
        static_edit_cost=0,
        residual_m=2,
        locked_static_values={},
        slots=writable,
        slot_entries=public_slot_entries(writable, validator=env.validator),
        episodic_memory=[],
        pool=reference_pool,
        catalog=catalog,
        episode_slot_map=public_slot_entries(episode_slots, validator=env.validator),
    )
    a3_blob = json.dumps(a3_payload, sort_keys=True)
    for raw in PROXY_RAWS:
        assert f'"{raw}"' not in a3_blob
    mapping_actions = {
        item["action_key"] for item in a3_payload["choice_to_reference_mapping"]
    }
    assert ABSTRACT_PROXY_ACTIONS.issubset(mapping_actions)

    public = build_canonical_public_reference_view(reference_pool)
    assert not (set(public_safe_reference_field_names(reference_pool)) & PROXY_RAWS)
    for profile in public["profiles"]:
        assert not (set(profile["fields"]) & PROXY_RAWS)

    # Abstract proxy choices must not leak recoverable raw public values.
    proxy_choice = next(
        item
        for item in a1_payload["choice_catalogue"]
        if item["action_key"] in ABSTRACT_PROXY_ACTIONS
    )
    assert (
        choice_public_value_lookup(
            catalog_choices=a1_payload["choice_catalogue"],
            public_view=public,
            choice_id=proxy_choice["choice_id"],
        )
        is None
    )


def test_abstract_proxy_choices_resolve_only_via_trusted_reference_selection(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "resolve",
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
    )
    proxy_choice_ids = [
        cid
        for cid in list(catalog.static_choice_ids) + list(catalog.per_attempt_choice_ids)
        if catalog.get(cid) is not None
        and catalog.get(cid).action_key in ABSTRACT_PROXY_ACTIONS  # type: ignore[union-attr]
    ]
    assert proxy_choice_ids
    choice = catalog.get(proxy_choice_ids[0])
    assert choice is not None
    assert choice.action_key in ABSTRACT_PROXY_ACTIONS

    changes, err = resolve_choice_ids_to_changes([choice.choice_id], catalog)
    assert err == ""
    assert changes is not None
    assert set(changes) == {choice.action_key}
    selection = changes[choice.action_key]
    assert isinstance(selection, ReferenceSelection)

    rule = env.validator.policy.field_for_action(choice.action_key)
    assert rule is not None
    assert rule.agent_action_mode == "proxy_action"
    assert rule.feature in PROXY_RAWS
    resolved = resolve_reference_selection(
        choice.action_key, selection, reference_pool, rule
    )
    # Resolved value equals exact pool fragment for that reference_id.
    profile = next(
        p for p in reference_pool.profiles if p.profile_id == selection.reference_id
    )
    assert resolved == profile.fields[rule.feature]

    # A3 slot resolution also emits ReferenceSelection only.
    episode_slots = build_a3_v2_3_episode_action_slots(
        catalog, validator=env.validator
    )
    slot = next(
        episode_slots.get(sid)
        for sid in episode_slots.ordered_slot_ids
        if episode_slots.get(sid)
        and episode_slots.get(sid).action_key == choice.action_key  # type: ignore[union-attr]
    )
    assert slot is not None
    a3_changes, a3_err = resolve_a3_v2_selections(
        {slot.action_slot_id: choice.choice_id},
        slots=episode_slots,
        catalog=catalog,
    )
    assert a3_err == ""
    assert a3_changes is not None
    assert isinstance(a3_changes[choice.action_key], ReferenceSelection)


def test_proxy_submission_keeps_exact_k10_provenance_and_rejects_freeform(
    starting_case, governance_policy, reference_pool, tmp_path
):
    env = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "submit",
        require_reference_provenance=True,
    )
    catalog = build_v4_choice_catalog(
        validator=env.validator,
        pool=reference_pool,
        anchor=env.starting_case.features,
    )
    choice = next(
        catalog.get(cid)
        for cid in list(catalog.static_choice_ids) + list(catalog.per_attempt_choice_ids)
        if catalog.get(cid) is not None
        and catalog.get(cid).action_key == "home_phone_configuration"  # type: ignore[union-attr]
    )
    assert choice is not None
    proposal = AttackProposal(
        changes={choice.action_key: ReferenceSelection(reference_id=choice.reference_id)}
    )
    step = env.step(proposal)
    assert step.validity.is_valid
    assert env.attempts_used == 1
    assert step.submitted_edit_cost <= 2

    rule = env.validator.policy.field_for_action(choice.action_key)
    assert rule is not None
    candidate = step.validity.candidate_features
    assert candidate is not None
    audit = audit_reference_provenance(
        anchor=env.starting_case.features,
        candidate=candidate,
        pool=reference_pool,
        changed_fields=[rule.feature],
    )
    assert audit["status"] == "PASS"
    assert rule.feature in audit["fields"]
    assert audit["fields"][rule.feature]["status"] == "PASS"
    assert audit["fields"][rule.feature]["matching_reference_ids"]
    assert choice.reference_id in audit["fields"][rule.feature]["matching_reference_ids"]

    abstract_audit = audit_reference_provenance(
        anchor=env.starting_case.features,
        candidate=candidate,
        pool=reference_pool,
        changed_fields=[choice.action_key],
    )
    assert abstract_audit["status"] == "PASS"
    assert rule.feature in abstract_audit["fields"]
    assert choice.action_key not in abstract_audit["fields"]
    assert abstract_audit["fields"][rule.feature]["status"] == "PASS"
    assert (
        abstract_audit["fields"][rule.feature]["matching_reference_ids"]
        == audit["fields"][rule.feature]["matching_reference_ids"]
    )

    # Free-form / synthesised raw values cannot enter under provenance enforcement.
    env2 = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "freeform",
        require_reference_provenance=True,
    )
    bad = AttackProposal(changes={"home_phone_configuration": 1})
    bad_step = env2.step(bad)
    assert not bad_step.validity.is_valid
    # Raw proxy feature name is not an enabled attacker-facing action key.
    env3 = _make_env(
        starting_case=starting_case,
        governance_policy=governance_policy,
        reference_pool=reference_pool,
        tmp_path=tmp_path / "raw_key",
        require_reference_provenance=True,
    )
    raw_key = AttackProposal(
        changes={
            "phone_home_valid": ReferenceSelection(reference_id=choice.reference_id)
        }
    )
    raw_step = env3.step(raw_key)
    assert not raw_step.validity.is_valid


def test_abstract_proxy_key_mapping_is_audit_only():
    assert raw_feature_for_provenance_field("name_email_alignment") == (
        "name_email_similarity"
    )
    assert raw_feature_for_provenance_field("home_phone_configuration") == (
        "phone_home_valid"
    )
    assert raw_feature_for_provenance_field("mobile_phone_configuration") == (
        "phone_mobile_valid"
    )
    assert raw_feature_for_provenance_field("income") == "income"
    counts = provenance_audit_counts(
        {"status": "PASS", "fields": {"phone_home_valid": {"status": "PASS"}}}
    )
    assert counts["non_reference_backed"] == 0
    assert counts["reference_backed"] == 1

