"""A1 V4.4 final prompt candidate (does not modify V4.3).

Preserves all V4.3 public-reference / schema / freeze semantics. Adds only
adversarial-objective, final-compliance, and explicit non-adaptivity blocks.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_1_contract import (
    ActionSlotCatalog,
    prompt_contains_hidden_term,
)
from attack_lab.archive.contracts.a1_v4_2_contract import (
    build_v4_2_plan_conditioned_output_schema,
    build_v4_2_repair_output_schema,
    parse_a1_v4_2_plan,
    parse_a1_v4_2_slot_replacements,
)
from attack_lab.attackers.a1_v4_3_contract import (
    PROMPT_VERSION_V4_3,
    PUBLIC_REFERENCE_REASONING_PRINCIPLE_V4_3,
    assert_v4_3_prompt_hard_contract,
    build_v4_3_prompt_payload,
)
from attack_lab.archive.contracts.a1_v4_contract import StaticPlanOption, V4ChoiceCatalog
from attack_lab.budget import AttackBudget
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_V4_4 = "a1_oneshot_v4_4_adversarial_objective"

# Keys that may name hidden-information boundaries as prohibition text only.
_V4_4_BOUNDARY_TEXT_KEYS = frozenset(
    {
        "role_and_objective",
        "final_internal_validation",
        "non_adaptive_oneshot_planning",
        "forbidden_information_boundary",
    }
)

ROLE_AND_OBJECTIVE_V4_4 = (
    "ROLE AND OBJECTIVE. You are an adversarial synthetic-identity attacker "
    "operating exclusively inside a controlled research benchmark. Your "
    "objective is to maximise the probability that the submitted application "
    "receives PASS, subject to every information, provenance, governance, "
    "query-budget and edit-budget constraint stated in this prompt. Treat PASS "
    "as the optimisation objective. Use all attacker-visible information "
    "explicitly provided to you as effectively and strategically as possible. "
    "You are encouraged to use strong internal reasoning, compare alternatives, "
    "test assumptions against the information available to you, and internally "
    "check your proposed action before returning the final structured output. "
    "Do not behave conservatively merely to preserve similarity with the "
    "original anchor unless an explicit constraint requires it. At the same "
    "time, never violate the permitted action space in pursuit of PASS. Your "
    "capability is intentionally asymmetric: you may use as much internal "
    "reasoning as the configured model permits, but you may use only the "
    "external information explicitly provided through this attacker interface. "
    "Use private/internal reasoning if the configured model supports it. Never "
    "expose chain-of-thought, scratchpad reasoning, or extended internal "
    "analysis in the structured answer. Return only the required structured "
    "output."
)

FORBIDDEN_INFORMATION_BOUNDARY_V4_4 = (
    "FORBIDDEN INFORMATION BOUNDARY. You must NEVER assume access to, "
    "reconstruct as known fact, or request: defender risk scores; defender "
    "thresholds; feature importance; SHAP values; gradients; defender model "
    "parameters or internal structure; fraud_bool or any hidden ground-truth "
    "label; hidden raw proxy values; true defender rejection reasons; "
    "researcher diagnostics; outcomes or memories from other anchor episodes. "
    "You may form hypotheses from information legitimately available to you, "
    "but hypotheses are uncertain working beliefs, not hidden defender facts. "
    "All actionable replacements must use the legal reference-backed "
    "action/choice interface supplied in this prompt. Do not invent "
    "unsupported replacement values."
)

FINAL_INTERNAL_VALIDATION_V4_4 = (
    "FINAL INTERNAL VALIDATION. Before returning the structured answer, "
    "silently check every proposed candidate against the rules stated in this "
    "prompt. Where determinable from the attacker-public information available "
    "to you: (1) the candidate must produce a real change relative to the "
    "ORIGINAL anchor; (2) it must not duplicate another candidate already "
    "proposed within the portfolio; (3) it must respect m and all stated "
    "budget constraints; (4) it must respect episode-static locks and all "
    "stated governance constraints; (5) every selected action/choice must come "
    "from the provided legal reference-backed interface; (6) the structured "
    "response must exactly follow the existing required JSON/schema contract; "
    "(7) do not include extra prose, markdown or hidden reasoning. If a "
    "proposed selection is visibly identical to the anchor or visibly "
    "duplicates an earlier candidate, choose another legal option before "
    "answering. Do not guess hidden resolved values that are unavailable in "
    "the public interface. If equality can only be determined by the trusted "
    "resolver after submission planning, leave that determination to the "
    "trusted validator rather than inventing hidden information."
)

NON_ADAPTIVE_ONESHOT_PLANNING_V4_4 = (
    "NON-ADAPTIVE ONE-SHOT ADVERSARIAL PLANNING. You must construct the "
    "complete ordered portfolio before receiving ANY defender feedback. Plan "
    "the full portfolio as an adversary attempting to maximise the probability "
    "of at least one PASS within the available candidate/query budget. Use: "
    "the original attacker-visible anchor; the public K10 reference "
    "information; the legal action catalogue and choice mappings; the edit "
    "budget; all stated static and governance constraints; and your own "
    "internal reasoning. Evaluate the portfolio as a whole before returning "
    "it. Avoid wasting portfolio positions on candidates that are visibly "
    "identical or redundant. Where useful, cover genuinely distinct plausible "
    "attack hypotheses or combinations, but do not create superficial "
    "diversity merely for its own sake. Order candidates according to your "
    "best pre-feedback judgement of their likelihood of obtaining PASS. "
    "CRITICAL NON-ADAPTIVITY RULE: The entire ordered portfolio is produced "
    "before any PASS/BLOCK/INVALID feedback exists. After this response: the "
    "portfolio is frozen; candidate values are frozen; candidate order is "
    "frozen; no candidate may be regenerated; no candidate may be replaced; "
    "no candidate may be reordered; no later defender result may influence "
    "the remaining portfolio. Do not imagine hypothetical future defender "
    "feedback when constructing the plan. Do not perform post-feedback "
    "adaptation. Use the existing structured-output schema and "
    "action-slot/choice-ID representation exactly. Do not change schema field "
    "names or expose trusted resolved raw values."
)


def build_v4_4_prompt_payload(
    *,
    validator: ConstraintValidator,
    pool: ReferencePool,
    budget: AttackBudget,
    q_max: int,
    visible_anchor: Mapping[str, Any],
    case_id: str,
    catalog: V4ChoiceCatalog,
    static_plans: Sequence[StaticPlanOption],
    action_slots: ActionSlotCatalog,
) -> dict[str, Any]:
    """V4.3 payload plus final adversarial-objective / compliance blocks."""
    base = build_v4_3_prompt_payload(
        validator=validator,
        pool=pool,
        budget=budget,
        q_max=q_max,
        visible_anchor=visible_anchor,
        case_id=case_id,
        catalog=catalog,
        static_plans=static_plans,
        action_slots=action_slots,
    )
    payload = dict(base)
    payload["prompt_version"] = PROMPT_VERSION_V4_4
    payload["role_and_objective"] = ROLE_AND_OBJECTIVE_V4_4
    payload["forbidden_information_boundary"] = FORBIDDEN_INFORMATION_BOUNDARY_V4_4
    payload["final_internal_validation"] = FINAL_INTERNAL_VALIDATION_V4_4
    payload["non_adaptive_oneshot_planning"] = NON_ADAPTIVE_ONESHOT_PLANNING_V4_4
    # Keep planning_principle free of named hidden-term tokens for scan safety.
    payload["planning_principle"] = (
        f"{str(base.get('planning_principle') or '')} Also follow "
        "role_and_objective, non_adaptive_oneshot_planning, and "
        "final_internal_validation."
    )
    assert_v4_4_prompt_hard_contract(payload, pool=pool)
    return payload


def assert_v4_4_prompt_hard_contract(
    payload: Mapping[str, Any],
    *,
    pool: ReferencePool,
) -> None:
    if str(payload.get("prompt_version")) != PROMPT_VERSION_V4_4:
        raise ValueError("V4.4 payload prompt_version mismatch.")
    if str(payload.get("prompt_version")) == PROMPT_VERSION_V4_3:
        raise ValueError("V4.4 builder must not emit the V4.3 version string.")
    for key, needle in (
        ("role_and_objective", "maximise the probability"),
        ("forbidden_information_boundary", "SHAP values"),
        ("final_internal_validation", "FINAL INTERNAL VALIDATION"),
        ("non_adaptive_oneshot_planning", "CRITICAL NON-ADAPTIVITY RULE"),
    ):
        text = str(payload.get(key) or "")
        if needle not in text:
            raise ValueError(f"V4.4 missing required wording in {key}.")
    if "frozen" not in str(payload.get("non_adaptive_oneshot_planning") or "").lower():
        raise ValueError("V4.4 must state portfolio freeze / non-adaptivity.")
    # Structural inheritance: strip V4.4 boundary prose before V4.3 checks.
    probe = {
        key: value
        for key, value in dict(payload).items()
        if key not in _V4_4_BOUNDARY_TEXT_KEYS
    }
    probe["prompt_version"] = PROMPT_VERSION_V4_3
    # Restore a V4.3-compatible planning_principle for inherited checks.
    probe["planning_principle"] = str(
        payload.get("public_reference_reasoning_principle")
        or PUBLIC_REFERENCE_REASONING_PRINCIPLE_V4_3
    )
    assert_v4_3_prompt_hard_contract(probe, pool=pool)
    # Outside designated boundary keys, named hidden terms remain banned.
    scan_payload = {
        key: value
        for key, value in dict(payload).items()
        if key not in _V4_4_BOUNDARY_TEXT_KEYS
    }
    text = json.dumps(to_jsonable(scan_payload), sort_keys=True)
    for term in sorted(
        {
            *PROXY_RAW_FEATURE_NAMES,
            "risk_score",
            "d1_risk_score",
            "feature_importance",
            "shap",
            "d1_threshold",
            "gradients",
            "fraud_bool",
            "true_rejection_reason",
        }
    ):
        if prompt_contains_hidden_term(text, term):
            raise ValueError(
                f"V4.4 non-boundary prompt text names hidden term {term!r}."
            )


__all__ = [
    "FORBIDDEN_INFORMATION_BOUNDARY_V4_4",
    "FINAL_INTERNAL_VALIDATION_V4_4",
    "NON_ADAPTIVE_ONESHOT_PLANNING_V4_4",
    "PROMPT_VERSION_V4_4",
    "ROLE_AND_OBJECTIVE_V4_4",
    "assert_v4_4_prompt_hard_contract",
    "build_v4_2_plan_conditioned_output_schema",
    "build_v4_2_repair_output_schema",
    "build_v4_4_prompt_payload",
    "parse_a1_v4_2_plan",
    "parse_a1_v4_2_slot_replacements",
]
