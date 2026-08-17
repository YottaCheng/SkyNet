"""A3 V2.4 final prompt candidate (does not modify V2.3).

Preserves all V2.3 public-reference / reflection / cardinality-repair
semantics. Adds only adversarial-objective, final-compliance, and explicit
within-anchor adaptation blocks.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_1_contract import (
    ActionSlotCatalog,
    GENERIC_UNAVAILABLE_NOTICE,
    prompt_contains_hidden_term,
)
from attack_lab.archive.contracts.a1_v4_contract import V4ChoiceCatalog
from attack_lab.archive.contracts.a3_v2_2_contract import (
    CARDINALITY_REPAIR_INSTRUCTION,
    SELECTIONS_VS_HYPOTHESIS_NOTE,
    filter_mechanically_valid_proposed_pairs,
    parse_a3_v2_2_repair_selections,
    parse_a3_v2_2_strategic_response,
    build_a3_v2_2_cardinality_repair_schema,
    build_a3_v2_2_output_schema,
    build_a3_v2_2_repair_schema,
)
from attack_lab.attackers.a3_v2_3_contract import (
    MAX_HYPOTHESIS_CHARS_V2_3,
    PROMPT_VERSION_A3_V2_3,
    PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3,
    REFLECTION_MODE_DEFINITIONS_V2_3,
    REFLECTION_PURPOSE_V2_3,
    STATIC_RULE_DISCLOSURE_V2_3,
    assert_a3_v2_3_prompt_hard_contract,
    build_a3_v2_3_episode_action_slots,
    build_a3_v2_3_prompt_payload,
)
from attack_lab.budget import AttackBudget
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import to_jsonable

PROMPT_VERSION_A3_V2_4 = "a3_episodic_reflective_v2_4_adversarial_objective"
MAX_HYPOTHESIS_CHARS_V2_4 = MAX_HYPOTHESIS_CHARS_V2_3
REFLECTION_MODE_DEFINITIONS_V2_4 = dict(REFLECTION_MODE_DEFINITIONS_V2_3)
REFLECTION_PURPOSE_V2_4 = REFLECTION_PURPOSE_V2_3
STATIC_RULE_DISCLOSURE_V2_4 = STATIC_RULE_DISCLOSURE_V2_3

_V2_4_BOUNDARY_TEXT_KEYS = frozenset(
    {
        "role_and_objective",
        "final_internal_validation",
        "adaptive_episodic_adversarial_reasoning",
        "forbidden_information_boundary",
        "adaptation_note_limit",
    }
)

ROLE_AND_OBJECTIVE_V2_4 = (
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

FORBIDDEN_INFORMATION_BOUNDARY_V2_4 = (
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

FINAL_INTERNAL_VALIDATION_V2_4 = (
    "FINAL INTERNAL VALIDATION. Before returning the structured answer, "
    "silently check the proposed candidate against the rules stated in this "
    "prompt. Where determinable from the attacker-public information available "
    "to you: (1) the candidate must produce a real change relative to the "
    "ORIGINAL anchor; (2) it must not duplicate another candidate already "
    "proposed within this anchor episode; (3) it must respect m and all stated "
    "budget constraints; (4) it must respect episode-static locks and all "
    "stated governance constraints; (5) every selected action/choice must come "
    "from the provided legal reference-backed interface; (6) the structured "
    "response must exactly follow the existing required JSON/schema contract; "
    "(7) do not include extra prose, markdown or hidden reasoning. If a "
    "proposed selection is visibly identical to the anchor or visibly "
    "duplicates an earlier candidate, choose another legal option before "
    "answering. Do not guess hidden resolved values that are unavailable in "
    "the public interface."
)

ADAPTIVE_EPISODIC_REASONING_V2_4 = (
    "ADAPTIVE EPISODIC ADVERSARIAL REASONING. You generate ONE strategic "
    "candidate for the current external query. Your objective at every query "
    "is to choose the legal candidate that, based on the information currently "
    "available to you, you judge most likely to improve the episode's "
    "probability of eventually obtaining PASS. For the current query you may "
    "use ONLY: the original attacker-visible anchor; the public K10 reference "
    "information; the legal action catalogue and choice mappings; remaining Q "
    "and m information; your own previous submitted candidates from THIS "
    "anchor; the bounded structured episode memory from THIS anchor; public "
    "PASS/BLOCK/INVALID outcomes from previous submissions on THIS anchor; "
    "and governance-visible LOCAL_REJECT information that the existing "
    "public/local interface legitimately exposes. You must not use information "
    "from another anchor episode. INITIAL QUERY: Construct the strongest "
    "candidate you can from the currently available public information. Do "
    "not reserve good ideas merely because later queries may exist. AFTER "
    "BLOCK OR INVALID: Treat the observed public outcome as genuine new "
    "evidence, but only as sparse label-level evidence. Critically update the "
    "current working hypothesis. Explicitly choose, through the EXISTING "
    "structured decision mechanism, whether the previous strategy should be "
    "RETAINED, REVISED, or ABANDONED. Then generate the next candidate that "
    "you judge most likely to improve the probability of PASS. Use prior "
    "failures actively: do not mechanically repeat a failed candidate; "
    "reconsider which parts of the previous hypothesis remain useful; explore "
    "a meaningfully different legal candidate when the evidence supports "
    "doing so; preserve useful components when revision is more appropriate "
    "than complete abandonment. However: A BLOCK tells you only that the "
    "submitted candidate did not receive PASS. It does NOT reveal which field "
    "caused the decision. It does NOT reveal a score, threshold, gradient, "
    "feature importance or causal rejection reason. Never convert sparse "
    "feedback into a claimed hidden fact. Your adaptation_note / hypothesis "
    f"field is a compact operational memory tag, NOT chain-of-thought. "
    f"adaptation_note must be <= {MAX_HYPOTHESIS_CHARS_V2_4} characters. "
    "Keep it concise and factual enough to support the next query. Do not "
    "include extended explanation, scratchpad reasoning, markdown, scores or "
    "hidden-defender speculation. Use the existing structured-output schema "
    "and action-slot/choice-ID representation exactly. Do not change schema "
    "field names or expose trusted resolved raw values."
)

ADAPTATION_NOTE_LIMIT_V2_4 = (
    f"adaptation_note must be <= {MAX_HYPOTHESIS_CHARS_V2_4} characters."
)


def build_a3_v2_4_episode_action_slots(
    catalog: V4ChoiceCatalog,
    *,
    validator: Any,
) -> ActionSlotCatalog:
    return build_a3_v2_3_episode_action_slots(catalog, validator=validator)


def build_a3_v2_4_prompt_payload(
    *,
    case_id: str,
    visible_anchor: Mapping[str, Any],
    current_application: Mapping[str, Any],
    budget: AttackBudget,
    q_remaining: int,
    query_index: int,
    static_edit_cost: int,
    residual_m: int,
    locked_static_values: Mapping[str, Any],
    slots: ActionSlotCatalog,
    slot_entries: Sequence[Mapping[str, Any]],
    episodic_memory: Sequence[Mapping[str, Any]],
    pool: ReferencePool,
    catalog: V4ChoiceCatalog,
    episode_slot_map: Sequence[Mapping[str, Any]] | None = None,
    local_repair: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = build_a3_v2_3_prompt_payload(
        case_id=case_id,
        visible_anchor=visible_anchor,
        current_application=current_application,
        budget=budget,
        q_remaining=q_remaining,
        query_index=query_index,
        static_edit_cost=static_edit_cost,
        residual_m=residual_m,
        locked_static_values=locked_static_values,
        slots=slots,
        slot_entries=slot_entries,
        episodic_memory=episodic_memory,
        pool=pool,
        catalog=catalog,
        episode_slot_map=episode_slot_map,
        local_repair=local_repair,
    )
    payload = dict(base)
    payload["prompt_version"] = PROMPT_VERSION_A3_V2_4
    payload["role_and_objective"] = ROLE_AND_OBJECTIVE_V2_4
    payload["forbidden_information_boundary"] = FORBIDDEN_INFORMATION_BOUNDARY_V2_4
    payload["final_internal_validation"] = FINAL_INTERNAL_VALIDATION_V2_4
    payload["adaptive_episodic_adversarial_reasoning"] = (
        ADAPTIVE_EPISODIC_REASONING_V2_4
    )
    payload["adaptation_note_limit"] = ADAPTATION_NOTE_LIMIT_V2_4
    # Keep V2.3 task/attack_objective bodies free of named hidden-term tokens;
    # the new blocks live in dedicated keys and the system message renderer.
    payload["task"] = (
        f"{str(base.get('task') or '')} Also follow role_and_objective, "
        "adaptive_episodic_adversarial_reasoning, final_internal_validation, "
        "and adaptation_note_limit."
    )
    assert_a3_v2_4_prompt_hard_contract(payload, pool=pool)
    return payload


def assert_a3_v2_4_prompt_hard_contract(
    payload: Mapping[str, Any],
    *,
    pool: ReferencePool,
) -> None:
    if str(payload.get("prompt_version")) != PROMPT_VERSION_A3_V2_4:
        raise ValueError("A3 V2.4 prompt_version mismatch.")
    if str(payload.get("prompt_version")) == PROMPT_VERSION_A3_V2_3:
        raise ValueError("A3 V2.4 builder must not emit the V2.3 version string.")
    for key, needle in (
        ("role_and_objective", "maximise the probability"),
        ("forbidden_information_boundary", "SHAP values"),
        ("final_internal_validation", "FINAL INTERNAL VALIDATION"),
        ("adaptive_episodic_adversarial_reasoning", "RETAINED"),
        (
            "adaptation_note_limit",
            f"<= {MAX_HYPOTHESIS_CHARS_V2_4} characters",
        ),
    ):
        if needle not in str(payload.get(key) or ""):
            raise ValueError(f"A3 V2.4 missing required wording in {key}.")
    # Structural inheritance without boundary prose that names banned terms.
    probe = {
        key: value
        for key, value in dict(payload).items()
        if key not in _V2_4_BOUNDARY_TEXT_KEYS
    }
    probe["prompt_version"] = PROMPT_VERSION_A3_V2_3
    assert_a3_v2_3_prompt_hard_contract(probe, pool=pool)
    scan_payload = {
        key: value
        for key, value in dict(payload).items()
        if key not in _V2_4_BOUNDARY_TEXT_KEYS
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
                f"A3 V2.4 non-boundary prompt text names hidden term {term!r}."
            )


def render_a3_v2_4_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    system = (
        f"{ROLE_AND_OBJECTIVE_V2_4} {FORBIDDEN_INFORMATION_BOUNDARY_V2_4} "
        f"{ADAPTIVE_EPISODIC_REASONING_V2_4} {FINAL_INTERNAL_VALIDATION_V2_4} "
        f"{SELECTIONS_VS_HYPOTHESIS_NOTE} "
        f"{PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3} "
        "Emit ONLY the JSON object required by the output schema. "
        "Never emit raw feature values, action_key strings as outputs, "
        "reference_id, or a changes object. "
        "At query 1, reflection_update.mode must be INITIALIZE. "
        "After a submitted BLOCK or INVALID, first produce RETAIN, REVISE, or "
        "ABANDON based only on public outcomes, then choose the next selections "
        "in the same response. "
        "You do not know which selected action, field, value or interaction "
        "caused BLOCK. "
        f"{GENERIC_UNAVAILABLE_NOTICE}"
    )
    repair = payload.get("local_selection_repair")
    if repair is not None:
        residual = (payload.get("budget") or {}).get("residual_m")
        system = (
            "Local compliance repair only. "
            f"{CARDINALITY_REPAIR_INSTRUCTION} "
            f"Current residual_m / maximum selections this query: {residual}. "
            "Return ONLY {\"selections\": {...}}. "
            f"{GENERIC_UNAVAILABLE_NOTICE}"
        )
    user = json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


parse_a3_v2_4_strategic_response = parse_a3_v2_2_strategic_response
parse_a3_v2_4_repair_selections = parse_a3_v2_2_repair_selections
build_a3_v2_4_output_schema = build_a3_v2_2_output_schema
build_a3_v2_4_repair_schema = build_a3_v2_2_repair_schema
build_a3_v2_4_cardinality_repair_schema = build_a3_v2_2_cardinality_repair_schema


__all__ = [
    "ADAPTATION_NOTE_LIMIT_V2_4",
    "ADAPTIVE_EPISODIC_REASONING_V2_4",
    "CARDINALITY_REPAIR_INSTRUCTION",
    "FINAL_INTERNAL_VALIDATION_V2_4",
    "FORBIDDEN_INFORMATION_BOUNDARY_V2_4",
    "MAX_HYPOTHESIS_CHARS_V2_4",
    "PROMPT_VERSION_A3_V2_4",
    "REFLECTION_MODE_DEFINITIONS_V2_4",
    "REFLECTION_PURPOSE_V2_4",
    "ROLE_AND_OBJECTIVE_V2_4",
    "SELECTIONS_VS_HYPOTHESIS_NOTE",
    "STATIC_RULE_DISCLOSURE_V2_4",
    "assert_a3_v2_4_prompt_hard_contract",
    "build_a3_v2_4_cardinality_repair_schema",
    "build_a3_v2_4_episode_action_slots",
    "build_a3_v2_4_output_schema",
    "build_a3_v2_4_prompt_payload",
    "build_a3_v2_4_repair_schema",
    "filter_mechanically_valid_proposed_pairs",
    "parse_a3_v2_4_repair_selections",
    "parse_a3_v2_4_strategic_response",
    "render_a3_v2_4_messages",
]
