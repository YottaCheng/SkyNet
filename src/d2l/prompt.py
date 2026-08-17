"""Frozen D2-L prompt.  Do not edit after the Month-6 sanity freeze."""

from __future__ import annotations

from d2l.application_view import serialize_application_view
from d2l.contract import (
    APPLICATION_FIELDS,
    FIELD_DESCRIPTIONS,
    FORBIDDEN_PROMPT_SUBSTRINGS,
    GENERIC_REVIEW_DIMENSIONS,
    PROMPT_VERSION,
)
from d2l.errors import D2LContractError

SYSTEM_PROMPT = """You are a second-layer application-consistency reviewer for a digital bank-account opening workflow.

Your task is not to classify an application as fraudulent or legitimate. Given one application, assess whether the fields form a coherent, internally plausible profile, or whether cross-field inconsistencies warrant additional human review.

You may reason about the application as a whole, including generic dimensions such as:
- age / life-stage coherence
- employment / housing coherence
- address-history coherence
- banking / payment coherence
- contact configuration
- device / session / request context
- cross-field consistency across the whole profile

Missing values appear as JSON null. Official missing encodings have already been converted to null. Treat null as not observed, not as a numeric value.

Several fields are anonymised categorical codes. Do not invent decoded meanings for those codes. Judge only whether the observed combination of fields is internally coherent.

Use only the application fields and the field descriptions provided. Do not assume access to identity documents, external databases, or any field that is not listed.

Output strict JSON only, with no markdown, no commentary, and no extra keys:
{"consistency_risk_score": <integer 0-100>, "reason_codes": ["<short generic reason>", ...], "summary": "<maximum two short sentences>"}

Scoring:
- 0 means highly coherent / no reason for additional review
- 100 means highly inconsistent / strong reason for additional review
- Use an integer on this scale. Do not output a final CLEAR, REVIEW, fraud, or not-fraud decision.

reason_codes must be short, generic, and non-operational. Do not provide advice on how an application could be changed.
"""

_DICTIONARY_LINES = "\n".join(
    f"- {name}: {FIELD_DESCRIPTIONS[name]}" for name in APPLICATION_FIELDS
)

USER_PROMPT_TEMPLATE = """Field descriptions:
{dictionary}

Application:
{application_json}
"""


def system_prompt() -> str:
    return SYSTEM_PROMPT.strip() + "\n"


def user_prompt(application_json: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        dictionary=_DICTIONARY_LINES,
        application_json=application_json,
    )


def build_messages(view: dict[str, object]) -> list[dict[str, str]]:
    application_json = serialize_application_view(view)
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(application_json)},
    ]


def prompt_text() -> str:
    """Human-readable frozen prompt file contents."""
    return (
        f"PROMPT_VERSION={PROMPT_VERSION}\n\n"
        "===== SYSTEM =====\n"
        f"{system_prompt()}\n"
        "===== USER TEMPLATE =====\n"
        f"{user_prompt('<APPLICATION_JSON>')}\n"
    )


def assert_prompt_has_no_forbidden_substrings(
    text: str,
    *,
    require_dimensions: bool = False,
) -> None:
    lowered = text.lower()
    for token in FORBIDDEN_PROMPT_SUBSTRINGS:
        if token.lower() in lowered:
            raise D2LContractError(
                f"Frozen prompt contains forbidden substring {token!r}."
            )
    if require_dimensions:
        for dimension in GENERIC_REVIEW_DIMENSIONS:
            if dimension not in text:
                raise D2LContractError(
                    f"Frozen prompt is missing generic dimension {dimension!r}."
                )


assert_prompt_has_no_forbidden_substrings(system_prompt(), require_dimensions=True)
assert_prompt_has_no_forbidden_substrings(user_prompt('{"income": 0.5}'))
