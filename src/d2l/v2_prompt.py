"""D2-L V2 prompts: rubric induction and independent categorical scoring."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from d2l.contract import APPLICATION_FIELDS, FIELD_DESCRIPTIONS, FORBIDDEN_PROMPT_SUBSTRINGS
from d2l.errors import D2LContractError
from d2l.v2_contract import DIMENSIONS, LABELS, PROMPT_VERSION

_DIMENSION_LINES = "\n".join(f"- `{dim_id}`: {title}" for dim_id, title in DIMENSIONS)

RUBRIC_SYSTEM = f"""You are constructing a generic application-consistency audit rubric for digital bank-account opening applications.

Your task is to infer what internally coherent applications look like from a sample of applications, then write a reusable rubric. You are not classifying fraud. You are not choosing a review threshold. You are not assigning numeric weights or a total score.

Write EXACTLY 8 dimensions, using these identifiers and themes:
{_DIMENSION_LINES}

For each dimension, define what counts as GOOD, UNCERTAIN, and BAD using only decision-time application fields. Base the definitions on generic internal coherence, such as life-stage, employment/housing, address history, banking/payment, contact, device/session/request, financial/application, and overall cross-field consistency.

Rules:
- Do not invent decoded meanings for anonymised categorical codes.
- Treat JSON null as not observed, not as a numeric value.
- Do not mention fraud, risk scores, model scores, red-team identity, or review queues.
- Do not assign weights.
- Do not define a total score, CLEAR, REVIEW, or a threshold.

Output strict JSON only:
{{"dimensions": [{{"id": "<one of the eight ids>", "title": "<short title>", "good": "<definition>", "uncertain": "<definition>", "bad": "<definition>"}}]}}
The array must contain exactly the eight identifiers above, each once, in that order.
"""

SCORE_SYSTEM_TEMPLATE = """You are applying a frozen application-consistency rubric to bank-account opening applications.

Judge each application independently. Do not rank applications against other applications in this batch. Do not change standards within the batch. Do not use other records as a reference distribution. Apply the exact frozen rubric to every record.

For each application and each of the eight frozen dimensions, output exactly one of: GOOD, UNCERTAIN, BAD.

You must not output a total score, CLEAR, REVIEW, fraud, or not-fraud. Optional evidence is allowed but must not replace the categorical judgment.

Frozen rubric:
{rubric_json}

Output strict JSON only:
{{"results": [{{"id": "<application id>", "judgments": {{"<dimension_id>": "GOOD|UNCERTAIN|BAD", ...}}}}]}}
Each result must include all eight dimension identifiers. Optional "evidence" maps are allowed and ignored by the scorer.
"""


def _assert_clean(text: str) -> None:
    lowered = text.lower()
    for token in FORBIDDEN_PROMPT_SUBSTRINGS:
        if token.lower() in lowered:
            raise D2LContractError(f"V2 prompt contains forbidden substring {token!r}.")


def packed_applications(
    items: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    for app_id, view in items:
        packed.append({"id": app_id, "fields": dict(view)})
    return packed


def rubric_messages(items: Sequence[tuple[str, Mapping[str, Any]]]) -> list[dict[str, str]]:
    payload = {
        "field_descriptions": {
            name: FIELD_DESCRIPTIONS[name] for name in APPLICATION_FIELDS
        },
        "applications": packed_applications(items),
        "instruction": (
            "Infer GOOD/UNCERTAIN/BAD definitions for the eight fixed dimensions "
            "from these applications. Do not score the applications in this call."
        ),
    }
    user = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    messages = [
        {"role": "system", "content": RUBRIC_SYSTEM.strip() + "\n"},
        {"role": "user", "content": user},
    ]
    _assert_clean(messages[0]["content"])
    return messages


def scoring_messages(
    rubric: Mapping[str, Any],
    items: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[dict[str, str]]:
    rubric_json = json.dumps(rubric, ensure_ascii=True, allow_nan=False, indent=2)
    system = SCORE_SYSTEM_TEMPLATE.format(rubric_json=rubric_json).strip() + "\n"
    payload = {
        "field_descriptions": {
            name: FIELD_DESCRIPTIONS[name] for name in APPLICATION_FIELDS
        },
        "applications": packed_applications(items),
        "allowed_labels": list(LABELS),
        "instruction": (
            "Score each application independently with the frozen rubric. "
            "Do not rank records. Do not output a total."
        ),
    }
    user = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    _assert_clean(RUBRIC_SYSTEM)
    _assert_clean(SCORE_SYSTEM_TEMPLATE.format(rubric_json=""))
    return messages


def prompt_version() -> str:
    return PROMPT_VERSION


_assert_clean(RUBRIC_SYSTEM)
_assert_clean(SCORE_SYSTEM_TEMPLATE.format(rubric_json=""))
