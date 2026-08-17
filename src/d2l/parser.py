"""Strict JSON parser for the frozen D2-L output contract."""

from __future__ import annotations

import json
import re
from typing import Any

from d2l.contract import OUTPUT_REQUIRED_KEYS
from d2l.errors import D2LParseError

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    return stripped


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_fences(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise D2LParseError("LLM output did not contain a JSON object.")
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise D2LParseError(f"LLM output was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise D2LParseError("LLM JSON root is not an object.")
    return payload


def _as_int_score(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise D2LParseError("consistency_risk_score must be an integer 0-100.")
    if isinstance(value, int):
        score = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise D2LParseError(
                f"consistency_risk_score {value!r} is not an integer."
            )
        score = int(value)
    else:
        raise D2LParseError(
            f"consistency_risk_score has unsupported type {type(value)!r}."
        )
    if score < 0 or score > 100:
        raise D2LParseError(
            f"consistency_risk_score {score} is outside 0-100."
        )
    return score


def _as_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise D2LParseError("reason_codes must be a list of strings.")
    codes: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise D2LParseError("Each reason_code must be a string.")
        text = " ".join(item.split()).strip()
        if not text:
            continue
        codes.append(text[:120])
    return codes[:8]


def _as_summary(value: Any) -> str:
    if not isinstance(value, str):
        raise D2LParseError("summary must be a string.")
    text = " ".join(value.split()).strip()
    if not text:
        raise D2LParseError("summary is empty.")
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if len(parts) > 2:
        text = " ".join(parts[:2])
    return text[:400]


def parse_reviewer_output(text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    extra = sorted(set(payload) - set(OUTPUT_REQUIRED_KEYS))
    missing = [name for name in OUTPUT_REQUIRED_KEYS if name not in payload]
    if missing:
        raise D2LParseError(f"LLM JSON missing keys: {missing}")
    parsed = {
        "consistency_risk_score": _as_int_score(payload["consistency_risk_score"]),
        "reason_codes": _as_reason_codes(payload["reason_codes"]),
        "summary": _as_summary(payload["summary"]),
    }
    if extra:
        parsed["dropped_extra_keys"] = extra
    return parsed
