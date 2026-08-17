"""Parse frozen V2 rubric and categorical batch judgments."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from d2l.errors import D2LContractError, D2LParseError
from d2l.parser import extract_json_object
from d2l.v2_contract import DIMENSION_IDS, LABELS


def _norm_label(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("judgment", "label", "rating", "value"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, str):
        raise D2LParseError(f"Judgment is not a string: {value!r}")
    label = value.strip().upper()
    if label not in LABELS:
        raise D2LParseError(f"Judgment {value!r} is not one of {LABELS}.")
    return label


def parse_rubric(text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    rows = payload.get("dimensions")
    if isinstance(rows, Mapping):
        rebuilt = []
        for dim_id in DIMENSION_IDS:
            item = rows.get(dim_id)
            if not isinstance(item, Mapping):
                raise D2LParseError(f"Rubric dimension {dim_id} is missing.")
            rebuilt.append({"id": dim_id, **dict(item)})
        rows = rebuilt
    if not isinstance(rows, list) or len(rows) != 8:
        raise D2LParseError("Rubric must contain exactly 8 dimensions.")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise D2LParseError("Each rubric dimension must be an object.")
        dim_id = str(row.get("id") or "").strip()
        if dim_id not in DIMENSION_IDS:
            raise D2LParseError(f"Unknown rubric dimension id {dim_id!r}.")
        if dim_id in by_id:
            raise D2LParseError(f"Duplicate rubric dimension {dim_id!r}.")
        good = str(row.get("good") or row.get("GOOD") or "").strip()
        uncertain = str(row.get("uncertain") or row.get("UNCERTAIN") or "").strip()
        bad = str(row.get("bad") or row.get("BAD") or "").strip()
        if not good or not uncertain or not bad:
            raise D2LParseError(f"Dimension {dim_id} is missing GOOD/UNCERTAIN/BAD text.")
        forbidden = {"weight", "weights", "threshold", "total_score"}
        extra = {str(k).lower() for k in row} & forbidden
        if extra:
            raise D2LParseError(f"Dimension {dim_id} contains forbidden keys {sorted(extra)}.")
        by_id[dim_id] = {
            "id": dim_id,
            "title": str(row.get("title") or dim_id).strip(),
            "good": good,
            "uncertain": uncertain,
            "bad": bad,
        }
    missing = [dim_id for dim_id in DIMENSION_IDS if dim_id not in by_id]
    if missing:
        raise D2LParseError(f"Rubric missing dimensions: {missing}")
    return {
        "dimensions": [by_id[dim_id] for dim_id in DIMENSION_IDS],
        "weights": None,
        "total_score_defined_by_llm": False,
        "clear_review_defined_by_llm": False,
        "threshold_defined_by_llm": False,
    }


def assert_rubric_immutable(rubric: Mapping[str, Any]) -> None:
    ids = [row["id"] for row in rubric.get("dimensions") or []]
    if tuple(ids) != DIMENSION_IDS:
        raise D2LContractError("Frozen rubric dimension order drifted.")
    if rubric.get("weights") is not None:
        raise D2LContractError("Frozen rubric must not contain weights.")
    if rubric.get("total_score_defined_by_llm") or rubric.get("threshold_defined_by_llm"):
        raise D2LContractError("Frozen rubric must not define a total or threshold.")


def parse_batch_scores(
    text: str,
    expected_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    payload = extract_json_object(text)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise D2LParseError("Scoring JSON missing results array.")
    found: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise D2LParseError("Each scoring result must be an object.")
        app_id = str(row.get("id") or "").strip()
        if not app_id:
            raise D2LParseError("Scoring result missing id.")
        source = row.get("judgments") if isinstance(row.get("judgments"), Mapping) else row
        judgments: dict[str, str] = {}
        for dim_id in DIMENSION_IDS:
            if dim_id not in source:
                raise D2LParseError(f"{app_id} missing judgment for {dim_id}.")
            judgments[dim_id] = _norm_label(source[dim_id])
        found[app_id] = judgments
    missing = [app_id for app_id in expected_ids if app_id not in found]
    extra = sorted(set(found) - set(expected_ids))
    if missing:
        raise D2LParseError(f"Scoring output missing applications: {missing}")
    if extra:
        raise D2LParseError(f"Scoring output has unexpected ids: {extra}")
    return found
