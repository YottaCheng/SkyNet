"""Decision-time application view for D2-L.

The same constructor is used for legitimate and attacked applications.
Only frozen core application fields are retained.  Leakage keys are dropped
rather than forwarded.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from d2l.contract import APPLICATION_FIELDS, FORBIDDEN_INPUT_KEYS
from d2l.errors import D2LContractError


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _canonical_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        if number.is_integer():
            return int(number)
        return number
    if isinstance(value, str):
        return value
    if isinstance(value, (int,)):
        return int(value)
    raise D2LContractError(f"Unsupported application-field type: {type(value)!r}")


def application_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen decision-time field dict.

    Nested ``application`` maps are unwrapped.  Forbidden keys are ignored.
    Missing required fields fail closed.
    """
    if "application" in record and isinstance(record["application"], Mapping):
        source = record["application"]
    else:
        source = record
    missing = [name for name in APPLICATION_FIELDS if name not in source]
    if missing:
        raise D2LContractError(f"Application missing required fields: {missing}")
    view = {name: _canonical_value(source[name]) for name in APPLICATION_FIELDS}
    leaked = sorted(set(view).intersection(FORBIDDEN_INPUT_KEYS))
    if leaked:
        raise D2LContractError(f"Application view retained forbidden keys: {leaked}")
    return view


def serialize_application_view(view: Mapping[str, Any]) -> str:
    """Stable JSON for hashing, caching, and the LLM user message."""
    ordered = {name: view[name] for name in APPLICATION_FIELDS}
    return json.dumps(ordered, sort_keys=False, ensure_ascii=True, allow_nan=False)


def application_view_sha256(view: Mapping[str, Any]) -> str:
    payload = serialize_application_view(view)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_view_has_no_forbidden_fields(view: Mapping[str, Any], blob: str) -> None:
    """Fail closed if a forbidden key is present in the view or serialised blob."""
    for key in sorted(FORBIDDEN_INPUT_KEYS):
        if key in view:
            raise D2LContractError(f"Forbidden key {key!r} present in application view.")
        token = f'"{key}"'
        if token in blob:
            raise D2LContractError(
                f"Forbidden key {key!r} leaked into the serialised application view."
            )
