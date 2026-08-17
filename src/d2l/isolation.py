"""Month-7 and path isolation for D2-L."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from d2.contract import CALIBRATION_MONTHS, SEALED_MONTHS
from d2l.errors import D2LDataError


def assert_not_month7_path(path: Path) -> None:
    text = str(path).lower()
    if "month7" in text or "month_7" in text:
        raise D2LDataError(f"Refusing a Month-7 path: {path}")


def assert_months_allowed(months: Iterable[int]) -> tuple[int, ...]:
    requested = tuple(int(m) for m in months)
    sealed = sorted(set(requested).intersection(SEALED_MONTHS))
    if sealed:
        raise D2LDataError(
            f"Month(s) {sealed} are sealed and cannot be opened, loaded, "
            "summarised, or scored by D2-L."
        )
    illegal = sorted(set(requested) - set(CALIBRATION_MONTHS))
    if illegal:
        raise D2LDataError(
            f"Month(s) {illegal} are not permitted for D2-L Month-6 development."
        )
    return requested
