"""Month-6 loading for D2-L. Month 7 is sealed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from baf_data.config import FROZEN_CONFIG
from d2.calibrate import load_month6_d1_scores, load_month6_d1_threshold
from d2.contract import CALIBRATION_MONTHS, SEALED_MONTHS
from d2.data import DEFAULT_RAW_PATH, apply_official_sentinels
from d2l.contract import APPLICATION_FIELDS
from d2l.errors import D2LDataError
from d2l.isolation import assert_months_allowed, assert_not_month7_path

_LOAD_COLUMNS: tuple[str, ...] = (
    "fraud_bool",
    "month",
    *APPLICATION_FIELDS,
)


def load_month6_core_applications(
    raw_path: Path = DEFAULT_RAW_PATH,
    *,
    fraud_bool: int | None,
    verify_hash: bool = True,
):
    """Load Month-6 rows with the full 27-field core application view.

    Implemented by reading through the shared D2 chunked loader after
    temporarily substituting the column list would risk changing D2-S.
    This function therefore repeats the sealed-month chunked read with
    D2-L's own column set.
    """
    from baf_data.integrity import verify_raw_source

    requested = assert_months_allowed(CALIBRATION_MONTHS)
    raw_path = Path(raw_path)
    assert_not_month7_path(raw_path)
    raw_sha256 = (
        verify_raw_source(raw_path, FROZEN_CONFIG.expected_sha256) if verify_hash else ""
    )
    wanted = set(requested)
    sealed = set(SEALED_MONTHS)
    pieces: list[pd.DataFrame] = []
    n_read = 0
    n_sealed_skipped = 0
    usecols = [c for c in _LOAD_COLUMNS if c in FROZEN_CONFIG.raw_column_names]
    missing = [c for c in APPLICATION_FIELDS if c not in usecols]
    if missing:
        raise D2LDataError(f"Core application fields missing from schema: {missing}")

    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=100_000):
        n_chunk = len(chunk)
        source_row_id = pd.RangeIndex(n_read, n_read + n_chunk)
        n_read += n_chunk
        month_values = chunk["month"].astype("int64")
        n_sealed_skipped += int(month_values.isin(sealed).sum())
        keep = month_values.isin(wanted)
        if fraud_bool is not None:
            keep = keep & (chunk["fraud_bool"].astype("int64") == int(fraud_bool))
        kept = chunk.loc[keep].copy()
        if not kept.empty:
            kept.insert(0, "source_row_id", source_row_id[keep.to_numpy()].astype("int64"))
            pieces.append(kept)

    if pieces:
        frame = pd.concat(pieces, ignore_index=True)
    else:
        frame = pd.DataFrame(columns=["source_row_id", *usecols])
    if len(frame) and frame["month"].isin(list(SEALED_MONTHS)).any():
        raise D2LDataError("Sealed-month rows were retained; aborting.")
    normalised = apply_official_sentinels(frame, FROZEN_CONFIG)
    observed = tuple(sorted(int(m) for m in normalised["month"].unique())) if len(normalised) else tuple()
    if set(observed) - set(requested):
        raise D2LDataError(f"Loaded months {observed} exceed {requested}.")
    from d2.data import LoadedD2Frame

    return LoadedD2Frame(
        frame=normalised,
        months=observed,
        raw_sha256=raw_sha256,
        raw_path=raw_path,
        n_rows_read=n_read,
        n_rows_retained=len(normalised),
        n_sealed_rows_skipped=n_sealed_skipped,
        fraud_bool_filter=fraud_bool,
        month7_opened=False,
    )


def month6_legitimate_d1_pass_core(
    raw_path: Path = DEFAULT_RAW_PATH,
    *,
    verify_hash: bool = True,
) -> pd.DataFrame:
    """Month-6 fraud_bool==0 applications that PASS frozen D1, with 27 core fields."""
    loaded = load_month6_core_applications(
        raw_path, fraud_bool=0, verify_hash=verify_hash
    )
    if loaded.month7_opened or set(loaded.months) - set(CALIBRATION_MONTHS):
        raise D2LDataError("Month-6 load violated the sealed-month boundary.")
    scores = load_month6_d1_scores()
    threshold = load_month6_d1_threshold()
    merged = loaded.frame.merge(
        scores,
        left_on="source_row_id",
        right_on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if int((merged["y_true"] != 0).sum()):
        raise D2LDataError("D1 score join produced y_true != 0 on a legit Month-6 frame.")
    passed = merged.loc[merged["y_score"] < threshold].copy()
    missing = [name for name in APPLICATION_FIELDS if name not in passed.columns]
    if missing:
        raise D2LDataError(f"D1-PASS frame missing application fields: {missing}")
    return passed
