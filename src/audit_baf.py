"""Read-only audit of the Bank Account Fraud (BAF) Base dataset.

Reads one CSV file, computes descriptive statistics, and writes five
artefacts (two CSV tables, one JSON summary, one PNG plot, one Markdown
report) to an output directory.

The script never modifies the input file and performs no cleaning,
imputation, splitting or modelling. Sentinel-looking values such as -1
are counted but deliberately not interpreted or replaced.

Usage:
    python audit_baf.py --input /path/to/Base.csv --output /path/to/baf_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: we only save files, never open windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("audit_baf")

LABEL_COLUMN = "fraud_bool"
MONTH_COLUMN = "month"


# --------------------------------------------------------------------------
# CLI and input handling
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the BAF Base dataset (no cleaning, no splits, no models)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw Base.csv file (opened read-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where audit artefacts are written (created if missing).",
    )
    return parser.parse_args(argv)


def sha256_of_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute the SHA-256 of a file in chunks so large files fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV with pandas' default type inference (no transformation)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Check that the external drive is mounted and the path is correct."
        )
    logger.info("Reading %s (this may take a moment for ~200 MB) ...", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns.", df.shape[0], df.shape[1])

    for required in (LABEL_COLUMN, MONTH_COLUMN):
        if required not in df.columns:
            raise ValueError(
                f"Expected column '{required}' is missing. "
                "This file may not be the BAF Base dataset."
            )
    return df


# --------------------------------------------------------------------------
# Statistics (all read-only; the dataframe is never altered)
# --------------------------------------------------------------------------

def summarise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missingness, sentinel counts and basic stats.

    Only facts are recorded. A value of -1 is counted because it is a common
    sentinel code, but its meaning per column is NOT assumed here.
    """
    rows: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        row: dict[str, Any] = {
            "column": name,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "count_minus_1": None,
            "count_other_negative": None,
            "count_pos_inf": None,
            "count_neg_inf": None,
            "min": None,
            "max": None,
            "median": None,
        }

        if is_numeric:
            values = series.to_numpy()
            row["count_minus_1"] = int((values == -1).sum())
            # Negative values other than -1: potentially unusual, reported
            # as a fact without interpreting what they encode.
            row["count_other_negative"] = int(((values < 0) & (values != -1)).sum())
            row["count_pos_inf"] = int(np.isposinf(values).sum())
            row["count_neg_inf"] = int(np.isneginf(values).sum())
            row["min"] = float(series.min())
            row["max"] = float(series.max())
            row["median"] = float(series.median())

        rows.append(row)
    return pd.DataFrame(rows)


def summarise_months(df: pd.DataFrame) -> pd.DataFrame:
    """Sample count, fraud count and fraud rate for every month value."""
    grouped = (
        df.groupby(MONTH_COLUMN, dropna=False)[LABEL_COLUMN]
        .agg(sample_count="size", fraud_count="sum")
        .reset_index()
        .sort_values(MONTH_COLUMN)
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["sample_count"]
    return grouped


def build_summary(
    df: pd.DataFrame,
    input_path: Path,
    file_sha256: str,
    column_summary: pd.DataFrame,
    month_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the machine-readable audit summary (for audit_summary.json)."""
    label_counts = df[LABEL_COLUMN].value_counts(dropna=False)
    label_props = df[LABEL_COLUMN].value_counts(normalize=True, dropna=False)

    return {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": {
            "filename": input_path.name,
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": file_sha256,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "fraud_bool": {
            "values": [int(v) for v in sorted(label_counts.index)],
            "counts": {str(int(k)): int(v) for k, v in label_counts.items()},
            "proportions": {str(int(k)): float(v) for k, v in label_props.items()},
        },
        "month": {
            "values": [int(v) for v in sorted(df[MONTH_COLUMN].unique())],
            "per_month": month_summary.to_dict(orient="records"),
        },
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
        "pos_inf_total": int(column_summary["count_pos_inf"].fillna(0).sum()),
        "neg_inf_total": int(column_summary["count_neg_inf"].fillna(0).sum()),
        "minus_1_total": int(column_summary["count_minus_1"].fillna(0).sum()),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def plot_fraud_rate_by_month(month_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        month_summary[MONTH_COLUMN],
        month_summary["fraud_rate"] * 100,
        marker="o",
    )
    ax.set_xlabel("month (as recorded in the dataset)")
    ax.set_ylabel("fraud rate (%)")
    ax.set_title("BAF Base: fraud rate by month")
    ax.set_xticks(month_summary[MONTH_COLUMN])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown_report(
    summary: dict[str, Any],
    column_summary: pd.DataFrame,
    month_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write DATASET_AUDIT.md with facts, concerns and open decisions kept apart."""
    src = summary["source_file"]
    shape = summary["shape"]
    fraud = summary["fraud_bool"]

    minus1_cols = column_summary[column_summary["count_minus_1"].fillna(0) > 0]
    other_neg_cols = column_summary[column_summary["count_other_negative"].fillna(0) > 0]
    missing_cols = column_summary[column_summary["missing_count"] > 0]

    fraud_rate_overall = fraud["proportions"].get("1", 0.0)
    min_month_rate = month_summary["fraud_rate"].min()
    max_month_rate = month_summary["fraud_rate"].max()

    lines: list[str] = []
    add = lines.append

    add("# BAF Base Dataset Audit")
    add("")
    add(f"Generated (UTC): {summary['audit_generated_utc']}  ")
    add(f"Script: `04_implementation/src/audit_baf.py` (read-only audit; no cleaning, no splits, no models)  ")
    add(
        f"Environment: Python {summary['environment']['python']}, "
        f"pandas {summary['environment']['pandas']}, numpy {summary['environment']['numpy']}"
    )
    add("")

    add("## 1. Observed facts")
    add("")
    add(f"- Source file: `{src['filename']}` at `{src['path']}`")
    add(f"- File size: {src['size_bytes']:,} bytes")
    add(f"- SHA-256: `{src['sha256']}`")
    add(f"- Shape: {shape['rows']:,} rows x {shape['columns']} columns")
    add(f"- In-memory size (pandas, deep): {summary['memory_usage_bytes']:,} bytes")
    add(f"- Duplicate rows (all columns identical): {summary['duplicate_row_count']:,}")
    add(f"- Missing values (pandas NaN/NA) across all columns: {summary['missing_values_total']:,}")
    add(
        f"- Infinity values: +inf = {summary['pos_inf_total']:,}, "
        f"-inf = {summary['neg_inf_total']:,}"
    )
    add(f"- Total occurrences of the value -1 across numeric columns: {summary['minus_1_total']:,}")
    add("")
    add(f"- `{LABEL_COLUMN}` values: {fraud['values']}")
    for value in fraud["values"]:
        key = str(value)
        add(
            f"  - {LABEL_COLUMN} = {value}: {fraud['counts'][key]:,} rows "
            f"({fraud['proportions'][key]:.4%})"
        )
    add("")
    add(f"- `{MONTH_COLUMN}` values present: {summary['month']['values']}")
    add(
        f"- Fraud rate per month ranges from {min_month_rate:.4%} to {max_month_rate:.4%} "
        "(full table in `month_summary.csv`, plot in `fraud_rate_by_month.png`)."
    )
    add("")
    add("- Columns containing the value -1 (counted, not interpreted):")
    if minus1_cols.empty:
        add("  - none")
    else:
        for _, row in minus1_cols.iterrows():
            add(f"  - `{row['column']}`: {int(row['count_minus_1']):,} occurrences")
    add("")
    add("- Columns containing negative values other than -1:")
    if other_neg_cols.empty:
        add("  - none")
    else:
        for _, row in other_neg_cols.iterrows():
            add(f"  - `{row['column']}`: {int(row['count_other_negative']):,} occurrences")
    add("")
    add("- Columns containing missing values (pandas NaN/NA):")
    if missing_cols.empty:
        add("  - none")
    else:
        for _, row in missing_cols.iterrows():
            add(f"  - `{row['column']}`: {int(row['missing_count']):,} missing")
    add("")
    add("Per-column dtype, unique counts, min/max/median and sentinel counts are in `column_summary.csv`.")
    add("")

    add("## 2. Possible concerns")
    add("")
    add(
        f"- Severe class imbalance: overall fraud rate is {fraud_rate_overall:.4%}. "
        "Accuracy alone would be uninformative; evaluation metrics need to account for this."
    )
    if not minus1_cols.empty:
        add(
            f"- {len(minus1_cols)} column(s) contain -1. In many datasets -1 encodes "
            "'missing' or 'not applicable', but the meaning here has not been verified "
            "against the BAF datasheet yet. If -1 is a sentinel, naive numeric statistics "
            "(min/median) for those columns are distorted."
        )
    if not other_neg_cols.empty:
        add(
            f"- {len(other_neg_cols)} column(s) contain negative values other than -1. "
            "Whether these are valid measurements or additional sentinel codes is unverified."
        )
    if summary["duplicate_row_count"] > 0:
        add(
            f"- {summary['duplicate_row_count']:,} fully duplicated rows exist. Whether they are "
            "legitimate repeated observations or artefacts is unknown."
        )
    if summary["missing_values_total"] == 0:
        add(
            "- No pandas-level missing values were found. Missingness may instead be encoded "
            "as sentinel values (for example -1), so 'no NaN' should not be read as 'complete data'."
        )
    add(
        "- Fraud rate varies across months "
        f"({min_month_rate:.4%} to {max_month_rate:.4%}); any random split that ignores "
        "time could leak temporal structure between training and evaluation."
    )
    add(
        "- Column meanings are not inferred from names in this audit; every semantic claim "
        "must be checked against the BAF datasheet/paper before use in the dissertation."
    )
    add("")

    add("## 3. Decisions not yet made")
    add("")
    add("- How (and whether) to treat -1 and other candidate sentinel values per column.")
    if summary["duplicate_row_count"] > 0:
        add("- How to handle the duplicated rows, if at all.")
    add("- Train / validation / test split strategy (temporal by month vs random), and which months form the untouched final evaluation set.")
    add("- Encoding strategy for the string-typed (categorical-looking) columns.")
    add("- Evaluation metrics and operating thresholds appropriate for the observed class imbalance.")
    add("- Whether any BAF variant datasets (beyond Base) will be used.")
    add("")
    add("No cleaning, transformation, splitting or modelling has been performed.")
    add("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    input_path: Path = args.input.expanduser().resolve()
    output_dir: Path = args.output.expanduser().resolve()

    # Refuse to write into the directory holding the raw file, as a guard
    # against accidentally polluting (or overwriting inside) raw data.
    if input_path.parent in (output_dir, *output_dir.parents) or output_dir == input_path.parent:
        raise ValueError(
            f"Output directory {output_dir} would write next to or above the raw file. "
            "Choose a separate documentation directory."
        )

    df = load_dataset(input_path)

    logger.info("Computing SHA-256 of the source file ...")
    file_sha256 = sha256_of_file(input_path)
    logger.info("SHA-256: %s", file_sha256)

    logger.info("Building column summary ...")
    column_summary = summarise_columns(df)
    logger.info("Building month summary ...")
    month_summary = summarise_months(df)
    summary = build_summary(df, input_path, file_sha256, column_summary, month_summary)

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "column_summary.csv": lambda p: column_summary.to_csv(p, index=False),
        "month_summary.csv": lambda p: month_summary.to_csv(p, index=False),
        "audit_summary.json": lambda p: p.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        ),
        "fraud_rate_by_month.png": lambda p: plot_fraud_rate_by_month(month_summary, p),
        "DATASET_AUDIT.md": lambda p: write_markdown_report(
            summary, column_summary, month_summary, p
        ),
    }
    for filename, writer in outputs.items():
        path = output_dir / filename
        writer(path)
        logger.info("Wrote %s", path)

    logger.info("Audit complete. %d artefacts written to %s", len(outputs), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
