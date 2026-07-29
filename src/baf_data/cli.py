"""Command-line entry point for the deterministic BAF data layer.

Usage:
    python src/prepare_baf_data.py \
        --raw /Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv \
        --output /Volumes/Study/ucl_dissertation_data/splits/baf_base

Verifies the raw hash, validates the schema, applies the frozen sentinel
rules in memory, builds the frozen temporal split and writes the split
and feature manifests plus a structured run log. Trains nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from baf_data.errors import DataLayerError
from baf_data.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic BAF data layer: verify, validate, split and "
            "write manifests. Performs no imputation, encoding or training."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Path to the immutable raw Base.csv (opened read-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for manifests and the run log (must be outside raw/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    try:
        result = run_pipeline(args.raw, args.output)
    except DataLayerError as exc:
        logger.error("Data layer refused to run: %s", exc)
        return 1
    for name, entry in result.split_manifest["splits"].items():
        logger.info(
            "%s: %s rows, %s fraud (months %s)",
            name, f"{entry['row_count']:,}", f"{entry['fraud_count']:,}", entry["months"],
        )
    logger.info("Done. Manifests in %s", {k: str(v) for k, v in result.written_paths.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
