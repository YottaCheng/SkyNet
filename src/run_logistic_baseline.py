"""CLI entry point for the Logistic Regression development baseline.

Fits two prespecified class_weight variants on months 0–5 and evaluates
them on month 6 only. Month 7 is loaded by the data layer for integrity
but is never scored or evaluated by this script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from baf_data.config import FROZEN_CONFIG
from baf_models.artifacts import (
    RUN_ID,
    append_run_log,
    plot_confusion_matrices,
    plot_development_curves,
    run_directory,
    save_comparison_table,
    save_variant_artifacts,
    write_environment,
)
from baf_models.evaluation import DEFAULT_MAX_FPR
from baf_models.training import (
    NonConvergenceError,
    build_variant_configs,
    fit_and_score_variant,
    load_base_config,
    load_train_dev_bundle,
)

logger = logging.getLogger(__name__)

DEFAULT_RAW = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")
DEFAULT_YAML = (
    Path(__file__).resolve().parents[1] / "config" / "logistic_baseline.yaml"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit LR baseline variants on months 0–5 and evaluate on month 6 only. "
            "Does not score month 7."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--config", type=Path, default=DEFAULT_YAML)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root under which the deterministic run directory is created.",
    )
    parser.add_argument("--run-id", type=str, default=RUN_ID)
    parser.add_argument("--max-fpr", type=float, default=DEFAULT_MAX_FPR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    output_dir = run_directory(args.output_root, args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading train/dev only from %s", args.raw)
    bundle = load_train_dev_bundle(args.raw, FROZEN_CONFIG)

    base = load_base_config(args.config)
    variants = build_variant_configs(base)
    logger.info(
        "Prespecified variants: %s (shared C=%s, solver=%s, max_iter=%s, seed=%s).",
        list(variants),
        base.C,
        base.solver,
        base.max_iter,
        base.random_state,
    )

    results = {}
    try:
        for name, config in variants.items():
            results[name] = fit_and_score_variant(
                bundle,
                name,
                config,
                FROZEN_CONFIG,
                max_fpr=args.max_fpr,
                require_convergence=True,
            )
    except NonConvergenceError as exc:
        logger.error("%s", exc)
        append_run_log(
            output_dir / "run_log.jsonl",
            {
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "outcome": "non_convergence",
                "error": str(exc),
                "run_id": args.run_id,
            },
        )
        return 2

    write_environment(output_dir)
    for name, result in results.items():
        save_variant_artifacts(
            result,
            output_dir / name,
            data_config=FROZEN_CONFIG,
            raw_sha256=bundle.raw_sha256,
        )
    comparison_path = save_comparison_table(results, output_dir)
    figures_dir = output_dir / "figures"
    curve_paths = plot_development_curves(results, figures_dir)
    confusion_paths = plot_confusion_matrices(results, figures_dir)

    append_run_log(
        output_dir / "run_log.jsonl",
        {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "outcome": "success",
            "run_id": args.run_id,
            "raw_sha256": bundle.raw_sha256,
            "train_rows": int(len(bundle.train.X)),
            "development_rows": int(len(bundle.development.X)),
            "variants": {
                name: {
                    "converged": result.converged,
                    "auprc": result.evaluation["threshold_independent"]["auprc"],
                    "auroc": result.evaluation["threshold_independent"]["auroc"],
                    "brier_score": result.evaluation["threshold_independent"][
                        "brier_score"
                    ],
                    "operating_threshold": result.evaluation[
                        "operating_point_fpr_le_max"
                    ]["threshold"],
                    "tpr": result.evaluation["operating_point_fpr_le_max"]["tpr"],
                    "fpr": result.evaluation["operating_point_fpr_le_max"]["fpr"],
                    "fit_seconds": result.fit_seconds,
                }
                for name, result in results.items()
            },
            "comparison_csv": str(comparison_path),
            "figures": {
                **{k: str(v) for k, v in curve_paths.items()},
                **{f"confusion_{k}": str(v) for k, v in confusion_paths.items()},
            },
            "note": (
                "Month 7 was not scored or evaluated. "
                "FPR<=5% is a literature-linked experimental operating point, "
                "not a real-bank business tolerance."
            ),
        },
    )

    for name, result in results.items():
        ranking = result.evaluation["threshold_independent"]
        op = result.evaluation["operating_point_fpr_le_max"]
        logger.info(
            "%s: AUPRC=%.6f AUROC=%.6f Brier=%.6f | "
            "FPR<=5%% threshold=%.6g TPR=%.6f precision=%.6f review=%.6f",
            name,
            ranking["auprc"],
            ranking["auroc"],
            ranking["brier_score"],
            op["threshold"],
            op["tpr"],
            op["precision"],
            op["review_rate"],
        )
    logger.info("Development baseline complete. Artefacts in %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
