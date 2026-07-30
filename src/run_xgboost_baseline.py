"""CLI entry point for the XGBoost development baseline.

Fits the frozen XGBoost configuration on months 0–5 and evaluates on
month 6 only. Compares against the saved unweighted Logistic Regression
development artefacts. Month 7 is never scored or evaluated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from baf_data.config import FROZEN_CONFIG
from baf_models.artifacts import (
    DEFAULT_LR_UNWEIGHTED_DIR,
    XGBOOST_OUTPUT_ROOT,
    XGBOOST_RUN_ID,
    append_run_log,
    assert_path_outside_logistic_outputs,
    plot_confusion_matrices,
    plot_development_curves,
    run_directory,
    save_variant_artifacts,
    save_xgboost_vs_lr_comparison,
    write_environment,
)
from baf_models.evaluation import DEFAULT_MAX_FPR
from baf_models.training import fit_and_score_xgboost, load_train_dev_bundle
from baf_models.xgboost_model import XGBoostBaselineConfig

logger = logging.getLogger(__name__)

DEFAULT_RAW = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")
DEFAULT_YAML = (
    Path(__file__).resolve().parents[1] / "config" / "xgboost_baseline.yaml"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the frozen XGBoost baseline on months 0–5 and evaluate on "
            "month 6 only. Does not score month 7. Does not tune hyperparameters."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_YAML,
        help="Path to xgboost_baseline.yaml (CLI does not override model params).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=XGBOOST_OUTPUT_ROOT,
        help="Root under which the deterministic run directory is created.",
    )
    parser.add_argument("--run-id", type=str, default=XGBOOST_RUN_ID)
    parser.add_argument("--max-fpr", type=float, default=DEFAULT_MAX_FPR)
    parser.add_argument(
        "--lr-artifact-dir",
        type=Path,
        default=DEFAULT_LR_UNWEIGHTED_DIR,
        help="Directory containing saved unweighted LR development artefacts.",
    )
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
    assert_path_outside_logistic_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.config.is_file():
        raise FileNotFoundError(f"XGBoost config not found: {args.config}")
    if not args.lr_artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Unweighted LR artefact directory not found: {args.lr_artifact_dir}"
        )

    model_config = XGBoostBaselineConfig.from_yaml(args.config)
    logger.info(
        "Resolved frozen XGBoost config from %s: objective=%s, eval_metric=%s, "
        "n_estimators=%s, max_depth=%s, learning_rate=%s, scale_pos_weight=%s, "
        "tree_method=%s, random_state=%s, n_jobs=%s.",
        args.config,
        model_config.objective,
        model_config.eval_metric,
        model_config.n_estimators,
        model_config.max_depth,
        model_config.learning_rate,
        model_config.scale_pos_weight,
        model_config.tree_method,
        model_config.random_state,
        model_config.n_jobs,
    )

    logger.info("Loading train/dev only from %s", args.raw)
    bundle = load_train_dev_bundle(args.raw, FROZEN_CONFIG)

    result = fit_and_score_xgboost(
        bundle,
        model_config,
        FROZEN_CONFIG,
        max_fpr=args.max_fpr,
        variant_name="xgboost",
    )
    results = {"xgboost": result}

    write_environment(output_dir)
    variant_paths = save_variant_artifacts(
        result,
        output_dir / "xgboost",
        data_config=FROZEN_CONFIG,
        raw_sha256=bundle.raw_sha256,
    )
    comparison_csv, comparison_json = save_xgboost_vs_lr_comparison(
        result,
        output_dir,
        lr_variant_dir=args.lr_artifact_dir,
    )
    figures_dir = output_dir / "figures"
    curve_paths = plot_development_curves(results, figures_dir)
    confusion_paths = plot_confusion_matrices(results, figures_dir)

    ranking = result.evaluation["threshold_independent"]
    op = result.evaluation["operating_point_fpr_le_max"]
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
            "model": {
                "variant": result.variant_name,
                "auprc": ranking["auprc"],
                "auroc": ranking["auroc"],
                "brier_score": ranking["brier_score"],
                "operating_threshold": op["threshold"],
                "tpr": op["tpr"],
                "fpr": op["fpr"],
                "precision": op["precision"],
                "review_rate": op["review_rate"],
                "fit_seconds": result.fit_seconds,
                "predict_seconds": result.predict_seconds,
                "resolved_config": variant_paths["config"].name,
            },
            "comparison_csv": str(comparison_csv),
            "comparison_json": str(comparison_json),
            "lr_artifact_dir": str(args.lr_artifact_dir),
            "figures": {
                **{k: str(v) for k, v in curve_paths.items()},
                **{f"confusion_{k}": str(v) for k, v in confusion_paths.items()},
            },
            "note": (
                "Month 7 was not scored or evaluated. "
                "FPR<=5% is a literature-linked experimental operating point, "
                "not a real-bank business tolerance. "
                "This run does not declare a final statistical defence model."
            ),
        },
    )

    logger.info(
        "xgboost: AUPRC=%.6f AUROC=%.6f Brier=%.6f | "
        "FPR<=5%% threshold=%.6g TPR=%.6f precision=%.6f review=%.6f | "
        "fit=%.3fs predict=%.3fs",
        ranking["auprc"],
        ranking["auroc"],
        ranking["brier_score"],
        op["threshold"],
        op["tpr"],
        op["precision"],
        op["review_rate"],
        result.fit_seconds,
        result.predict_seconds,
    )
    logger.info("XGBoost development baseline complete. Artefacts in %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
