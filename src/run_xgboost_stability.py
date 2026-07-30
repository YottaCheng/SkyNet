"""CLI: fixed-seed stability check for the frozen XGBoost baseline.

Fits months 0–5 and evaluates month 6 for each seed in the frozen list,
changing only ``random_state``. Does not select a best seed, does not
tune hyperparameters, and never scores month 7.
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
    XGBOOST_STABILITY_OUTPUT_ROOT,
    XGBOOST_STABILITY_RUN_ID,
    append_run_log,
    assert_stability_output_isolated,
    load_saved_development_metrics,
    run_directory,
    save_stability_artifacts,
    write_environment,
)
from baf_models.evaluation import DEFAULT_MAX_FPR
from baf_models.stability import (
    STABILITY_SEEDS,
    build_stability_summary,
    config_with_seed,
    metrics_row_from_result,
    validate_stability_seeds,
)
from baf_models.training import config_to_dict, fit_and_score_xgboost, load_train_dev_bundle
from baf_models.xgboost_model import XGBoostBaselineConfig

logger = logging.getLogger(__name__)

DEFAULT_RAW = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")
DEFAULT_YAML = (
    Path(__file__).resolve().parents[1] / "config" / "xgboost_baseline.yaml"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen XGBoost baseline across fixed random seeds on "
            "month 6 only. Does not tune parameters or score month 7."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--config", type=Path, default=DEFAULT_YAML)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=XGBOOST_STABILITY_OUTPUT_ROOT,
    )
    parser.add_argument("--run-id", type=str, default=XGBOOST_STABILITY_RUN_ID)
    parser.add_argument("--max-fpr", type=float, default=DEFAULT_MAX_FPR)
    parser.add_argument(
        "--lr-artifact-dir",
        type=Path,
        default=DEFAULT_LR_UNWEIGHTED_DIR,
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
    seeds = validate_stability_seeds(STABILITY_SEEDS)
    output_dir = run_directory(args.output_root, args.run_id)
    assert_stability_output_isolated(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.config.is_file():
        raise FileNotFoundError(f"XGBoost config not found: {args.config}")
    if not args.lr_artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Unweighted LR artefact directory not found: {args.lr_artifact_dir}"
        )

    base_config = XGBoostBaselineConfig.from_yaml(args.config)
    if base_config.random_state != 42:
        raise RuntimeError(
            "Formal YAML random_state must remain 42 before the stability loop; "
            f"got {base_config.random_state}."
        )

    logger.info("Loading train/dev only from %s", args.raw)
    bundle = load_train_dev_bundle(args.raw, FROZEN_CONFIG)

    seed_rows: list[dict] = []
    for seed in seeds:
        model_config = config_with_seed(base_config, seed)
        logger.info(
            "Stability seed=%s: fitting on train only with frozen hyperparameters.",
            seed,
        )
        result = fit_and_score_xgboost(
            bundle,
            model_config,
            FROZEN_CONFIG,
            max_fpr=args.max_fpr,
            variant_name=f"xgboost_seed_{seed}",
        )
        row = metrics_row_from_result(result, seed)
        seed_rows.append(row)
        logger.info(
            "seed=%s AUPRC=%.6f AUROC=%.6f TPR@FPR<=5%%=%.6f fit=%.3fs",
            seed,
            row["auprc"],
            row["auroc"],
            row["tpr_at_fpr_le_5pct"],
            row["fit_seconds"],
        )

    lr_metrics = load_saved_development_metrics(args.lr_artifact_dir)
    summary = build_stability_summary(
        seed_rows,
        lr_metrics=lr_metrics,
        base_random_state=base_config.random_state,
    )

    write_environment(output_dir)
    paths = save_stability_artifacts(
        output_dir=output_dir,
        seed_rows=seed_rows,
        summary=summary,
        base_config={
            "yaml_path": str(args.config),
            "model": config_to_dict(base_config),
            "varied_parameter": "random_state",
            "seeds": list(seeds),
            "formal_candidate_random_state": 42,
            "feature_columns": list(FROZEN_CONFIG.feature_columns),
            "raw_sha256": bundle.raw_sha256,
            "training_months": list(FROZEN_CONFIG.split_months["train"]),
            "development_month": 6,
        },
    )

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
            "seeds": list(seeds),
            "aggregates": summary["aggregates"],
            "all_seeds_auprc_gt_lr": summary["all_seeds_auprc_gt_lr"],
            "artefacts": {key: str(path) for key, path in paths.items()},
            "note": summary["note"],
        },
    )
    logger.info("Stability check complete. Artefacts in %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
