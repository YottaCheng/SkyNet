"""CLI: bounded one-shot XGBoost improvement challenge.

Internal selection on months 0–4 / 5; one final refit on months 0–5;
single month-6 evaluation. Month 7 is never scored.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from baf_data.config import FROZEN_CONFIG
from baf_models.artifacts import (
    FROZEN_XGBOOST_BASELINE_DIR,
    XGBOOST_CHALLENGE_OUTPUT_ROOT,
    XGBOOST_CHALLENGE_RUN_ID,
    append_run_log,
    assert_challenge_output_isolated,
    load_saved_development_metrics,
    run_directory,
    save_challenge_artifacts,
    write_environment,
)
from baf_models.challenge import (
    CANDIDATE_ORDER,
    CHALLENGE_SEEDS,
    assert_candidates_match_frozen_spec,
    build_candidate_config,
    candidate_summary_to_dict,
    fit_final_challenge_on_train_dev,
    fit_internal_with_early_stopping,
    internal_result_to_dict,
    judge_improvement,
    load_challenge_bundle,
    resolve_final_config,
    select_candidate,
    selection_to_dict,
    summarise_candidate,
    validate_challenge_seeds,
)
from baf_models.evaluation import DEFAULT_MAX_FPR
from baf_models.training import config_to_dict

logger = logging.getLogger(__name__)

DEFAULT_RAW = Path("/Volumes/Study/ucl_dissertation_data/raw/baf/Base.csv")
DEFAULT_FROZEN_XGB = FROZEN_XGBOOST_BASELINE_DIR / "xgboost"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded XGBoost challenge: select on month 5, evaluate once on "
            "month 6. Does not score month 7 or expand the search space."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--output-root", type=Path, default=XGBOOST_CHALLENGE_OUTPUT_ROOT
    )
    parser.add_argument("--run-id", type=str, default=XGBOOST_CHALLENGE_RUN_ID)
    parser.add_argument("--max-fpr", type=float, default=DEFAULT_MAX_FPR)
    parser.add_argument(
        "--frozen-xgb-dir",
        type=Path,
        default=DEFAULT_FROZEN_XGB,
        help="Directory with frozen XGBoost seed-42 development metrics.",
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
    assert_candidates_match_frozen_spec()
    seeds = validate_challenge_seeds(CHALLENGE_SEEDS)
    output_dir = run_directory(args.output_root, args.run_id)
    assert_challenge_output_isolated(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.frozen_xgb_dir.is_dir():
        raise FileNotFoundError(
            f"Frozen XGBoost artefact directory not found: {args.frozen_xgb_dir}"
        )

    logger.info(
        "Challenge protocol note: month 6 baseline metrics were previously "
        "observed; this is a bounded development improvement, not a blind "
        "evaluation. Month 7 remains sealed."
    )
    logger.info("Loading challenge bundle (internal 0–4/5 + development 6).")
    bundle = load_challenge_bundle(args.raw, FROZEN_CONFIG)
    logger.info(
        "Rows: internal_fit=%d internal_val=%d development=%d.",
        len(bundle.internal_fit.X),
        len(bundle.internal_val.X),
        len(bundle.development.X),
    )

    internal_results = []
    for candidate_id in CANDIDATE_ORDER:
        for seed in seeds:
            model_config = build_candidate_config(candidate_id, random_state=seed)
            result = fit_internal_with_early_stopping(
                bundle,
                model_config,
                FROZEN_CONFIG,
                candidate_id=candidate_id,
                max_fpr=args.max_fpr,
            )
            internal_results.append(result)
            logger.info(
                "%s seed=%s month5 AUPRC=%.6f TPR=%.6f best_iteration=%d n_trees=%d",
                candidate_id,
                seed,
                result.auprc,
                result.tpr_at_fpr_le_5pct,
                result.best_iteration,
                result.n_trees_at_best,
            )

    summaries = [
        summarise_candidate(
            candidate_id,
            [r for r in internal_results if r.candidate_id == candidate_id],
        )
        for candidate_id in CANDIDATE_ORDER
    ]
    selection = select_candidate(summaries)
    logger.info(
        "Selected %s: %s | median_best_iteration=%d median_n_trees=%d",
        selection.selected_candidate_id,
        selection.reason,
        selection.median_best_iteration,
        selection.median_n_trees,
    )

    resolved = resolve_final_config(
        selection.selected_candidate_id,
        median_n_trees=selection.median_n_trees,
    )
    logger.info(
        "Final resolved config: n_estimators=%d max_depth=%d "
        "learning_rate=%s scale_pos_weight=%s random_state=%d",
        resolved.n_estimators,
        resolved.max_depth,
        resolved.learning_rate,
        resolved.scale_pos_weight,
        resolved.random_state,
    )

    logger.info("Refitting selected config on months 0–5; evaluating month 6 once.")
    final_result = fit_final_challenge_on_train_dev(
        bundle, resolved, FROZEN_CONFIG, max_fpr=args.max_fpr
    )

    frozen_metrics = load_saved_development_metrics(args.frozen_xgb_dir)
    improvement = judge_improvement(final_result.evaluation, frozen_metrics)
    c_rank = final_result.evaluation["threshold_independent"]
    c_op = final_result.evaluation["operating_point_fpr_le_max"]
    f_rank = frozen_metrics["threshold_independent"]
    f_op = frozen_metrics["operating_point_fpr_le_max"]
    comparison = {
        "split": "development",
        "month": 6,
        "frozen_artifact_dir": str(args.frozen_xgb_dir),
        "selected_candidate_id": selection.selected_candidate_id,
        "challenge": {
            "auprc": c_rank["auprc"],
            "auroc": c_rank["auroc"],
            "brier_score": c_rank["brier_score"],
            "tpr_at_fpr_le_5pct": c_op["tpr"],
            "precision_at_fpr_le_5pct": c_op["precision"],
            "review_rate_at_fpr_le_5pct": c_op["review_rate"],
            "fpr_at_fpr_le_5pct": c_op["fpr"],
            "threshold": c_op["threshold"],
            "tp": c_op["tp"],
            "fp": c_op["fp"],
            "tn": c_op["tn"],
            "fn": c_op["fn"],
            "fit_seconds": final_result.fit_seconds,
            "predict_seconds": final_result.predict_seconds,
            "model_complexity": (
                f"{selection.selected_candidate_id}; "
                f"n_estimators={resolved.n_estimators}, "
                f"max_depth={resolved.max_depth}, "
                f"learning_rate={resolved.learning_rate}, "
                f"scale_pos_weight={resolved.scale_pos_weight}"
            ),
        },
        "frozen_xgboost_seed42": {
            "auprc": f_rank["auprc"],
            "auroc": f_rank["auroc"],
            "brier_score": f_rank["brier_score"],
            "tpr_at_fpr_le_5pct": f_op["tpr"],
            "precision_at_fpr_le_5pct": f_op["precision"],
            "review_rate_at_fpr_le_5pct": f_op["review_rate"],
            "fpr_at_fpr_le_5pct": f_op["fpr"],
            "threshold": f_op["threshold"],
            "tp": f_op["tp"],
            "fp": f_op["fp"],
            "tn": f_op["tn"],
            "fn": f_op["fn"],
            "fit_seconds": frozen_metrics.get("fit_seconds"),
            "predict_seconds": frozen_metrics.get("predict_seconds"),
            "model_complexity": (
                "frozen C0; n_estimators=500, max_depth=6, "
                "learning_rate=0.05, scale_pos_weight=1"
            ),
        },
        "deltas": {
            "auprc": improvement["delta_auprc"],
            "tpr_at_fpr_le_5pct": improvement["delta_tpr_at_fpr_le_5pct"],
            "brier_score": improvement["delta_brier_score"],
        },
        "note": improvement["note"],
    }

    flat_summaries = []
    for summary in summaries:
        payload = candidate_summary_to_dict(summary)
        flat_summaries.append(
            {
                "candidate_id": payload["candidate_id"],
                "mean_auprc": payload["mean_auprc"],
                "mean_auroc": payload["mean_auroc"],
                "mean_brier": payload["mean_brier"],
                "mean_tpr_at_fpr_le_5pct": payload["mean_tpr_at_fpr_le_5pct"],
                "mean_precision_at_fpr_le_5pct": payload[
                    "mean_precision_at_fpr_le_5pct"
                ],
                "mean_review_rate_at_fpr_le_5pct": payload[
                    "mean_review_rate_at_fpr_le_5pct"
                ],
                "mean_fit_seconds": payload["mean_fit_seconds"],
                "best_iterations": list(payload["best_iterations"]),
                "n_trees_at_best": list(payload["n_trees_at_best"]),
                "median_best_iteration": payload["median_best_iteration"],
                "median_n_trees": payload["median_n_trees"],
                "max_depth": payload["max_depth"],
            }
        )

    write_environment(output_dir)
    paths = save_challenge_artifacts(
        output_dir=output_dir,
        internal_rows=[internal_result_to_dict(r) for r in internal_results],
        candidate_summaries=flat_summaries,
        selection=selection_to_dict(selection),
        resolved_config={
            "selected_candidate_id": selection.selected_candidate_id,
            "selection_reason": selection.reason,
            "median_best_iteration": selection.median_best_iteration,
            "median_n_trees": selection.median_n_trees,
            "model": config_to_dict(resolved),
            "formal_random_state": 42,
            "early_stopping_on_month6": False,
            "feature_columns": list(FROZEN_CONFIG.feature_columns),
            "raw_sha256": bundle.raw_sha256,
            "internal_fit_months": [0, 1, 2, 3, 4],
            "internal_val_month": 5,
            "final_train_months": [0, 1, 2, 3, 4, 5],
            "development_month": 6,
            "candidate_hyperparams": {
                s.candidate_id: s.hyperparams for s in summaries
            },
        },
        final_result=final_result,
        comparison=comparison,
        improvement=improvement,
        data_config=FROZEN_CONFIG,
        raw_sha256=bundle.raw_sha256,
    )

    append_run_log(
        output_dir / "run_log.jsonl",
        {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "outcome": "success",
            "run_id": args.run_id,
            "raw_sha256": bundle.raw_sha256,
            "selected_candidate_id": selection.selected_candidate_id,
            "median_n_trees": selection.median_n_trees,
            "month6_auprc": c_rank["auprc"],
            "improvement_decision": improvement["decision"],
            "meaningful_improvement_candidate": improvement[
                "meaningful_improvement_candidate"
            ],
            "artefacts": {k: str(v) for k, v in paths.items()},
            "note": (
                "Month 7 was not scored. Month 6 was evaluated once for the "
                "single selected configuration. No further tuning was performed."
            ),
        },
    )
    logger.info(
        "Challenge complete: decision=%s AUPRC=%.6f (delta=%+.6f). Artefacts: %s",
        improvement["decision"],
        c_rank["auprc"],
        improvement["delta_auprc"],
        output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
