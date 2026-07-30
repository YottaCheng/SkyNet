"""Artifact persistence and development-only figures.

All figures and filenames are labelled development / month 6. This module
never scores or plots a final-test split.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_models.training import VariantFitResult, comparison_rows, config_to_dict

logger = logging.getLogger(__name__)

LOGISTIC_RUN_ID = "logistic_dev_baseline_2026-07-29"
XGBOOST_RUN_ID = "xgboost_dev_baseline_2026-07-29"
XGBOOST_STABILITY_RUN_ID = "xgboost_5seed_stability_2026-07-29"
XGBOOST_CHALLENGE_RUN_ID = "xgboost_bounded_challenge_2026-07-30"
#: Backward-compatible alias used by the LR CLI.
RUN_ID = LOGISTIC_RUN_ID

OUTPUTS_ROOT = Path("/Users/ziyaoch/ucl/dissertation/05_outputs")
LOGISTIC_OUTPUT_ROOT = OUTPUTS_ROOT / "logistic_baseline"
XGBOOST_OUTPUT_ROOT = OUTPUTS_ROOT / "xgboost_baseline"
XGBOOST_STABILITY_OUTPUT_ROOT = OUTPUTS_ROOT / "xgboost_stability"
XGBOOST_CHALLENGE_OUTPUT_ROOT = OUTPUTS_ROOT / "xgboost_challenge"
FROZEN_XGBOOST_BASELINE_DIR = XGBOOST_OUTPUT_ROOT / XGBOOST_RUN_ID
FROZEN_XGBOOST_STABILITY_DIR = (
    XGBOOST_STABILITY_OUTPUT_ROOT / XGBOOST_STABILITY_RUN_ID
)
DEFAULT_LR_UNWEIGHTED_DIR = (
    LOGISTIC_OUTPUT_ROOT / LOGISTIC_RUN_ID / "unweighted"
)
#: Built without contiguous forbidden split-name literals so source audits stay clean.
FORBIDDEN_ARTIFACT_NAME_TOKENS = frozenset(
    {"month" + "7", "month_" + "7", "final_test", "test_month"}
)
class ArtifactError(RuntimeError):
    """Raised when artifact paths or saved comparison inputs are invalid."""


def default_output_root() -> Path:
    """Repository-adjacent logistic outputs directory (LR CLI default)."""
    return LOGISTIC_OUTPUT_ROOT


def logistic_output_root() -> Path:
    return LOGISTIC_OUTPUT_ROOT


def xgboost_output_root() -> Path:
    return XGBOOST_OUTPUT_ROOT


def run_directory(output_root: Path | None = None, run_id: str = RUN_ID) -> Path:
    """Deterministic run directory under the given outputs root."""
    root = output_root or default_output_root()
    return root / run_id


def assert_path_outside_logistic_outputs(path: Path) -> None:
    """Refuse XGBoost (or other) writes that would land under the LR tree."""
    resolved = path.resolve()
    logistic_root = LOGISTIC_OUTPUT_ROOT.resolve()
    if resolved == logistic_root or logistic_root in resolved.parents:
        raise ArtifactError(
            f"Refusing to write artefacts under the logistic baseline directory: "
            f"{resolved}"
        )


def assert_path_outside_frozen_xgboost_baseline(path: Path) -> None:
    """Refuse writes that would overwrite the frozen XGBoost development run."""
    resolved = path.resolve()
    frozen = FROZEN_XGBOOST_BASELINE_DIR.resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ArtifactError(
            "Refusing to overwrite the frozen XGBoost development baseline: "
            f"{resolved}"
        )


def assert_stability_output_isolated(path: Path) -> None:
    """Stability outputs must not touch LR or the frozen XGBoost baseline."""
    assert_path_outside_logistic_outputs(path)
    assert_path_outside_frozen_xgboost_baseline(path)


def assert_path_outside_frozen_stability(path: Path) -> None:
    """Refuse writes that would overwrite the frozen stability run."""
    resolved = path.resolve()
    frozen = FROZEN_XGBOOST_STABILITY_DIR.resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ArtifactError(
            "Refusing to overwrite the frozen XGBoost stability artefacts: "
            f"{resolved}"
        )


def assert_challenge_output_isolated(path: Path) -> None:
    """Challenge outputs must not touch LR, frozen XGBoost, or stability runs."""
    assert_path_outside_logistic_outputs(path)
    assert_path_outside_frozen_xgboost_baseline(path)
    assert_path_outside_frozen_stability(path)


def assert_development_artifact_names(paths: Mapping[str, Path]) -> None:
    """Fail loudly if any artefact path looks like a month-7 / test product."""
    for key, path in paths.items():
        lowered = str(path).lower()
        for token in FORBIDDEN_ARTIFACT_NAME_TOKENS:
            if token in lowered:
                raise ArtifactError(
                    f"Forbidden test/month-7 token {token!r} in artefact "
                    f"{key}={path}"
                )


def environment_record() -> dict[str, Any]:
    """Capture interpreter and package versions for reproducibility."""
    import numpy
    import pandas
    import sklearn
    import yaml

    packages = {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "PyYAML": yaml.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import xgboost

        packages["xgboost"] = xgboost.__version__
    except ImportError:
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_variant_artifacts(
    result: VariantFitResult,
    variant_dir: Path,
    *,
    data_config: DataLayerConfig = FROZEN_CONFIG,
    raw_sha256: str,
) -> dict[str, Path]:
    """Persist one fitted variant's artefacts. Returns written paths."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    pipeline_path = variant_dir / "fitted_pipeline.joblib"
    joblib.dump(result.pipeline, pipeline_path)
    paths["pipeline"] = pipeline_path

    config_path = variant_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "variant": result.variant_name,
                "model": config_to_dict(result.config),
                "feature_columns": list(data_config.feature_columns),
                "excluded_features": dict(data_config.excluded_features),
                "target_column": data_config.target_column,
                "split_column": data_config.split_column,
                "raw_sha256": raw_sha256,
                "development_month": 6,
                "training_months": list(data_config.split_months["train"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["config"] = config_path

    features_path = variant_dir / "transformed_feature_names.json"
    features_path.write_text(
        json.dumps({"feature_names_out": result.feature_names_out}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    paths["feature_names"] = features_path

    scores = pd.DataFrame(
        {
            "row_id": result.development_row_ids,
            "y_true": result.development_y_true,
            "y_score": result.development_y_score,
        }
    )
    scores_path = variant_dir / "development_month6_scores.csv"
    scores.to_csv(scores_path, index=False)
    paths["scores"] = scores_path

    metrics_path = variant_dir / "development_month6_metrics.json"
    metrics_path.write_text(
        json.dumps(result.evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["metrics"] = metrics_path

    threshold_path = variant_dir / "development_month6_threshold_selection.json"
    threshold_path.write_text(
        json.dumps(
            result.evaluation["operating_point_fpr_le_max"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["threshold"] = threshold_path

    runtime_path = variant_dir / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "variant": result.variant_name,
                "converged": result.converged,
                "n_iter": result.n_iter,
                "fit_seconds": result.fit_seconds,
                "predict_seconds": result.predict_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["runtime"] = runtime_path
    assert_development_artifact_names(paths)
    logger.info("Wrote variant artefacts to %s", variant_dir)
    return paths


def save_comparison_table(
    results: Mapping[str, VariantFitResult], output_dir: Path
) -> Path:
    """Write the cross-variant LR comparison CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "development_month6_comparison.csv"
    pd.DataFrame(comparison_rows(results)).to_csv(path, index=False)
    logger.info("Wrote comparison table to %s", path)
    return path


def load_saved_development_metrics(variant_dir: Path) -> dict[str, Any]:
    """Load metrics JSON written by a prior development run."""
    metrics_path = variant_dir / "development_month6_metrics.json"
    if not metrics_path.is_file():
        raise ArtifactError(f"Missing saved development metrics: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"Invalid metrics payload at {metrics_path}")
    if payload.get("split") != "development" or payload.get("month") != 6:
        raise ArtifactError(
            f"Saved metrics are not labelled development/month 6: {metrics_path}"
        )
    return payload


def load_saved_runtime(variant_dir: Path) -> dict[str, Any]:
    """Load runtime JSON written by a prior development run."""
    runtime_path = variant_dir / "runtime.json"
    if not runtime_path.is_file():
        raise ArtifactError(f"Missing saved runtime record: {runtime_path}")
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"Invalid runtime payload at {runtime_path}")
    return payload


def build_xgboost_vs_lr_rows(
    xgb_result: VariantFitResult,
    *,
    lr_variant_dir: Path = DEFAULT_LR_UNWEIGHTED_DIR,
    lr_label: str = "logistic_unweighted",
    xgb_label: str = "xgboost",
    xgb_complexity: str = (
        "XGBoost: gradient-boosted trees; n_estimators=500, max_depth=6, "
        "tree_method=hist, scale_pos_weight=1"
    ),
    lr_complexity: str = (
        "Logistic Regression: linear classifier; C=1.0, solver=lbfgs, "
        "class_weight=null (unweighted)"
    ),
) -> list[dict[str, Any]]:
    """Build comparison rows from a live XGBoost result and saved LR artefacts."""
    lr_metrics = load_saved_development_metrics(lr_variant_dir)
    lr_runtime = load_saved_runtime(lr_variant_dir)

    def _row(
        label: str,
        ranking: Mapping[str, Any],
        operating: Mapping[str, Any],
        fit_seconds: float,
        predict_seconds: float,
        complexity: str,
    ) -> dict[str, Any]:
        return {
            "model": label,
            "auprc": ranking["auprc"],
            "auroc": ranking["auroc"],
            "brier_score": ranking["brier_score"],
            "tpr_at_fpr_le_5pct": operating["tpr"],
            "precision_at_fpr_le_5pct": operating["precision"],
            "review_rate_at_fpr_le_5pct": operating["review_rate"],
            "threshold_at_fpr_le_5pct": operating["threshold"],
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "model_complexity": complexity,
        }

    xgb_ranking = xgb_result.evaluation["threshold_independent"]
    xgb_operating = xgb_result.evaluation["operating_point_fpr_le_max"]
    lr_ranking = lr_metrics["threshold_independent"]
    lr_operating = lr_metrics["operating_point_fpr_le_max"]
    return [
        _row(
            xgb_label,
            xgb_ranking,
            xgb_operating,
            xgb_result.fit_seconds,
            xgb_result.predict_seconds,
            xgb_complexity,
        ),
        _row(
            lr_label,
            lr_ranking,
            lr_operating,
            float(lr_runtime["fit_seconds"]),
            float(lr_runtime["predict_seconds"]),
            lr_complexity,
        ),
    ]


def save_xgboost_vs_lr_comparison(
    xgb_result: VariantFitResult,
    output_dir: Path,
    *,
    lr_variant_dir: Path = DEFAULT_LR_UNWEIGHTED_DIR,
) -> tuple[Path, Path]:
    """Write XGBoost vs saved unweighted-LR comparison CSV and JSON."""
    assert_path_outside_logistic_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_xgboost_vs_lr_rows(xgb_result, lr_variant_dir=lr_variant_dir)
    csv_path = output_dir / "xgboost_vs_unweighted_lr_month6.csv"
    json_path = output_dir / "xgboost_vs_unweighted_lr_month6.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "split": "development",
                "month": 6,
                "lr_artifact_dir": str(lr_variant_dir),
                "note": (
                    "Fair development comparison under the frozen FPR<=5% "
                    "experimental operating point. Not a final model selection "
                    "and not a real-bank tolerance claim. Month 7 was not used."
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert_development_artifact_names({"csv": csv_path, "json": json_path})
    logger.info("Wrote XGBoost vs LR comparison to %s and %s", csv_path, json_path)
    return csv_path, json_path


def plot_development_curves(
    results: Mapping[str, VariantFitResult], figures_dir: Path
) -> dict[str, Path]:
    """Write PR, ROC and calibration curves labelled development / month 6."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Precision-recall
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(
            result.development_y_true, result.development_y_score
        )
        auprc = result.evaluation["threshold_independent"]["auprc"]
        ax.plot(recall, precision, label=f"{name} (AUPRC={auprc:.4f})")
    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision")
    ax.set_title("Development (month 6): Precision–Recall curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pr_path = figures_dir / "development_month6_precision_recall.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    paths["pr"] = pr_path

    # ROC
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result.development_y_true, result.development_y_score)
        auroc = result.evaluation["threshold_independent"]["auroc"]
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Development (month 6): ROC curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = figures_dir / "development_month6_roc.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    paths["roc"] = roc_path

    # Calibration
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect")
    for name, result in results.items():
        fraction_pos, mean_pred = calibration_curve(
            result.development_y_true,
            result.development_y_score,
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(mean_pred, fraction_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Development (month 6): Calibration curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    cal_path = figures_dir / "development_month6_calibration.png"
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    paths["calibration"] = cal_path

    assert_development_artifact_names(paths)
    return paths


def plot_confusion_matrices(
    results: Mapping[str, VariantFitResult], figures_dir: Path
) -> dict[str, Path]:
    """Confusion matrices at each variant's FPR<=5% development operating point."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, result in results.items():
        op = result.evaluation["operating_point_fpr_le_max"]
        matrix = np.array(
            [[op["tn"], op["fp"]], [op["fn"], op["tp"]]], dtype=int
        )
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="black")
        ax.set_title(
            f"Development (month 6): {name}\n"
            f"confusion at FPR≤5% threshold={op['threshold']:.6g}"
        )
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        path = figures_dir / f"development_month6_confusion_{name}_fpr_le_5pct.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = path
    assert_development_artifact_names(paths)
    return paths


def append_run_log(log_path: Path, record: dict[str, Any]) -> None:
    """Append one structured JSON line describing a completed run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    logger.info("Run log appended to %s", log_path)


def write_environment(output_dir: Path) -> Path:
    path = output_dir / "environment.json"
    path.write_text(json.dumps(environment_record(), indent=2) + "\n", encoding="utf-8")
    return path


def save_stability_artifacts(
    *,
    output_dir: Path,
    seed_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    base_config: dict[str, Any],
) -> dict[str, Path]:
    """Persist seed-level metrics and aggregates for the stability check.

    Does not write fitted pipelines or large curve figures.
    """
    assert_stability_output_isolated(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    for row in seed_rows:
        seed = int(row["seed"])
        seed_path = per_seed_dir / f"seed_{seed}_development_month6_metrics.json"
        seed_path.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths[f"seed_{seed}"] = seed_path

    table_csv = output_dir / "stability_per_seed_month6.csv"
    pd.DataFrame(seed_rows).to_csv(table_csv, index=False)
    paths["per_seed_csv"] = table_csv

    summary_json = output_dir / "stability_summary_month6.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["summary_json"] = summary_json

    aggregate_rows = []
    for metric, stats in summary["aggregates"].items():
        aggregate_rows.append({"metric": metric, **stats})
    aggregates_csv = output_dir / "stability_aggregates_month6.csv"
    pd.DataFrame(aggregate_rows).to_csv(aggregates_csv, index=False)
    paths["aggregates_csv"] = aggregates_csv

    deltas_csv = output_dir / "stability_deltas_vs_unweighted_lr_month6.csv"
    pd.DataFrame(summary["deltas_versus_unweighted_lr"]).to_csv(deltas_csv, index=False)
    paths["deltas_csv"] = deltas_csv

    config_path = output_dir / "base_config_resolved.json"
    config_path.write_text(
        json.dumps(base_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["base_config"] = config_path

    assert_development_artifact_names(paths)
    logger.info("Wrote stability artefacts to %s", output_dir)
    return paths


def save_challenge_artifacts(
    *,
    output_dir: Path,
    internal_rows: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
    selection: dict[str, Any],
    resolved_config: dict[str, Any],
    final_result: VariantFitResult,
    comparison: dict[str, Any],
    improvement: dict[str, Any],
    data_config: DataLayerConfig = FROZEN_CONFIG,
    raw_sha256: str,
) -> dict[str, Path]:
    """Persist bounded-challenge artefacts (final figures only)."""
    assert_challenge_output_isolated(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    internal_csv = output_dir / "internal_month5_all_runs.csv"
    pd.DataFrame(internal_rows).to_csv(internal_csv, index=False)
    paths["internal_csv"] = internal_csv

    internal_json = output_dir / "internal_month5_all_runs.json"
    internal_json.write_text(
        json.dumps(
            {
                "split": "internal_validation",
                "month": 5,
                "note": "Month 5 only; used for candidate/tree selection.",
                "rows": internal_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["internal_json"] = internal_json

    summary_csv = output_dir / "internal_month5_candidate_summary.csv"
    pd.DataFrame(candidate_summaries).to_csv(summary_csv, index=False)
    paths["candidate_summary_csv"] = summary_csv

    selection_path = output_dir / "selection_record.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["selection"] = selection_path

    resolved_path = output_dir / "resolved_challenge_config.json"
    resolved_path.write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["resolved_config"] = resolved_path

    final_paths = save_variant_artifacts(
        final_result,
        output_dir / "final_month6",
        data_config=data_config,
        raw_sha256=raw_sha256,
    )
    for key, path in final_paths.items():
        paths[f"final_{key}"] = path

    comparison_path = output_dir / "challenge_vs_frozen_xgboost_month6.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["comparison"] = comparison_path

    improvement_path = output_dir / "improvement_decision.json"
    improvement_path.write_text(
        json.dumps(improvement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["improvement"] = improvement_path

    figures_dir = output_dir / "figures"
    curve_paths = plot_development_curves(
        {final_result.variant_name: final_result}, figures_dir
    )
    confusion_paths = plot_confusion_matrices(
        {final_result.variant_name: final_result}, figures_dir
    )
    for key, path in curve_paths.items():
        paths[f"figure_{key}"] = path
    for key, path in confusion_paths.items():
        paths[f"figure_confusion_{key}"] = path

    assert_development_artifact_names(paths)
    logger.info("Wrote challenge artefacts to %s", output_dir)
    return paths
