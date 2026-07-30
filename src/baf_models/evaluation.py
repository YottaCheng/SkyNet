"""Development-set metrics and FPR-constrained threshold selection.

All functions operate on score arrays already produced for the
development split. This module never loads data, never fits a model and
never accepts a month-7 / test split argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

#: Literature-linked experimental FPR ceiling for the development
#: operating point (BAF paper benchmark convention). Not a claimed
#: real-bank business tolerance.
DEFAULT_MAX_FPR = 0.05


@dataclass(frozen=True)
class ConfusionCounts:
    """Confusion-matrix counts at a fixed decision threshold."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def tpr(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return float(self.fp / denom) if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return float(self.tn / denom) if denom else 0.0

    @property
    def review_rate(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return float((self.tp + self.fp) / total) if total else 0.0


@dataclass(frozen=True)
class ThresholdIndependentMetrics:
    """Ranking and calibration metrics that do not need a threshold."""

    auprc: float
    auroc: float
    brier_score: float


@dataclass(frozen=True)
class ThresholdDependentMetrics:
    """Classification metrics at one decision threshold."""

    threshold: float
    counts: ConfusionCounts
    precision: float
    recall: float
    fpr: float
    specificity: float
    review_rate: float

    @property
    def tpr(self) -> float:
        """Alias for recall (true positive rate)."""
        return self.recall


@dataclass(frozen=True)
class OperatingPoint:
    """Selected FPR-constrained development operating point."""

    threshold: float
    max_fpr: float
    metrics: ThresholdDependentMetrics
    rule: str = (
        "Maximise TPR subject to FPR <= max_fpr; "
        "ties broken by lowest FPR then highest threshold. "
        "Literature-linked experimental benchmark, not a real-bank tolerance."
    )


def confusion_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> ConfusionCounts:
    """Compute confusion counts using ``score >= threshold`` as positive."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ConfusionCounts(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def threshold_dependent_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> ThresholdDependentMetrics:
    """Assemble threshold-dependent metrics at one operating point."""
    counts = confusion_at_threshold(y_true, y_score, threshold)
    return ThresholdDependentMetrics(
        threshold=float(threshold),
        counts=counts,
        precision=counts.precision,
        recall=counts.tpr,
        fpr=counts.fpr,
        specificity=counts.specificity,
        review_rate=counts.review_rate,
    )


def threshold_independent_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> ThresholdIndependentMetrics:
    """Compute AUPRC (primary), AUROC and Brier score."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    return ThresholdIndependentMetrics(
        auprc=float(average_precision_score(y_true, y_score)),
        auroc=float(roc_auc_score(y_true, y_score)),
        brier_score=float(brier_score_loss(y_true, y_score)),
    )


def select_fpr_constrained_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> OperatingPoint:
    """Select the development threshold under the frozen FPR rule.

    Candidates are every unique score plus the endpoints 0 and 1.
    Among thresholds with empirical FPR <= ``max_fpr``, maximise TPR;
    ties break to the lowest FPR, then to the highest threshold.

    Implementation is O(n log n): scores are sorted once and confusion
    counts are maintained with cumulative positives/negatives rather than
    recomputing a full confusion matrix at every candidate.
    """
    if not (0.0 < max_fpr <= 1.0):
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}.")

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        raise ValueError("Cannot select a threshold on an empty development set.")
    if not np.isfinite(y_score).all():
        raise ValueError("Development scores contain non-finite values.")

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    # Prefix sums of labels in ascending score order: for threshold t,
    # predicted negatives are scores < t (strict), i.e. the prefix before
    # the first score >= t.
    prefix_pos = np.concatenate([[0], np.cumsum(sorted_true)])
    # prefix_pos[i] = number of positives among the i lowest scores.

    candidates = np.unique(np.concatenate([sorted_scores, np.array([0.0, 1.0])]))
    best_key: tuple[float, float, float] | None = None
    best_threshold: float | None = None

    for threshold in candidates:
        # First index with score >= threshold.
        start = int(np.searchsorted(sorted_scores, threshold, side="left"))
        # Predicted negative = scores < threshold = first `start` rows.
        fn = int(prefix_pos[start])
        tn = start - fn
        tp = n_pos - fn
        fp = n_neg - tn
        tpr = float(tp / n_pos) if n_pos else 0.0
        fpr = float(fp / n_neg) if n_neg else 0.0
        if fpr > max_fpr:
            continue
        key = (-tpr, fpr, -float(threshold))
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)

    if best_threshold is None:
        raise ValueError(
            f"No finite threshold achieves FPR <= {max_fpr} on the development set."
        )

    metrics = threshold_dependent_metrics(y_true, y_score, best_threshold)
    return OperatingPoint(threshold=best_threshold, max_fpr=max_fpr, metrics=metrics)


def evaluate_development_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> dict[str, Any]:
    """Full development evaluation record for one score vector.

    Returns a JSON-serialisable dictionary with threshold-independent
    metrics, the default 0.5 operating point and the FPR-constrained
    selected operating point. Never evaluates a test split.
    """
    ranking = threshold_independent_metrics(y_true, y_score)
    at_half = threshold_dependent_metrics(y_true, y_score, 0.5)
    operating = select_fpr_constrained_threshold(y_true, y_score, max_fpr=max_fpr)
    return {
        "split": "development",
        "month": 6,
        "n_rows": int(len(y_true)),
        "n_positives": int(np.asarray(y_true).sum()),
        "threshold_independent": asdict(ranking),
        "at_threshold_0_5": _metrics_to_dict(at_half),
        "operating_point_fpr_le_max": {
            "max_fpr": operating.max_fpr,
            "rule": operating.rule,
            **_metrics_to_dict(operating.metrics),
        },
    }


def _metrics_to_dict(metrics: ThresholdDependentMetrics) -> dict[str, Any]:
    return {
        "threshold": metrics.threshold,
        "tp": metrics.counts.tp,
        "fp": metrics.counts.fp,
        "tn": metrics.counts.tn,
        "fn": metrics.counts.fn,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "tpr": metrics.recall,
        "fpr": metrics.fpr,
        "specificity": metrics.specificity,
        "review_rate": metrics.review_rate,
    }
