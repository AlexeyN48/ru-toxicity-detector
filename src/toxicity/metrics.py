"""Quality metrics and threshold selection (no torch dependency)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, min_recall: float = 0.85) -> float:
    """Threshold with the best F1 among those keeping recall >= `min_recall`.

    Falls back to the best-F1 threshold if no threshold reaches the required recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more element than thresholds
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    allowed = recall >= min_recall
    if not allowed.any():
        allowed = np.ones_like(recall, dtype=bool)
    best = np.argmax(np.where(allowed, f1, -1.0))
    return float(thresholds[best])


def classification_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1_toxic": float(f1_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_toxic": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_toxic": float(recall_score(y_true, y_pred)),
        "threshold": float(threshold),
        "n": len(y_true),
    }


TARGETS = {
    "roc_auc": 0.95,
    "pr_auc": 0.90,
    "f1_toxic": 0.85,
    "recall_toxic": 0.85,
    "precision_toxic": 0.80,
}


def check_targets(report: dict) -> dict[str, bool]:
    """Compare metrics with the target values from the Model Card."""
    return {name: report[name] >= target for name, target in TARGETS.items()}
