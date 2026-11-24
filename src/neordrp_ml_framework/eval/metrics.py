"""Task-specific evaluation metrics.

These functions are shared by the trainer and the standalone quick evaluator.
Binary thresholds are always chosen on negatives only; that keeps
``recall_at_fpr`` honest when positives are rare.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def recall_at_fpr(
    labels: np.ndarray, scores: np.ndarray, fpr: float = 1e-3
) -> tuple[float, float, float]:
    """Return ``(recall, actual_fpr, threshold)`` at a negative-only FPR budget."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    negatives = scores[~labels]
    positives = scores[labels]
    if len(negatives) == 0 or len(positives) == 0:
        return float("nan"), float("nan"), float("nan")
    threshold = float(np.quantile(negatives, 1.0 - fpr, method="higher"))
    actual_fpr = float((negatives >= threshold).mean())
    recall = float((positives >= threshold).mean())
    return recall, actual_fpr, threshold


def binary_metrics(labels: np.ndarray, scores: np.ndarray, fpr: float = 1e-3) -> dict[str, float]:
    """AUROC, AUPRC, and recall at a fixed false-positive rate."""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    recall, actual_fpr, threshold = recall_at_fpr(labels, scores, fpr)
    # Subgroups in the quick report can contain a single class.  ROC AUC and
    # average precision are undefined there; return NaN instead of emitting a
    # warning for every metadata slice.
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
    return {
        "n": float(len(labels)),
        "n_pos": float(labels.sum()),
        "n_neg": float((labels == 0).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "recall_at_fpr": recall,
        "fpr_actual": actual_fpr,
        "threshold": threshold,
    }


def multiclass_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Macro-F1, top-1, and top-2 accuracy."""
    labels = np.asarray(labels).astype(np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    prediction = logits.argmax(axis=1)
    top2 = np.argsort(-logits, axis=1)[:, :2]
    return {
        "n": float(len(labels)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "top1": float((prediction == labels).mean()),
        "top2": float(np.mean([labels[i] in top2[i] for i in range(len(labels))])),
    }


def boundary_metrics(
    start_true: np.ndarray,
    end_true: np.ndarray,
    start_pred: np.ndarray,
    end_pred: np.ndarray,
    tolerance: int = 10,
) -> dict[str, float]:
    """Start/end exact-at-10-residue accuracy plus MAE and IoU."""
    start_true = np.asarray(start_true, dtype=np.int64)
    end_true = np.asarray(end_true, dtype=np.int64)
    start_pred = np.asarray(start_pred, dtype=np.int64)
    end_pred = np.asarray(end_pred, dtype=np.int64)
    start_error = np.abs(start_pred - start_true)
    end_error = np.abs(end_pred - end_true)
    intersection = np.maximum(0, np.minimum(end_pred, end_true) - np.maximum(start_pred, start_true) + 1)
    union = np.maximum(end_pred, end_true) - np.minimum(start_pred, start_true) + 1
    iou = intersection / np.maximum(union, 1)
    return {
        "n": float(len(start_true)),
        "start_e10": float((start_error <= tolerance).mean()),
        "end_e10": float((end_error <= tolerance).mean()),
        "both_e10": float(((start_error <= tolerance) & (end_error <= tolerance)).mean()),
        "mae_start": float(start_error.mean()),
        "mae_end": float(end_error.mean()),
        "iou_median": float(np.median(iou)),
        "iou_mean": float(iou.mean()),
    }
