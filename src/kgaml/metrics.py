"""Imbalance-aware evaluation for the illicit (positive) class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising illicit F1 on a validation window."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    yhat = (p >= threshold).astype(int)
    k = max(1, int(round(0.01 * len(y))))  # analyst budget: top 1% of transactions
    top = np.argsort(-p)[:k]
    return {
        "f1": f1_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "pr_auc": average_precision_score(y, p) if y.any() else np.nan,
        "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "precision_at_1pct": float(y[top].mean()),
        "threshold": threshold,
    }
