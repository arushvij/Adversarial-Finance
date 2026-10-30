"""Credit-scoring metrics and decision-threshold selection.

Convention: a model outputs P(default). A loan is **approved** if
P(default) < threshold, **rejected** otherwise. "Accuracy" is computed at the
threshold, so it is meaningful under attack (an attack succeeds if it moves a
prediction across the threshold).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve


def youden_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising TPR - FPR (balanced-accuracy optimum).

    With ~20% defaults, 0.5 would reject almost nobody; Youden's J gives a
    sensible operating point. Fit on VALIDATION data only, then freeze.
    """
    fpr, tpr, thr = roc_curve(y_true, p)
    return float(thr[np.argmax(tpr - fpr)])


def ks_statistic(y_true: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic, the standard credit-risk separation metric."""
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Discrimination, calibration and decision-level metrics at ``threshold``."""
    pred = (p >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "ks": ks_statistic(y_true, p),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "approval_rate": float((p < threshold).mean()),
        "default_rate_among_approved": float(y_true[p < threshold].mean()) if (p < threshold).any() else float("nan"),
        "threshold": float(threshold),
    }
