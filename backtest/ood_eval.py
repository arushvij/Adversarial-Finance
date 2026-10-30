"""Out-of-distribution evaluation across time periods.

The OOD set (2015-2016) was held out entirely from training and tuning.
This module slices it into monthly cohorts and computes metrics per cohort,
revealing how model performance drifts over time — a key risk management metric.

Why this matters for the resume:
  A model that holds AUC on in-time test but drops on OOD demonstrates
  real concept drift (the lending market changed post-2014). Quantifying this
  drift is standard practice in production ML for finance.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from common import get_logger
from models.evaluate import compute_metrics

log = get_logger(__name__)

PredictFn = Callable[[np.ndarray], np.ndarray]


def evaluate_over_time(
    X_ood: np.ndarray,
    y_ood: np.ndarray,
    issue_dates: np.ndarray,       # datetime64 array aligned to X_ood
    predictors: dict[str, PredictFn],
    thresholds: dict[str, float],
    freq: str = "Q",               # "M" monthly, "Q" quarterly
    min_samples: int = 200,
) -> pd.DataFrame:
    """Compute metrics per time cohort for each model.

    Args:
        freq: pandas frequency string for grouping ("Q" = quarterly recommended
              since monthly cohorts on 662K OOD rows are very stable).
    Returns:
        DataFrame with columns: model, period, n, auc, ks, accuracy,
        approval_rate, default_rate
    """
    dates = pd.to_datetime(issue_dates)
    periods = pd.Series(dates).dt.to_period(freq)
    unique_periods = sorted(periods.unique())

    rows = []
    for period in unique_periods:
        mask = (periods == period).to_numpy()
        if mask.sum() < min_samples:
            continue
        X_p = X_ood[mask]
        y_p = y_ood[mask]
        for name, predict in predictors.items():
            thr = thresholds[name]
            p = predict(X_p)
            if len(np.unique(y_p)) < 2:
                continue
            m = compute_metrics(y_p, p, thr)
            rows.append({
                "model": name,
                "period": str(period),
                "period_start": period.start_time,
                "n": int(mask.sum()),
                "default_rate": float(y_p.mean()),
                **m,
            })
        log.info("period %s | n=%d | default_rate=%.3f", period, mask.sum(), y_p.mean())

    return pd.DataFrame(rows)


def compute_drift_summary(ood_df: pd.DataFrame,
                          id_metrics: dict[str, dict]) -> pd.DataFrame:
    """Compare OOD cohort metrics to in-time test metrics.

    Args:
        ood_df: output of evaluate_over_time
        id_metrics: {model_name: {metric: value}} for in-time test set
    Returns:
        DataFrame summarising drift per model.
    """
    rows = []
    for model, grp in ood_df.groupby("model"):
        if model not in id_metrics:
            continue
        id_m = id_metrics[model]
        rows.append({
            "model": model,
            "id_auc": id_m.get("auc", float("nan")),
            "ood_auc_mean": grp["auc"].mean(),
            "ood_auc_min": grp["auc"].min(),
            "auc_drift": grp["auc"].mean() - id_m.get("auc", float("nan")),
            "id_accuracy": id_m.get("accuracy", float("nan")),
            "ood_accuracy_mean": grp["accuracy"].mean(),
            "accuracy_drift": grp["accuracy"].mean() - id_m.get("accuracy", float("nan")),
            "id_default_rate": id_m.get("default_rate", float("nan")),
            "ood_default_rate_mean": grp["default_rate"].mean(),
        })
    return pd.DataFrame(rows)
