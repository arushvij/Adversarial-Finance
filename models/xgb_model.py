"""XGBoost credit-default model."""
from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              params: dict[str, Any], seed: int = 42) -> XGBClassifier:
    """Train with early stopping on validation AUC."""
    model = XGBClassifier(
        n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
        max_depth=params["max_depth"], subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"], min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"], early_stopping_rounds=params["early_stopping_rounds"],
        eval_metric="auc", tree_method="hist", random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_proba_xgb(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """P(default) as a 1-D float array."""
    return model.predict_proba(X)[:, 1]
