"""Fast Gradient Sign Method adapted for tabular financial data.

FGSM (Goodfellow et al., 2014) applies a single gradient step:
    x_adv = x + epsilon * sign(grad_x L(x, y))

For the financial threat model we want to *minimise* P(default), so:
    x_adv = x - epsilon * sign(grad_x P(default | x))

Applied only to actionable features, with direction and box constraints.

Works for the neural network (exact gradient via autograd) and for XGBoost
(numerical gradient via finite differences -- XGBoost has no native autograd).
"""
from __future__ import annotations

import numpy as np
import torch

from attacks.base import AttackConfig, PredictFn, TabularAttack
from models.nn_model import CreditMLP


class FGSM(TabularAttack):
    """Single-step gradient attack. Fast, weak upper bound on model brittleness."""

    def __init__(self, predict_fn: PredictFn, cfg: AttackConfig,
                 nn_model: CreditMLP | None = None, fd_eps: float = 1e-3) -> None:
        super().__init__(predict_fn, cfg)
        self.nn_model = nn_model   # if provided, use autograd; else finite differences
        self.fd_eps = fd_eps

    def _gradient(self, X: np.ndarray) -> np.ndarray:
        """d P(default) / d X, shape (n, d)."""
        if self.nn_model is not None:
            return self._autograd_gradient(X)
        return self._fd_gradient(X)

    def _autograd_gradient(self, X: np.ndarray) -> np.ndarray:
        model = self.nn_model
        model.eval()
        xt = torch.from_numpy(X.astype("float32")).requires_grad_(True)
        logits = model(xt)
        loss = torch.sigmoid(logits).sum()
        loss.backward()
        return xt.grad.detach().numpy()

    def _fd_gradient(self, X: np.ndarray) -> np.ndarray:
        """Central finite differences -- used for XGBoost."""
        n, d = X.shape
        grad = np.zeros_like(X)
        h = self.fd_eps
        for j in range(d):
            if not self.cfg.actionable_mask[j]:
                continue
            Xp, Xm = X.copy(), X.copy()
            Xp[:, j] += h; Xm[:, j] -= h
            grad[:, j] = (self.predict_fn(Xp) - self.predict_fn(Xm)) / (2 * h)
        return grad

    def perturb(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        eps_abs = cfg.epsilon * cfg.feature_range()
        grad = self._gradient(X)
        # Step toward lower P(default): subtract sign(grad)
        delta = -np.sign(grad) * eps_abs
        delta = np.where(cfg.actionable_mask, delta, 0.0)
        return self._project(X + delta, X)
