"""Projected Gradient Descent (Madry et al., 2018) for tabular financial data.

PGD is the gold-standard iterative attack. Each step:
    x^{t+1} = Project(x^t - alpha * sign(grad_x P(default | x^t)))
where Project clips to the epsilon-ball, direction constraints, and box.

For XGBoost we use finite-difference gradients (slow for large n; use a
subsample). For the NN we use autograd (fast).

The attack finds the *minimum perturbation needed to flip a decision*, which
is the "strategic recourse cost" metric you'll cite on your resume.
"""
from __future__ import annotations

import numpy as np
import torch

from attacks.base import AttackConfig, PredictFn, TabularAttack
from models.nn_model import CreditMLP


class PGD(TabularAttack):
    """Multi-step projected gradient descent attack."""

    def __init__(self, predict_fn: PredictFn, cfg: AttackConfig,
                 n_steps: int = 40, step_size: float = 0.02,
                 n_restarts: int = 1, nn_model: CreditMLP | None = None,
                 fd_eps: float = 1e-3) -> None:
        super().__init__(predict_fn, cfg)
        self.n_steps = n_steps
        self.step_size = step_size       # fraction of feature range per step
        self.n_restarts = n_restarts
        self.nn_model = nn_model
        self.fd_eps = fd_eps

    def _gradient_batch(self, X: np.ndarray) -> np.ndarray:
        if self.nn_model is not None:
            model = self.nn_model
            model.eval()
            xt = torch.from_numpy(X.astype("float32")).requires_grad_(True)
            torch.sigmoid(model(xt)).sum().backward()
            return xt.grad.detach().numpy()
        # Finite differences for XGBoost (only actionable dims)
        n, d = X.shape
        grad = np.zeros_like(X)
        h = self.fd_eps
        for j in np.where(self.cfg.actionable_mask)[0]:
            Xp, Xm = X.copy(), X.copy()
            Xp[:, j] += h; Xm[:, j] -= h
            grad[:, j] = (self.predict_fn(Xp) - self.predict_fn(Xm)) / (2 * h)
        return grad

    def _run_one(self, X: np.ndarray, x_start: np.ndarray) -> np.ndarray:
        """One PGD trajectory from x_start."""
        x_adv = x_start.copy()
        alpha = self.step_size * self.cfg.feature_range()
        for _ in range(self.n_steps):
            grad = self._gradient_batch(x_adv)
            x_adv = x_adv - alpha * np.sign(grad)   # minimise P(default)
            x_adv = self._project(x_adv, X)
        return x_adv

    def perturb(self, X: np.ndarray) -> np.ndarray:
        """Return best (lowest P(default)) adversarial example over restarts."""
        best_adv = self._run_one(X, X.copy())
        best_p = self.predict_fn(best_adv)
        for _ in range(self.n_restarts - 1):
            # Random start within the epsilon-ball
            eps_abs = self.cfg.epsilon * self.cfg.feature_range()
            noise = np.random.uniform(-1, 1, X.shape) * eps_abs * self.cfg.actionable_mask
            x_start = self._project(X + noise, X)
            x_adv = self._run_one(X, x_start)
            p = self.predict_fn(x_adv)
            better = p < best_p
            best_adv[better] = x_adv[better]
            best_p[better] = p[better]
        return best_adv
