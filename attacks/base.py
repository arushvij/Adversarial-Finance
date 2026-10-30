"""Base class and shared utilities for all tabular adversarial attacks.

Key design decisions (document in README):
  1. Attacks operate in RAW ECONOMIC UNITS (dollars, %, counts), not scaled space.
     This makes epsilon directly interpretable: eps=5 on dti means "5 percentage
     points of debt-to-income ratio", not an abstract L_inf ball.
  2. Only ACTIONABLE features are perturbed (schema.py threat model).
  3. Direction constraints are enforced: dti can only decrease, annual_inc can
     only increase, etc. This makes attacks economically plausible.
  4. Perturbed inputs are clipped to the preprocessor's winsorisation bounds, so
     they stay in-distribution.
  5. Integer features (open_acc, total_acc, etc.) are rounded after each step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]   # X -> P(default)


@dataclass
class AttackConfig:
    """Shared hyperparameters for all attacks."""
    epsilon: float = 0.10          # L_inf budget as fraction of each feature's range
    target_class: int = 0          # 0 = push toward "Fully Paid" (flip reject -> approve)
    clip_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    clip_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    actionable_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    integer_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    def feature_range(self) -> np.ndarray:
        return self.clip_upper - self.clip_lower


class TabularAttack(ABC):
    """Abstract base for FGSM / PGD."""

    def __init__(self, predict_fn: PredictFn, cfg: AttackConfig) -> None:
        self.predict_fn = predict_fn
        self.cfg = cfg

    def _project(self, x_adv: np.ndarray, x_orig: np.ndarray) -> np.ndarray:
        """Clip to box, enforce direction constraints, round integers."""
        cfg = self.cfg
        eps_abs = cfg.epsilon * cfg.feature_range()

        # L_inf ball around original
        lo = np.maximum(x_orig - eps_abs, cfg.clip_lower)
        hi = np.minimum(x_orig + eps_abs, cfg.clip_upper)

        # Direction constraints (only where actionable)
        # direction +1 -> can only increase: lo = x_orig
        # direction -1 -> can only decrease: hi = x_orig
        inc = cfg.actionable_mask & (cfg.direction == 1)
        dec = cfg.actionable_mask & (cfg.direction == -1)
        lo = np.where(inc, x_orig, lo)
        hi = np.where(dec, x_orig, hi)

        # Zero out non-actionable features
        x_adv = np.where(cfg.actionable_mask, x_adv, x_orig)
        x_adv = np.clip(x_adv, lo, hi)

        # Round integer features
        if cfg.integer_mask.any():
            x_adv = np.where(cfg.integer_mask, np.round(x_adv), x_adv)
            x_adv = np.clip(x_adv, lo, hi)
        return x_adv.astype(np.float32)

    @abstractmethod
    def perturb(self, X: np.ndarray) -> np.ndarray:
        """Return adversarial examples for a batch X of shape (n, d)."""
        ...
