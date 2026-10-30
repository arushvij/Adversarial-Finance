"""Randomized smoothing for certified L2 robustness (Cohen et al., 2019).

Randomized smoothing converts any base classifier f into a certifiably robust
classifier g:
    g(x) = argmax_c P(f(x + eps) = c)   where eps ~ N(0, sigma^2 I)

If g predicts class c with smoothed confidence p_A, it is GUARANTEED correct
for all inputs within L2 radius:
    R = sigma * Phi^{-1}(p_A_lower_bound)

p_A_lower_bound comes from a one-sided Clopper-Pearson (exact binomial) CI,
which gives a (1-alpha) confidence guarantee over the Monte Carlo randomness.

sigma is in RAW ECONOMIC UNITS -- sigma=1.0 means 1 unit of feature std
(e.g. ~1 pp of DTI, ~$1 of revol_bal after scaling). Choose sigma to match
the attack threat model.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from common import get_logger

log = get_logger(__name__)


@dataclass
class SmoothingResult:
    predicted_class: int        # -1 = abstain
    p_A: float                  # empirical smoothed probability
    p_A_lower: float            # Clopper-Pearson lower bound
    certified_radius: float     # L2 radius guarantee (0.0 if abstain)
    abstained: bool


class SmoothedClassifier:
    """Wraps any binary predict_proba function with randomized smoothing.

    Args:
        predict_proba_fn : X (n,d) float32 -> P(default) (n,) in [0,1]
        sigma            : Gaussian noise std in raw feature units
        n_samples        : MC samples used for certification (>=1000 recommended)
        n_samples_pred   : smaller count for fast prediction only
        alpha            : one-sided error rate for Clopper-Pearson CI
        threshold        : base classifier decision threshold
    """
    def __init__(self, predict_proba_fn, sigma: float = 1.0,
                 n_samples: int = 1000, n_samples_pred: int = 100,
                 alpha: float = 0.001, threshold: float = 0.5) -> None:
        self.predict_proba_fn = predict_proba_fn
        self.sigma = sigma
        self.n_samples = n_samples
        self.n_samples_pred = n_samples_pred
        self.alpha = alpha
        self.threshold = threshold

    def _vote(self, x: np.ndarray, n: int) -> np.ndarray:
        """Return (n,) binary votes of base classifier under Gaussian noise."""
        noise = np.random.randn(n, x.shape[0]).astype(np.float32) * self.sigma
        X_noisy = (x[None, :] + noise).astype(np.float32)
        p = self.predict_proba_fn(X_noisy)
        return (p >= self.threshold).astype(np.int32)

    def predict(self, x: np.ndarray) -> int:
        votes = self._vote(x, self.n_samples_pred)
        return int(votes.mean() >= 0.5)

    def certify(self, x: np.ndarray) -> SmoothingResult:
        """Certify a single sample. Returns abstain if majority is not confident."""
        votes = self._vote(x, self.n_samples)
        n_1 = int(votes.sum())      # votes for class 1 (default)
        n_0 = self.n_samples - n_1  # votes for class 0

        if n_1 >= n_0:
            pred_class = 1
            n_maj = n_1
        else:
            pred_class = 0
            n_maj = n_0

        p_A_emp = n_maj / self.n_samples

        # Clopper-Pearson one-sided lower bound via statsmodels (exact)
        p_A_low, _ = proportion_confint(n_maj, self.n_samples,
                                        alpha=2 * self.alpha, method="beta")

        if p_A_low <= 0.5:
            return SmoothingResult(predicted_class=-1, p_A=p_A_emp,
                                   p_A_lower=p_A_low, certified_radius=0.0,
                                   abstained=True)

        # Certified L2 radius: R = sigma * Phi^{-1}(p_A_lower)
        radius = float(self.sigma * norm.ppf(p_A_low))
        return SmoothingResult(predicted_class=pred_class, p_A=p_A_emp,
                               p_A_lower=p_A_low,
                               certified_radius=max(radius, 0.0),
                               abstained=False)

    def certify_dataset(self, X: np.ndarray,
                        batch_log_every: int = 100) -> list[SmoothingResult]:
        results = []
        for i, x in enumerate(X):
            results.append(self.certify(x))
            if (i + 1) % batch_log_every == 0:
                non_abs = [r for r in results if not r.abstained]
                log.info("  certified %d/%d | abstain=%.3f | mean_R=%.4f",
                         i + 1, len(X),
                         np.mean([r.abstained for r in results]),
                         np.mean([r.certified_radius for r in non_abs]) if non_abs else 0.0)
        return results


def summarise_certification(results: list[SmoothingResult],
                             y_true: np.ndarray | None = None) -> dict[str, float]:
    n = len(results)
    abstained = np.array([r.abstained for r in results])
    radii = np.array([r.certified_radius for r in results])
    non_abs_mask = ~abstained

    summary: dict[str, float] = {
        "n_total": float(n),
        "abstain_rate": float(abstained.mean()),
        "certified_rate": float(non_abs_mask.mean()),
        "mean_certified_radius": float(radii[non_abs_mask].mean()) if non_abs_mask.any() else 0.0,
        "median_certified_radius": float(np.median(radii[non_abs_mask])) if non_abs_mask.any() else 0.0,
        "frac_radius_gt_0.5": float((radii > 0.5).mean()),
        "frac_radius_gt_1.0": float((radii > 1.0).mean()),
        "frac_radius_gt_2.0": float((radii > 2.0).mean()),
    }
    if y_true is not None and non_abs_mask.any():
        preds = np.array([r.predicted_class for r in results])
        summary["smoothed_accuracy"] = float((preds[non_abs_mask] == y_true[non_abs_mask]).mean())
    return summary
