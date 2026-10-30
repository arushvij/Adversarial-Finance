"""Attack evaluation metrics.

These are the numbers you'll put on your resume. Every function takes
numpy arrays and returns plain Python floats or dicts.
"""
from __future__ import annotations

import numpy as np

from attacks.base import AttackConfig, PredictFn


def attack_success_rate(predict_fn: PredictFn, X_adv: np.ndarray,
                        y_true: np.ndarray, threshold: float) -> float:
    """Fraction of originally-correctly-rejected loans that flip to approved.

    "Success" = borrower was correctly predicted to default (rejected) but
    after strategic perturbation is now predicted not to default (approved).
    """
    p_orig = predict_fn(X_adv)   # using X_adv here; caller compares to X_clean
    raise NotImplementedError("use accuracy_under_attack instead")


def accuracy_under_attack(predict_fn: PredictFn, X_clean: np.ndarray,
                          X_adv: np.ndarray, y_true: np.ndarray,
                          threshold: float) -> dict[str, float]:
    """Core benchmark: clean accuracy vs. post-attack accuracy.

    Returns:
        clean_acc, adv_acc, acc_drop, flip_rate (rejected -> approved among
        true defaulters), and mean P(default) before/after.
    """
    p_clean = predict_fn(X_clean)
    p_adv = predict_fn(X_adv)
    pred_clean = (p_clean >= threshold).astype(int)
    pred_adv = (p_adv >= threshold).astype(int)

    clean_acc = float((pred_clean == y_true).mean())
    adv_acc = float((pred_adv == y_true).mean())

    # Among true defaulters who were correctly rejected, how many flip?
    true_defaulters_rejected = (y_true == 1) & (pred_clean == 1)
    if true_defaulters_rejected.sum() > 0:
        flip_rate = float(pred_adv[true_defaulters_rejected].mean() == 0 if False else
                          (pred_adv[true_defaulters_rejected] == 0).mean())
    else:
        flip_rate = float("nan")

    return {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "acc_drop": clean_acc - adv_acc,
        "flip_rate": flip_rate,
        "mean_p_clean": float(p_clean.mean()),
        "mean_p_adv": float(p_adv.mean()),
        "mean_p_drop": float((p_clean - p_adv).mean()),
    }


def perturbation_cost(X_clean: np.ndarray, X_adv: np.ndarray,
                      cfg: AttackConfig, feature_names: list[str]) -> dict[str, float]:
    """Summarise the cost of perturbations in economic units.

    Returns per-feature mean absolute change (only actionable dims) and the
    overall L_inf and L2 norms as fractions of the epsilon budget.
    """
    delta = X_adv - X_clean
    eps_abs = cfg.epsilon * cfg.feature_range()
    results: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        if cfg.actionable_mask[j]:
            results[f"mean_abs_change_{name}"] = float(np.abs(delta[:, j]).mean())
    results["l_inf_utilisation"] = float(np.abs(delta[:, cfg.actionable_mask]).max() /
                                          eps_abs[cfg.actionable_mask].max())
    results["l2_mean"] = float(np.linalg.norm(delta, axis=1).mean())
    return results


def minimum_epsilon_to_flip(predict_fn: PredictFn, x: np.ndarray,
                             cfg: AttackConfig, threshold: float,
                             n_steps: int = 40,
                             eps_grid: np.ndarray | None = None) -> float:
    """Binary search for the smallest epsilon that flips a single prediction.

    Used to compute "minimum perturbation multiplier" (resume metric).
    """
    from attacks.pgd import PGD
    if eps_grid is None:
        eps_grid = np.linspace(0.01, 0.50, 30)
    X = x.reshape(1, -1)
    p_orig = predict_fn(X)[0]
    if p_orig < threshold:   # already approved, nothing to flip
        return float("nan")
    for eps in eps_grid:
        cfg_eps = AttackConfig(
            epsilon=eps, target_class=cfg.target_class,
            clip_lower=cfg.clip_lower, clip_upper=cfg.clip_upper,
            actionable_mask=cfg.actionable_mask, direction=cfg.direction,
            integer_mask=cfg.integer_mask,
        )
        pgd = PGD(predict_fn, cfg_eps, n_steps=n_steps, step_size=eps / n_steps * 2)
        x_adv = pgd.perturb(X)
        if predict_fn(x_adv)[0] < threshold:
            return float(eps)
    return float("nan")   # could not flip within budget
