"""Minimum-cost algorithmic recourse for rejected borrowers.

Given a rejected applicant x (P(default) >= threshold), find the smallest
actionable perturbation x' such that P(default | x') < threshold.

IMPLEMENTATION: Coordinate-descent + greedy combination.
We use coordinate descent rather than gradient descent because:
  - XGBoost is piecewise-constant; gradients are 0 almost everywhere
  - The economic action space is discrete in practice (pay off a card, ask for
    less loan) -- coordinate moves are more realistic than smooth gradient steps
  - For the NN, gradient steps are also supported via the PGD attack module

Algorithm:
  1. Try perturbing each actionable feature individually (coordinate descent)
  2. Greedily combine the cheapest moves that together flip the decision
  3. Return the minimum-cost combination found within epsilon_max budget

Cost function: cost(x, x') = sum_j w_j * |x'_j - x_j| / range_j
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from attacks.base import AttackConfig, PredictFn
from common import get_logger

log = get_logger(__name__)

DEFAULT_COST_WEIGHTS: dict[str, float] = {
    "loan_amnt":   0.3,
    "annual_inc":  1.0,
    "dti":         0.4,
    "revol_util":  0.4,
    "revol_bal":   0.5,
    "open_acc":    0.6,
    "total_acc":   0.8,
}

N_STEPS = 20   # number of epsilon values to try per feature


@dataclass
class RecourseResult:
    x_orig: np.ndarray
    x_recourse: np.ndarray
    found: bool
    cost: float
    n_changed_features: int
    changed_features: dict[str, float]   # name -> delta in raw units
    p_orig: float
    p_recourse: float


def _build_cost_weights(feature_names: list[str],
                        cost_weights: dict[str, float]) -> np.ndarray:
    return np.array([cost_weights.get(n, 0.0) for n in feature_names], dtype=np.float32)


def _clip_to_bounds(x_new: np.ndarray, x_orig: np.ndarray,
                    cfg: AttackConfig, eps: float) -> np.ndarray:
    feat_range = cfg.feature_range()
    eps_abs = eps * feat_range
    lo = np.maximum(x_orig - eps_abs, cfg.clip_lower)
    hi = np.minimum(x_orig + eps_abs, cfg.clip_upper)
    inc = cfg.actionable_mask & (cfg.direction == 1)
    dec = cfg.actionable_mask & (cfg.direction == -1)
    lo = np.where(inc, x_orig, lo)
    hi = np.where(dec, x_orig, hi)
    x_new = np.where(cfg.actionable_mask, x_new, x_orig)
    x_new = np.clip(x_new, lo, hi)
    if cfg.integer_mask.any():
        x_new = np.where(cfg.integer_mask, np.round(x_new), x_new)
        x_new = np.clip(x_new, lo, hi)
    return x_new.astype(np.float32)


def _weighted_cost(x_orig: np.ndarray, x_rec: np.ndarray,
                   weights: np.ndarray, feat_range: np.ndarray,
                   actionable: np.ndarray) -> float:
    delta = np.abs(x_rec - x_orig)
    safe_range = np.where(feat_range > 0, feat_range, 1.0)
    return float((weights * (delta / safe_range) * actionable).sum())


def compute_recourse(
    x: np.ndarray,
    predict_fn: PredictFn,
    attack_cfg: AttackConfig,
    feature_names: list[str],
    threshold: float,
    cost_weights: dict[str, float] | None = None,
    epsilon_max: float = 0.50,
) -> RecourseResult:
    """Find minimum-cost recourse via coordinate descent + greedy combination."""
    if cost_weights is None:
        cost_weights = DEFAULT_COST_WEIGHTS

    w = _build_cost_weights(feature_names, cost_weights)
    feat_range = attack_cfg.feature_range()
    act_idx = np.where(attack_cfg.actionable_mask)[0]
    p_orig = float(predict_fn(x[None, :])[0])

    if p_orig < threshold:
        return RecourseResult(x, x.copy(), True, 0.0, 0, {}, p_orig, p_orig)

    # ------------------------------------------------------------------
    # Step 1: per-feature sweep — find cheapest single-feature recourse
    # ------------------------------------------------------------------
    single_moves: list[tuple[float, np.ndarray]] = []   # (cost, x_candidate)

    for j in act_idx:
        dir_j = attack_cfg.direction[j]
        lo_j = attack_cfg.clip_lower[j]
        hi_j = attack_cfg.clip_upper[j]
        range_j = max(feat_range[j], 1e-6)

        # Try epsilon from small to epsilon_max
        for eps in np.linspace(0.02, epsilon_max, N_STEPS):
            delta = eps * range_j
            if dir_j == 1:      # can only increase
                x_new_j = min(x[j] + delta, hi_j)
            elif dir_j == -1:   # can only decrease
                x_new_j = max(x[j] - delta, lo_j)
            else:               # both: try decrease first (reduces risk)
                x_new_j = max(x[j] - delta, lo_j)

            x_cand = x.copy()
            x_cand[j] = round(x_new_j) if attack_cfg.integer_mask[j] else x_new_j
            x_cand = _clip_to_bounds(x_cand, x, attack_cfg, epsilon_max)
            p_cand = predict_fn(x_cand[None, :])[0]

            if p_cand < threshold:
                cost = _weighted_cost(x, x_cand, w, feat_range,
                                      attack_cfg.actionable_mask)
                single_moves.append((cost, x_cand.copy()))
                break   # cheapest for this feature found

    # ------------------------------------------------------------------
    # Step 2: if single-feature not enough, try greedy combinations (pairs)
    # ------------------------------------------------------------------
    best_x = x.copy()
    best_cost = float("inf")
    best_found = False

    # Check all single-feature moves
    for cost, x_cand in single_moves:
        if cost < best_cost:
            best_cost = cost
            best_x = x_cand
            best_found = True

    # If nothing single-feature worked, try pairs
    if not best_found and len(act_idx) >= 2:
        for j1, j2 in combinations(act_idx, 2):
            x_cand = x.copy()
            for j in (j1, j2):
                dir_j = attack_cfg.direction[j]
                delta = epsilon_max * max(feat_range[j], 1e-6)
                if dir_j == 1:    x_cand[j] = min(x[j] + delta, attack_cfg.clip_upper[j])
                elif dir_j == -1: x_cand[j] = max(x[j] - delta, attack_cfg.clip_lower[j])
                else:             x_cand[j] = max(x[j] - delta, attack_cfg.clip_lower[j])
                if attack_cfg.integer_mask[j]:
                    x_cand[j] = round(x_cand[j])
            x_cand = _clip_to_bounds(x_cand, x, attack_cfg, epsilon_max)
            p_cand = predict_fn(x_cand[None, :])[0]
            if p_cand < threshold:
                cost = _weighted_cost(x, x_cand, w, feat_range,
                                      attack_cfg.actionable_mask)
                if cost < best_cost:
                    best_cost, best_x, best_found = cost, x_cand.copy(), True

    p_rec = float(predict_fn(best_x[None, :])[0])
    delta = best_x - x
    changed = {
        feature_names[j]: float(delta[j])
        for j in np.where(attack_cfg.actionable_mask & (np.abs(delta) > 1e-4))[0]
    }
    return RecourseResult(
        x_orig=x, x_recourse=best_x, found=best_found,
        cost=best_cost if best_found else float("nan"),
        n_changed_features=len(changed),
        changed_features=changed,
        p_orig=p_orig, p_recourse=p_rec,
    )


def batch_recourse(
    X_rejected: np.ndarray,
    predict_fn: PredictFn,
    attack_cfg: AttackConfig,
    feature_names: list[str],
    threshold: float,
    cost_weights: dict[str, float] | None = None,
) -> list[RecourseResult]:
    results = []
    for i, x in enumerate(X_rejected):
        r = compute_recourse(x, predict_fn, attack_cfg, feature_names,
                             threshold, cost_weights)
        results.append(r)
        if (i + 1) % 50 == 0:
            found = sum(rr.found for rr in results)
            costs = [rr.cost for rr in results if rr.found]
            log.info("  recourse %d/%d | found=%d | mean_cost=%.4f",
                     i + 1, len(X_rejected), found,
                     float(np.mean(costs)) if costs else float("nan"))
    return results


def summarise_recourse(results: list[RecourseResult],
                       feature_names: list[str]) -> dict:
    found = [r for r in results if r.found]
    costs = [r.cost for r in found]
    n_changed = [r.n_changed_features for r in found]
    p_drops = [r.p_orig - r.p_recourse for r in found]

    feat_counts: dict[str, int] = {}
    feat_deltas: dict[str, list[float]] = {}
    for r in found:
        for name, delta in r.changed_features.items():
            feat_counts[name] = feat_counts.get(name, 0) + 1
            feat_deltas.setdefault(name, []).append(delta)

    return {
        "n_total": len(results),
        "n_found": len(found),
        "recourse_rate": len(found) / len(results) if results else 0.0,
        "mean_cost": float(np.mean(costs)) if costs else float("nan"),
        "median_cost": float(np.median(costs)) if costs else float("nan"),
        "mean_n_features_changed": float(np.mean(n_changed)) if n_changed else float("nan"),
        "mean_p_drop": float(np.mean(p_drops)) if p_drops else float("nan"),
        "top_changed_features": sorted(feat_counts, key=feat_counts.get, reverse=True)[:5],
        "mean_delta_per_feature": {k: float(np.mean(v)) for k, v in feat_deltas.items()},
    }
