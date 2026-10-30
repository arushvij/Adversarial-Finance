"""Day 2: adversarial attacks (FGSM + PGD) on both models.

Usage:
    python -m scripts.run_day2 --synthetic        # smoke test
    python -m scripts.run_day2                    # real data (run after Day 1)

Reads artifacts produced by run_day1.py (xgb.json, mlp.pt, preprocessor.joblib,
thresholds.json). Writes artifacts/attack_results.csv with the core benchmark
numbers for your resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from attacks.base import AttackConfig
from attacks.build_attack_cfg import build_attack_config
from attacks.fgsm import FGSM
from attacks.metrics import accuracy_under_attack, perturbation_cost
from attacks.pgd import PGD
from common import get_logger, load_config, set_seed
from data.loader import load_lending_club
from data.pipeline import prepare_datasets
from data.synthetic import make_synthetic_lending_club
from models.nn_model import CreditMLP, load_mlp, predict_proba_nn
from models.xgb_model import predict_proba_xgb

log = get_logger("day2")

EPSILONS = [0.05, 0.10, 0.20]   # L_inf budgets as fraction of feature range


def run_attacks(name: str, predict_fn, X_test: np.ndarray, y_test: np.ndarray,
                threshold: float, attack_cfg: AttackConfig,
                nn_model: CreditMLP | None = None) -> list[dict]:
    rows = []
    for eps in EPSILONS:
        cfg = AttackConfig(epsilon=eps, clip_lower=attack_cfg.clip_lower,
                           clip_upper=attack_cfg.clip_upper,
                           actionable_mask=attack_cfg.actionable_mask,
                           direction=attack_cfg.direction,
                           integer_mask=attack_cfg.integer_mask)

        # --- FGSM ---
        fgsm = FGSM(predict_fn, cfg, nn_model=nn_model)
        X_fgsm = fgsm.perturb(X_test)
        m_fgsm = accuracy_under_attack(predict_fn, X_test, X_fgsm, y_test, threshold)
        rows.append({"model": name, "attack": "FGSM", "epsilon": eps, **m_fgsm})
        log.info("[%s | FGSM | eps=%.2f] acc_drop=%.4f  flip_rate=%.4f",
                 name, eps, m_fgsm["acc_drop"], m_fgsm["flip_rate"])

        # --- PGD ---
        pgd = PGD(predict_fn, cfg, n_steps=40, step_size=eps / 20,
                  n_restarts=2, nn_model=nn_model)
        X_pgd = pgd.perturb(X_test)
        m_pgd = accuracy_under_attack(predict_fn, X_test, X_pgd, y_test, threshold)
        rows.append({"model": name, "attack": "PGD", "epsilon": eps, **m_pgd})
        log.info("[%s | PGD  | eps=%.2f] acc_drop=%.4f  flip_rate=%.4f",
                 name, eps, m_pgd["acc_drop"], m_pgd["flip_rate"])

        # Perturbation cost breakdown (log for eps=0.10 only)
        if abs(eps - 0.10) < 1e-9:
            cost = perturbation_cost(X_test, X_pgd, cfg, predict_fn.__self__.feature_names_ if hasattr(predict_fn, '__self__') else [])
            log.info("  PGD eps=0.10 L_inf_util=%.3f  L2_mean=%.3f",
                     cost.get("l_inf_utilisation", float("nan")),
                     cost.get("l2_mean", float("nan")))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--n-test", type=int, default=2000,
                    help="Subsample of test_id for speed (PGD+XGB is slow)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    art = Path(cfg["artifacts_dir"])
    if args.synthetic:
        art = art / "synthetic"
        cfg["data"]["processed_dir"] = str(Path(cfg["data"]["processed_dir"]) / "synthetic")
    art.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    raw = None
    if args.synthetic:
        raw = load_lending_club(art / "synthetic_raw.csv.gz")
    bundle = prepare_datasets(cfg, raw_df=raw, use_cache=not args.no_cache)

    # ---- load Day 1 artifacts ----
    from data.preprocess import TabularPreprocessor
    pre = TabularPreprocessor.load(art / "preprocessor.joblib")
    thresholds = json.loads((art / "thresholds.json").read_text())

    xgb = XGBClassifier()
    xgb.load_model(art / "xgb.json")
    nn_model = load_mlp(art / "mlp.pt")

    # Subsample test set for speed
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(bundle.X["test_id"]), size=min(args.n_test, len(bundle.X["test_id"])), replace=False)
    X_test = bundle.X["test_id"][idx]
    y_test = bundle.y["test_id"][idx]

    attack_cfg = build_attack_config(bundle)
    log.info("Actionable features: %s", [n for n, m in zip(pre.feature_names_, attack_cfg.actionable_mask) if m])

    all_rows: list[dict] = []

    # ---- XGBoost attacks (finite-difference gradients) ----
    log.info("=== XGBoost attacks ===")
    xgb_predict = lambda X: predict_proba_xgb(xgb, X)
    all_rows += run_attacks("xgboost", xgb_predict, X_test, y_test,
                            thresholds["xgboost"], attack_cfg)

    # ---- MLP attacks (autograd) ----
    log.info("=== MLP attacks ===")
    mlp_predict = lambda X: predict_proba_nn(nn_model, X)
    all_rows += run_attacks("mlp", mlp_predict, X_test, y_test,
                            thresholds["mlp"], attack_cfg, nn_model=nn_model)

    # ---- Save and print ----
    results = pd.DataFrame(all_rows)
    results.to_csv(art / "attack_results.csv", index=False)

    print("\n=== Attack Results (eps=0.10) ===")
    r10 = results[results["epsilon"] == 0.10].copy()
    print(r10[["model", "attack", "clean_acc", "adv_acc", "acc_drop", "flip_rate",
               "mean_p_clean", "mean_p_adv"]].round(4).to_string(index=False))

    print("\n=== Accuracy drop across epsilons ===")
    pivot = results.pivot_table(index=["model", "attack"], columns="epsilon", values="acc_drop").round(4)
    print(pivot)
    print(f"\nFull results saved to {art}/attack_results.csv")


if __name__ == "__main__":
    main()
