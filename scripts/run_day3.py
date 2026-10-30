"""Day 3: adversarial training + randomized smoothing certification.

Usage:
    python -m scripts.run_day3 --synthetic        # smoke test (~3 min)
    python -m scripts.run_day3                    # real data (~25-40 min)
    python -m scripts.run_day3 --skip-adv-train   # if adv model already saved

Reads:  artifacts/{xgb.json, mlp.pt, preprocessor.joblib, thresholds.json,
                   attack_results.csv}
Writes: artifacts/{mlp_adv.pt, robustness_results.csv, smoothing_results.csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from attacks.build_attack_cfg import build_attack_config
from attacks.pgd import PGD
from attacks.metrics import accuracy_under_attack
from common import get_logger, load_config, set_seed
from data.loader import load_lending_club
from data.pipeline import prepare_datasets
from data.preprocess import TabularPreprocessor
from data.synthetic import make_synthetic_lending_club
from models.evaluate import compute_metrics, youden_threshold
from models.nn_model import load_mlp, predict_proba_nn, save_mlp
from models.xgb_model import predict_proba_xgb
from robustness.adversarial_training import train_mlp_adversarial
from robustness.smoothing import SmoothedClassifier, summarise_certification

log = get_logger("day3")

ADV_PARAMS = {
    "pgd_steps": 7,
    "pgd_step_size": 0.007,
    "mix_alpha": 0.8,        # 80% adversarial, 20% clean -> preserves clean acc
    "max_epochs": 30,
    "patience": 6,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,
}

SIGMA_VALUES = [0.5, 1.0, 2.0]   # noise levels for smoothing sweep
N_CERTIFY = 500                    # samples to certify (speed vs precision)
N_SMOOTH_SAMPLES = 500             # Monte Carlo samples per certification


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip-adv-train", action="store_true",
                    help="Load mlp_adv.pt if it exists (skip retraining)")
    ap.add_argument("--n-certify", type=int, default=N_CERTIFY)
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
    thresholds = json.loads((art / "thresholds.json").read_text())
    xgb = XGBClassifier(); xgb.load_model(art / "xgb.json")
    mlp_base = load_mlp(art / "mlp.pt")
    attack_cfg = build_attack_config(bundle, epsilon=0.10)

    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(bundle.X["test_id"]),
                     size=min(2000, len(bundle.X["test_id"])), replace=False)
    X_test = bundle.X["test_id"][idx]
    y_test = bundle.y["test_id"][idx]

    # ================================================================
    # PART 1: Adversarial training
    # ================================================================
    adv_path = art / "mlp_adv.pt"
    if args.skip_adv_train and adv_path.exists():
        log.info("Loading existing adversarially trained model from %s", adv_path)
        mlp_adv = load_mlp(adv_path)
    else:
        log.info("=== Adversarial training (MLP) ===")
        mlp_adv = train_mlp_adversarial(
            bundle.X["train"], bundle.y["train"],
            bundle.X["val"], bundle.y["val"],
            mlp_base, attack_cfg, cfg["nn"], ADV_PARAMS, cfg["seed"],
        )
        save_mlp(mlp_adv, adv_path)
        log.info("Saved adversarially trained model -> %s", adv_path)

    # Freeze threshold for robust model on validation
    thr_adv = youden_threshold(bundle.y["val"], predict_proba_nn(mlp_adv, bundle.X["val"]))
    thresholds["mlp_adv"] = thr_adv
    (art / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # ================================================================
    # PART 2: Compare base vs robust model under PGD
    # ================================================================
    log.info("=== Evaluating base vs robust MLP under PGD ===")
    rows = []
    for eps in [0.05, 0.10, 0.20]:
        from attacks.base import AttackConfig
        cfg_eps = AttackConfig(epsilon=eps, clip_lower=attack_cfg.clip_lower,
                               clip_upper=attack_cfg.clip_upper,
                               actionable_mask=attack_cfg.actionable_mask,
                               direction=attack_cfg.direction,
                               integer_mask=attack_cfg.integer_mask)
        pgd = PGD(lambda X: predict_proba_nn(mlp_adv, X), cfg_eps,
                  n_steps=40, step_size=eps / 20, n_restarts=1, nn_model=mlp_adv)
        X_pgd = pgd.perturb(X_test)

        # Base model metrics
        m_base = accuracy_under_attack(
            lambda X: predict_proba_nn(mlp_base, X), X_test, X_pgd, y_test, thresholds["mlp"])
        m_adv = accuracy_under_attack(
            lambda X: predict_proba_nn(mlp_adv, X), X_test, X_pgd, y_test, thr_adv)

        # Clean accuracy for both
        base_clean = compute_metrics(y_test, predict_proba_nn(mlp_base, X_test), thresholds["mlp"])
        adv_clean = compute_metrics(y_test, predict_proba_nn(mlp_adv, X_test), thr_adv)

        rows.append({
            "epsilon": eps, "model": "mlp_base",
            "clean_acc": base_clean["accuracy"], "adv_acc": m_base["adv_acc"],
            "acc_drop": m_base["acc_drop"], "flip_rate": m_base["flip_rate"],
        })
        rows.append({
            "epsilon": eps, "model": "mlp_adv",
            "clean_acc": adv_clean["accuracy"], "adv_acc": m_adv["adv_acc"],
            "acc_drop": m_adv["acc_drop"], "flip_rate": m_adv["flip_rate"],
        })
        log.info("eps=%.2f | base acc_drop=%.4f flip=%.4f | adv acc_drop=%.4f flip=%.4f",
                 eps, m_base["acc_drop"], m_base["flip_rate"],
                 m_adv["acc_drop"], m_adv["flip_rate"])

    rob_df = pd.DataFrame(rows)
    rob_df.to_csv(art / "robustness_results.csv", index=False)

    # ================================================================
    # PART 3: Randomized smoothing certification
    # ================================================================
    log.info("=== Randomized smoothing certification ===")
    idx_cert = rng.choice(len(bundle.X["test_id"]),
                          size=min(args.n_certify, len(bundle.X["test_id"])), replace=False)
    X_cert = bundle.X["test_id"][idx_cert]
    y_cert = bundle.y["test_id"][idx_cert]

    smooth_rows = []
    for sigma in SIGMA_VALUES:
        for model_name, model, thr in [("mlp_base", mlp_base, thresholds["mlp"]),
                                        ("mlp_adv", mlp_adv, thr_adv)]:
            log.info("Certifying %s sigma=%.1f ...", model_name, sigma)
            sc = SmoothedClassifier(
                predict_proba_fn=lambda X, m=model: predict_proba_nn(m, X),
                sigma=sigma, n_samples=N_SMOOTH_SAMPLES,
                n_samples_pred=100, alpha=0.001, threshold=thr,
            )
            results = sc.certify_dataset(X_cert, batch_log_every=100)
            summary = summarise_certification(results, y_cert)
            smooth_rows.append({"model": model_name, "sigma": sigma, **summary})
            log.info("  sigma=%.1f | certified_rate=%.3f | mean_radius=%.3f | "
                     "frac_r>1.0=%.3f | smoothed_acc=%.3f",
                     sigma, summary["certified_rate"], summary["mean_certified_radius"],
                     summary["frac_radius_gt_1.0"], summary.get("smoothed_accuracy", float("nan")))

    smooth_df = pd.DataFrame(smooth_rows)
    smooth_df.to_csv(art / "smoothing_results.csv", index=False)

    # ================================================================
    # PRINT SUMMARY
    # ================================================================
    print("\n=== Robustness: Base vs Adversarially Trained MLP (under PGD) ===")
    pivot = rob_df.pivot_table(index=["model", "epsilon"], values=["clean_acc", "adv_acc", "acc_drop", "flip_rate"])

    print(pivot.round(4).to_string())

    print("\n=== Certified Robustness (Randomized Smoothing) ===")
    cols = ["model", "sigma", "certified_rate", "mean_certified_radius",
            "frac_radius_gt_1.0", "smoothed_accuracy"]
    print(smooth_df[[c for c in cols if c in smooth_df.columns]].round(4).to_string(index=False))
    print(f"\nArtifacts saved to {art}/")


if __name__ == "__main__":
    main()
