"""Day 4: algorithmic recourse + OOD backtest.

Usage:
    python -m scripts.run_day4 --synthetic        # smoke test (~2 min)
    python -m scripts.run_day4                    # real data (~15 min)

Reads:  artifacts/{xgb.json, mlp.pt, mlp_adv.pt, preprocessor.joblib,
                   thresholds.json}
Writes: artifacts/{recourse_results.csv, recourse_summary.json,
                   ood_timeseries.csv, drift_summary.csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from attacks.build_attack_cfg import build_attack_config
from backtest.ood_eval import compute_drift_summary, evaluate_over_time
from common import get_logger, load_config, set_seed
from data.loader import load_lending_club
from data.pipeline import prepare_datasets
from data.synthetic import make_synthetic_lending_club
from models.evaluate import compute_metrics
from models.nn_model import load_mlp, predict_proba_nn
from models.xgb_model import predict_proba_xgb
from recourse.algorithmic_recourse import (
    DEFAULT_COST_WEIGHTS, batch_recourse, summarise_recourse,
)

log = get_logger("day4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--n-recourse", type=int, default=200,
                    help="Number of rejected applicants to compute recourse for")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    art = Path(cfg["artifacts_dir"])
    if args.synthetic:
        art = art / "synthetic"
        cfg["data"]["processed_dir"] = str(Path(cfg["data"]["processed_dir"]) / "synthetic")
    art.mkdir(parents=True, exist_ok=True)

    raw = None
    if args.synthetic:
        raw = load_lending_club(art / "synthetic_raw.csv.gz")
    bundle = prepare_datasets(cfg, raw_df=raw, use_cache=not args.no_cache)

    thresholds_raw = json.loads((art / "thresholds.json").read_text())
    thresholds = {
        "xgboost": thresholds_raw["xgboost"],
        "mlp_base": thresholds_raw["mlp"],
        "mlp_adv":  thresholds_raw.get("mlp_adv", thresholds_raw["mlp"]),
    }
    xgb = XGBClassifier(); xgb.load_model(art / "xgb.json")
    mlp_base = load_mlp(art / "mlp.pt")
    mlp_adv  = load_mlp(art / "mlp_adv.pt")

    predictors = {
        "xgboost":  lambda X: predict_proba_xgb(xgb, X),
        "mlp_base": lambda X: predict_proba_nn(mlp_base, X),  # type: ignore
        "mlp_adv":  lambda X: predict_proba_nn(mlp_adv, X),
    }
    feature_names = bundle.feature_names
    attack_cfg = build_attack_config(bundle, epsilon=0.10)

    # ================================================================
    # PART 1: Algorithmic recourse on TEST_ID rejected applicants
    # ================================================================
    log.info("=== Algorithmic Recourse ===")
    rng = np.random.default_rng(cfg["seed"])

    all_recourse_rows = []
    recourse_summaries = {}

    for model_name, predict_fn in predictors.items():
        thr = thresholds[model_name]
        p_test = predict_fn(bundle.X["test_id"])
        rejected_idx = np.where(
            (p_test >= thr) & (bundle.y["test_id"] == 1)  # true defaulters, correctly rejected
        )[0]
        sample_idx = rng.choice(rejected_idx,
                                size=min(args.n_recourse, len(rejected_idx)),
                                replace=False)
        X_rej = bundle.X["test_id"][sample_idx]

        log.info("[%s] computing recourse for %d rejected applicants ...",
                 model_name, len(X_rej))
        results = batch_recourse(X_rej, predict_fn, attack_cfg,
                                 feature_names, thr, DEFAULT_COST_WEIGHTS)
        summary = summarise_recourse(results, feature_names)
        recourse_summaries[model_name] = summary

        log.info("[%s] recourse_rate=%.3f | mean_cost=%.4f | mean_features_changed=%.1f",
                 model_name, summary["recourse_rate"], summary["mean_cost"],
                 summary["mean_n_features_changed"])
        log.info("[%s] top changed features: %s", model_name, summary["top_changed_features"])
        log.info("[%s] mean deltas: %s", model_name,
                 {k: f"{v:+.2f}" for k, v in summary["mean_delta_per_feature"].items()})

        for r in results:
            row = {"model": model_name, "found": r.found, "cost": r.cost,
                   "n_changed": r.n_changed_features,
                   "p_orig": r.p_orig, "p_recourse": r.p_recourse}
            row.update({f"delta_{k}": v for k, v in r.changed_features.items()})
            all_recourse_rows.append(row)

    rec_df = pd.DataFrame(all_recourse_rows)
    rec_df.to_csv(art / "recourse_results.csv", index=False)
    (art / "recourse_summary.json").write_text(
        json.dumps(recourse_summaries, indent=2, default=str))

    # ================================================================
    # PART 2: OOD backtest — performance drift over time
    # ================================================================
    log.info("=== OOD Backtest ===")
    freq = "Q"
    ood_df = evaluate_over_time(
        bundle.X["ood"], bundle.y["ood"], bundle.issue_d["ood"],
        predictors, thresholds, freq=freq,
    )
    ood_df.to_csv(art / "ood_timeseries.csv", index=False)

    # In-time metrics for drift comparison
    id_metrics = {}
    for name, predict in predictors.items():
        thr = thresholds[name]
        p = predict(bundle.X["test_id"])
        m = compute_metrics(bundle.y["test_id"], p, thr)
        m["default_rate"] = float(bundle.y["test_id"].mean())
        id_metrics[name] = m

    drift_df = compute_drift_summary(ood_df, id_metrics)
    drift_df.to_csv(art / "drift_summary.csv", index=False)

    # ================================================================
    # PRINT SUMMARY
    # ================================================================
    print("\n=== Recourse Summary ===")
    rec_rows = []
    for model_name, s in recourse_summaries.items():
        rec_rows.append({
            "model": model_name,
            "recourse_rate": round(s["recourse_rate"], 3),
            "mean_cost": round(s["mean_cost"], 4),
            "mean_features_changed": round(s["mean_n_features_changed"], 1),
            "top_features": ", ".join(s["top_changed_features"][:3]),
        })
    print(pd.DataFrame(rec_rows).to_string(index=False))

    print("\n=== OOD Drift Summary ===")
    print(drift_df[["model", "id_auc", "ood_auc_mean", "auc_drift",
                     "id_accuracy", "ood_accuracy_mean", "accuracy_drift"]
                   ].round(4).to_string(index=False))

    print("\n=== OOD AUC by Quarter ===")
    pivot = ood_df.pivot_table(index="period", columns="model", values="auc").round(4)
    print(pivot.to_string())
    print(f"\nArtifacts saved to {art}/")


if __name__ == "__main__":
    main()
