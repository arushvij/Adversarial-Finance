"""Day 1: data pipeline + XGBoost and MLP baselines.

Usage:
    python -m scripts.run_day1                       # real LendingClub data
    python -m scripts.run_day1 --synthetic           # smoke test, NOT for reporting
    python -m scripts.run_day1 --no-cache            # rebuild processed data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import get_logger, load_config, set_seed
from data.pipeline import prepare_datasets
from data.synthetic import make_synthetic_lending_club
from data.loader import load_lending_club
from models.evaluate import compute_metrics, youden_threshold
from models.nn_model import predict_proba_nn, save_mlp, train_mlp
from models.xgb_model import predict_proba_xgb, train_xgb

log = get_logger("day1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test only)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    art = Path(cfg["artifacts_dir"])
    if args.synthetic:
        art = art / "synthetic"
        cfg["data"]["processed_dir"] = str(Path(cfg["data"]["processed_dir"]) / "synthetic")
    art.mkdir(parents=True, exist_ok=True)

    raw = make_synthetic_lending_club(60_000, cfg["seed"]) if args.synthetic else None
    if raw is not None:  # mimic the loader's filtering so the same path is exercised
        tmp = art / "synthetic_raw.csv.gz"
        raw.to_csv(tmp, index=False)
        raw = load_lending_club(tmp)
    bundle = prepare_datasets(cfg, raw_df=raw, use_cache=not args.no_cache)
    bundle.preprocessor.save(art / "preprocessor.joblib")

    # ---- train ----
    xgb = train_xgb(bundle.X["train"], bundle.y["train"], bundle.X["val"], bundle.y["val"], cfg["xgboost"], cfg["seed"])
    xgb.save_model(art / "xgb.json")
    nn_model = train_mlp(bundle.X["train"], bundle.y["train"], bundle.X["val"], bundle.y["val"], cfg["nn"], cfg["seed"])
    save_mlp(nn_model, art / "mlp.pt")

    predictors = {
        "xgboost": lambda X: predict_proba_xgb(xgb, X),
        "mlp": lambda X: predict_proba_nn(nn_model, X),
    }

    # ---- evaluate (threshold frozen from validation) ----
    rows, thresholds = [], {}
    for name, predict in predictors.items():
        thr = youden_threshold(bundle.y["val"], predict(bundle.X["val"]))
        thresholds[name] = thr
        for split in ("val", "test_id", "ood"):
            m = compute_metrics(bundle.y[split], predict(bundle.X[split]), thr)
            rows.append({"model": name, "split": split, **m})
    table = pd.DataFrame(rows)
    table.to_csv(art / "baseline_metrics.csv", index=False)
    (art / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    pd.set_option("display.width", 160)
    print("\n=== Baseline results ===")
    print(table[["model", "split", "auc", "ks", "accuracy", "balanced_accuracy", "approval_rate"]].round(4).to_string(index=False))
    print(f"\nArtifacts saved to {art}/")


if __name__ == "__main__":
    main()
