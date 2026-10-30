"""Generate all publication-quality plots from saved artifacts.

Run: python -m notebooks.plot_results [--synthetic] [--out figures/]

Produces 5 figures:
  1. attack_accuracy_drop.png   — acc drop vs epsilon, FGSM vs PGD, both models
  2. robustness_comparison.png  — base vs robust MLP under PGD across epsilons
  3. certified_radius_dist.png  — certified radius CDF at sigma=0.5 and 1.0
  4. ood_auc_timeseries.png     — quarterly AUC over 2015-2016 for all models
  5. recourse_cost_dist.png     — distribution of recourse costs per model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
}
plt.rcParams.update(STYLE)

COLORS = {"xgboost": "#2196F3", "mlp": "#FF5722",
          "mlp_base": "#FF5722", "mlp_adv": "#4CAF50",
          "FGSM": "#FF9800", "PGD": "#9C27B0"}


def fig1_attack_drop(atk: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, model in zip(axes, ["xgboost", "mlp"]):
        for attack, ls in [("FGSM", "--"), ("PGD", "-")]:
            sub = atk[(atk["model"] == model) & (atk["attack"] == attack)]
            sub = sub.sort_values("epsilon")
            ax.plot(sub["epsilon"], sub["acc_drop"].abs() * 100,
                    marker="o", ls=ls, color=COLORS[attack], label=attack, linewidth=2)
        ax.set_title(f"{model.upper()} — accuracy drop under attack")
        ax.set_xlabel("Epsilon (fraction of feature range)")
        ax.set_ylabel("Accuracy drop (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend()
    fig.suptitle("Model Accuracy Degradation Under FGSM and PGD Attacks", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "attack_accuracy_drop.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/attack_accuracy_drop.png")


def fig2_robustness(rob: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [("acc_drop", "Accuracy drop (%)", True),
               ("flip_rate", "Flip rate (rejected→approved)", False)]
    for ax, (col, label, pct) in zip(axes, metrics):
        for model, color in [("mlp_base", COLORS["mlp_base"]),
                              ("mlp_adv", COLORS["mlp_adv"])]:
            sub = rob[rob["model"] == model].sort_values("epsilon")
            vals = sub[col].abs() * (100 if pct else 1)
            ax.plot(sub["epsilon"], vals, marker="o", color=color, linewidth=2,
                    label="Base MLP" if "base" in model else "Robust MLP (adv. trained)")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
    fig.suptitle("Base vs. Adversarially Trained MLP Under PGD Attack", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "robustness_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/robustness_comparison.png")


def fig3_certified(smooth: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, sigma in zip(axes, [0.5, 1.0]):
        sub = smooth[smooth["sigma"] == sigma]
        for _, row in sub.iterrows():
            model = row["model"]
            color = COLORS.get(model, "#607D8B")
            label = "Base MLP" if "base" in model else "Robust MLP"
            # Approximate CDF from the three threshold points we have
            radii = [0.0, 0.5, 1.0, 2.0]
            fracs = [0.0,
                     float(row.get("frac_radius_gt_0.5", 0) if "frac_radius_gt_0.5" in row.index else 0),
                     float(row["frac_radius_gt_1.0"]),
                     float(row["frac_radius_gt_2.0"])]
            ax.step([0] + radii, [0] + fracs, where="post",
                    color=color, linewidth=2, label=label)
        ax.axvline(float(sub["mean_certified_radius"].mean()), color="grey",
                   ls=":", alpha=0.7, label="Mean radius")
        ax.set_xlabel("Certified L2 radius")
        ax.set_ylabel("Fraction of test set")
        ax.set_title(f"Certified Robustness CDF (σ={sigma})")
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.suptitle("Randomized Smoothing: Certified Radius Distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "certified_radius_dist.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/certified_radius_dist.png")


def fig4_ood(ood_ts: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for model in ood_ts["model"].unique():
        sub = ood_ts[ood_ts["model"] == model].sort_values("period_start")
        color = COLORS.get(model, "#607D8B")
        label = {"xgboost": "XGBoost", "mlp_base": "Base MLP",
                 "mlp_adv": "Robust MLP"}.get(model, model)
        axes[0].plot(range(len(sub)), sub["auc"], marker="o",
                     color=color, label=label, linewidth=2)
        axes[1].bar(np.arange(len(sub)) + list(ood_ts["model"].unique()).index(model) * 0.25,
                    sub["default_rate"], width=0.25, color=color, alpha=0.75, label=label)

    axes[0].set_xticks(range(len(sub)))
    axes[0].set_xticklabels(sub["period"].tolist(), rotation=45, ha="right")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("OOD AUC by Quarter (2015–2016)")
    axes[0].legend()

    n = len(ood_ts[ood_ts["model"] == ood_ts["model"].iloc[0]])
    axes[1].set_xticks(np.arange(n) + 0.25)
    axes[1].set_xticklabels(sub["period"].tolist(), rotation=45, ha="right")
    axes[1].set_ylabel("Default rate")
    axes[1].set_title("OOD Default Rate by Quarter (concept drift)")
    axes[1].legend()

    fig.suptitle("Out-of-Distribution Performance: 2015–2016 Quarterly Cohorts",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "ood_auc_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/ood_auc_timeseries.png")


def fig5_recourse(rec: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    models = rec["model"].unique()
    for ax, col, xlabel in [
        (axes[0], "cost", "Recourse cost (weighted L1)"),
        (axes[1], "n_changed", "Features changed"),
    ]:
        for model in models:
            sub = rec[(rec["model"] == model) & rec["found"]].dropna(subset=[col])
            color = COLORS.get(model, "#607D8B")
            label = {"xgboost": "XGBoost", "mlp_base": "Base MLP",
                     "mlp_adv": "Robust MLP"}.get(model, model)
            ax.hist(sub[col], bins=20, alpha=0.55, color=color, label=label, density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()

    axes[0].set_title("Distribution of Minimum Recourse Cost")
    axes[1].set_title("Features Changed per Recourse")
    fig.suptitle("Algorithmic Recourse: Minimum-Cost Feature Changes for Approval",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "recourse_cost_dist.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/recourse_cost_dist.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    from common import load_config
    cfg = load_config(args.config)
    art = Path(cfg["artifacts_dir"])
    if args.synthetic:
        art = art / "synthetic"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {art} ...")
    atk   = pd.read_csv(art / "attack_results.csv")
    rob   = pd.read_csv(art / "robustness_results.csv")
    smooth = pd.read_csv(art / "smoothing_results.csv")
    ood_ts = pd.read_csv(art / "ood_timeseries.csv")
    rec   = pd.read_csv(art / "recourse_results.csv")

    print("Generating figures ...")
    fig1_attack_drop(atk, out)
    fig2_robustness(rob, out)
    fig3_certified(smooth, out)
    fig4_ood(ood_ts, out)
    fig5_recourse(rec, out)
    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
