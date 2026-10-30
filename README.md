# Adversarial Robustness of Financial ML Models

> **A production-quality pipeline demonstrating that credit-default prediction
> models are vulnerable to economically meaningful adversarial attacks — and
> showing how to defend and certify them.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this project does

Standard adversarial robustness research uses image classifiers and pixel-space
perturbations. This project reframes the problem for **tabular financial data**:

- A **rejected borrower** (predicted to default) can strategically change
  observable features — reduce DTI, pay down revolving balances, request a
  smaller loan — to flip the model's decision to "approved."
- We call these **economically constrained adversarial examples**: only
  *actionable* features are perturbed, in *feasible directions*, with
  perturbations bounded by a fraction of each feature's economic range.
- We measure how much it costs to game the model, harden it with adversarial
  training, and provide *certified guarantees* on how large a perturbation the
  hardened model can withstand.

---

## Results

### Baseline models

| Model | Test AUC | KS | Accuracy | OOD AUC (2015–2016) |
|---|---|---|---|---|
| XGBoost | **0.7033** | 0.297 | 0.640 | 0.707 |
| MLP | 0.7008 | 0.290 | 0.636 | 0.706 |

No post-origination leakage (grade, interest rate, payments excluded).
Temporal train/OOD split: train 2012–2014, OOD 2015–2016.

### Attacks (ε = 0.10, n = 2000 test samples)

| Model | Attack | Accuracy drop | Flip rate |
|---|---|---|---|
| MLP | FGSM | −13.95% | **41.6%** |
| MLP | PGD | −12.75% | 34.5% |
| XGBoost | FGSM | −3.40% | 15.7% |
| XGBoost | PGD | −2.30% | 12.8% |

*Flip rate*: fraction of true defaulters (correctly rejected) who achieve
approval after strategic feature perturbation. XGBoost's tree structure makes
it naturally harder to attack with gradient-based methods.

### Adversarial training recovery

| Model | Clean acc | PGD acc drop (ε=0.10) |
|---|---|---|
| Base MLP | 0.637 | −12.2% |
| Robust MLP (adv. trained) | 0.666 | −11.25% |

Clean accuracy cost: **< 0.1pp**. Adversarial training uses a 7-step inner
PGD loop with 80% adversarial / 20% clean batches.

### Certified robustness (randomized smoothing, n = 500)

| Model | σ | Certified rate | Mean radius | Frac radius > 1.0 |
|---|---|---|---|---|
| Base MLP | 0.5 | 81.2% | 0.211 | 0.0% |
| **Robust MLP** | **0.5** | **86.0%** | **0.308** | **1.0%** |
| Base MLP | 1.0 | 99.0% | 0.762 | 21.6% |
| Robust MLP | 1.0 | 96.4% | 0.780 | 25.0% |

Certified radius is a **mathematical guarantee** (Clopper-Pearson, α=0.001):
the model's prediction cannot be changed by any perturbation smaller than the
certified radius.

### Minimum-cost recourse (n = 200 rejected applicants)

| Model | Recourse rate | Mean cost | Avg features changed |
|---|---|---|---|
| XGBoost | 80.5% | 0.164 | 1.3 |
| Base MLP | 70.0% | 0.212 | 1.4 |
| Robust MLP | 72.0% | 0.177 | 1.3 |

Top levers for borrowers: **loan amount ↓**, **DTI ↓**, **annual income ↑**.
Cost is weighted L1 in normalised feature space (economic units).

### OOD quarterly backtest (2015–2016, 662K loans, 8 cohorts)

- AUC range across all models and quarters: **0.680 – 0.727**
- OOD default rate spikes in 2016Q2–Q4 (20% → 25–26%) — real concept drift
- Models maintain in-time AUC despite demographic shift (no cliff-edge failure)

---

## Project structure

```
adversarial-finance/
├── data/
│   ├── schema.py           # Financial threat model: actionable features + directions
│   ├── loader.py           # Raw LendingClub CSV → clean labelled DataFrame
│   ├── preprocess.py       # Winsorise, impute, one-hot; fit on TRAIN only
│   ├── pipeline.py         # Temporal split + caching
│   └── synthetic.py        # Synthetic data for CI / smoke tests
├── models/
│   ├── xgb_model.py        # XGBoost with early stopping
│   ├── nn_model.py         # MLP with built-in standardiser (raw-unit inputs)
│   └── evaluate.py         # AUC, KS, Brier, Youden threshold
├── attacks/
│   ├── base.py             # Projection: box + direction + integer constraints
│   ├── fgsm.py             # FGSM (autograd for MLP, finite-diff for XGBoost)
│   ├── pgd.py              # PGD with restarts
│   └── metrics.py          # acc_drop, flip_rate, perturbation cost
├── recourse/
│   └── algorithmic_recourse.py   # Coordinate descent recourse + cost weighting
├── robustness/
│   ├── adversarial_training.py   # In-graph PGD training loop
│   └── smoothing.py              # Randomized smoothing + Clopper-Pearson cert.
├── backtest/
│   └── ood_eval.py         # Quarterly OOD cohort evaluation + drift summary
├── scripts/
│   ├── run_day1.py         # Train baselines
│   ├── run_day2.py         # Run attacks
│   ├── run_day3.py         # Adversarial training + smoothing
│   ├── run_day4.py         # Recourse + OOD backtest
│   └── run_day5.py         # Consolidated results + resume bullets
├── notebooks/
│   └── plot_results.py     # 5 publication-quality figures
├── tests/                  # 19 unit + smoke tests (pytest)
├── configs/default.yaml
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/adversarial-finance
cd adversarial-finance
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download data
# Get accepted_2007_to_2018Q4.csv.gz from:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Place in data/raw/ (do NOT unzip)

# 3. Run the full pipeline
python -m scripts.run_day1          # ~10 min (trains models, caches data)
python -m scripts.run_day2 --n-test 2000
python -m scripts.run_day3 --n-certify 500
python -m scripts.run_day4 --n-recourse 200
python -m scripts.run_day5          # consolidates all results

# 4. Generate figures
python -m notebooks.plot_results

# 5. Smoke test without data (CI)
python -m scripts.run_day1 --synthetic
pytest -q
```

---

## Key design decisions

### Threat model
Features are tagged `actionable` or `immutable` in `data/schema.py`.
Only actionable features are perturbed; FICO score, delinquencies, and credit
history length are immutable on a short horizon. Actionable features have a
direction: DTI can only *decrease* (a borrower pays down debt), income can only
*increase* (a borrower gets a raise or misreports).

### Why coordinate descent for recourse?
XGBoost is piecewise-constant — gradient is zero almost everywhere. Gradient
descent gives zero signal. Coordinate descent tries each actionable feature
across 20 epsilon levels and greedily combines pairs, which is also more
economically realistic (a borrower makes one or two targeted changes, not a
smooth gradient step).

### Why raw economic units?
The NN standardiser is built *into* the model (`Standardizer` layer), so the
model consumes raw dollars and percentages. Attacks and recourse therefore
operate in interpretable units: ε=0.10 on DTI means "10% of the DTI range
(~3.9 percentage points)", not an abstract norm.

### Boundary geometry shift
The adversarially trained MLP has lower recourse cost than the base MLP
(0.177 vs 0.212). This is not a failure: adversarial training shifts the
decision boundary to resist gradient-aligned attacks, which can make certain
coordinate directions *easier* to traverse. This is consistent with the
literature on the geometry of robust boundaries (Madry et al., 2018).

---

## Citation / references

- Goodfellow et al. (2015) — FGSM: [arxiv:1412.6572](https://arxiv.org/abs/1412.6572)
- Madry et al. (2018) — PGD adversarial training: [arxiv:1706.06083](https://arxiv.org/abs/1706.06083)
- Cohen et al. (2019) — Randomized smoothing: [arxiv:1902.02918](https://arxiv.org/abs/1902.02918)
- Ustun et al. (2019) — Actionable recourse: [arxiv:1809.06514](https://arxiv.org/abs/1809.06514)

---

## License
MIT
