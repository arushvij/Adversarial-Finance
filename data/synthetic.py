"""Synthetic LendingClub-shaped data for smoke tests / CI ONLY.

Never report numbers from this generator. Real results must come from the real
Kaggle file. The schema (string formats, column names) mimics the raw file so
the full loader -> preprocessor path is exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.schema import RAW_COLUMNS


def make_synthetic_lending_club(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = rng.choice([2012, 2013, 2014, 2015, 2016], size=n, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    months = rng.integers(1, 13, size=n)
    issue = pd.to_datetime(pd.DataFrame({"year": years, "month": months, "day": 1}))

    annual_inc = np.exp(rng.normal(11.1, 0.5, n))
    loan_amnt = np.clip(rng.normal(14000, 8000, n), 1000, 40000).round(-2)
    dti = np.clip(rng.normal(18, 8, n), 0, 50)
    revol_util = np.clip(rng.normal(55, 25, n), 0, 120)
    revol_bal = np.exp(rng.normal(9.4, 1.0, n))
    open_acc = rng.poisson(11, n).astype(float)
    total_acc = open_acc + rng.poisson(13, n)
    fico_low = (np.clip(rng.normal(695, 30, n), 660, 845) // 5 * 5)
    term = rng.choice([36, 60], size=n, p=[0.75, 0.25])
    hist_years = rng.uniform(3, 30, n)
    mort_acc = rng.poisson(1.5, n).astype(float)
    mort_acc[rng.random(n) < 0.05] = np.nan

    drift = 0.03 * (dti - 18) * (years >= 2015)   # relationship shifts out-of-time
    logit = (-1.55 + 0.035 * (dti - 18) + 0.012 * (revol_util - 55) - 0.02 * (fico_low - 695)
             + 0.35 * (term == 60) - 0.3 * (np.log(annual_inc) - 11.1) + drift)
    default = rng.random(n) < 1 / (1 + np.exp(-logit))
    status = np.where(default, "Charged Off", "Fully Paid").astype(object)
    status[rng.random(n) < 0.05] = "Current"

    emp = rng.choice(["< 1 year", "1 year", "3 years", "5 years", "10+ years", None], size=n)
    earliest = (issue - pd.to_timedelta(hist_years * 365.25, unit="D")).dt.strftime("%b-%Y")
    df = pd.DataFrame({
        "loan_status": status, "issue_d": issue.dt.strftime("%b-%Y"),
        "application_type": "Individual", "earliest_cr_line": earliest,
        "term": [f" {t} months" for t in term], "emp_length": emp,
        "loan_amnt": loan_amnt, "annual_inc": annual_inc, "dti": dti, "revol_util": revol_util,
        "revol_bal": revol_bal, "open_acc": open_acc, "total_acc": total_acc,
        "fico_range_low": fico_low, "fico_range_high": fico_low + 4,
        "inq_last_6mths": rng.poisson(0.8, n), "delinq_2yrs": rng.poisson(0.3, n),
        "pub_rec": rng.poisson(0.15, n), "pub_rec_bankruptcies": rng.poisson(0.08, n),
        "mort_acc": mort_acc,
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], size=n, p=[0.45, 0.4, 0.14, 0.01]),
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], size=n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other"], size=n),
    })
    return df[list(RAW_COLUMNS)]
