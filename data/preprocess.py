"""Feature engineering and a fit/transform preprocessor.

Design principle: the preprocessor outputs features in **raw economic units**
(dollars, percent, counts) plus one-hot columns. No standardisation happens here.
Scaling lives inside the neural network (see ``models/nn_model.py``), so any
attack can perturb ``dti`` by "+3 points" directly in economic units, and the
cost of a perturbation is interpretable.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data.schema import CATEGORICAL_FEATURES, FEATURE_SPECS, NUMERIC_NAMES, TARGET


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning + derived features from raw LendingClub columns."""
    issue = pd.to_datetime(raw["issue_d"])
    earliest = pd.to_datetime(raw["earliest_cr_line"], format="%b-%Y")

    emp = raw["emp_length"].astype("string")
    emp_years = emp.str.extract(r"(\d+)")[0].astype("float64")
    emp_years = emp_years.where(~emp.str.contains("<", na=False), 0.0)

    out = pd.DataFrame(index=raw.index)
    out["loan_amnt"] = raw["loan_amnt"].astype("float64")
    out["term_months"] = raw["term"].astype("string").str.extract(r"(\d+)")[0].astype("float64")
    out["annual_inc"] = raw["annual_inc"].astype("float64")
    out["dti"] = raw["dti"].astype("float64")
    out["revol_util"] = raw["revol_util"].astype("float64")
    out["revol_bal"] = raw["revol_bal"].astype("float64")
    out["open_acc"] = raw["open_acc"].astype("float64")
    out["total_acc"] = raw["total_acc"].astype("float64")
    out["fico_avg"] = (raw["fico_range_low"] + raw["fico_range_high"]) / 2.0
    out["inq_last_6mths"] = raw["inq_last_6mths"].astype("float64")
    out["delinq_2yrs"] = raw["delinq_2yrs"].astype("float64")
    out["pub_rec"] = raw["pub_rec"].astype("float64")
    out["pub_rec_bankruptcies"] = raw["pub_rec_bankruptcies"].astype("float64")
    out["mort_acc"] = raw["mort_acc"].astype("float64")
    out["credit_history_years"] = (issue - earliest).dt.days / 365.25
    out["emp_length_years"] = emp_years

    out["home_ownership"] = raw["home_ownership"].replace({"ANY": "OTHER", "NONE": "OTHER"})
    out["verification_status"] = raw["verification_status"]
    out["purpose"] = raw["purpose"]
    return out


class TabularPreprocessor:
    """Median-impute, winsorise and one-hot encode; fitted on TRAIN only.

    Attributes (after ``fit``):
        feature_names_: output column order (numeric first, then one-hots).
        lower_ / upper_: per-output-column valid range (winsor bounds for numerics,
            0/1 for one-hots). Attacks clip perturbed inputs to this box.
    """

    def __init__(self, winsor_quantiles: tuple[float, float] = (0.005, 0.995)) -> None:
        self.winsor_quantiles = winsor_quantiles
        self.numeric_cols: list[str] = list(NUMERIC_NAMES)
        self.categorical_cols: list[str] = list(CATEGORICAL_FEATURES)
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "TabularPreprocessor":
        num = features[self.numeric_cols]
        q_lo, q_hi = self.winsor_quantiles
        self.medians_ = num.median()
        self.num_lower_ = num.quantile(q_lo)
        self.num_upper_ = num.quantile(q_hi)
        self.categories_ = {c: sorted(features[c].dropna().unique().tolist()) for c in self.categorical_cols}

        cat_names = [f"{c}={v}" for c in self.categorical_cols for v in self.categories_[c]]
        self.feature_names_: list[str] = self.numeric_cols + cat_names
        self.lower_ = np.concatenate([self.num_lower_.to_numpy(), np.zeros(len(cat_names))]).astype(np.float32)
        self.upper_ = np.concatenate([self.num_upper_.to_numpy(), np.ones(len(cat_names))]).astype(np.float32)
        self._fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        num = features[self.numeric_cols].fillna(self.medians_)
        num = num.clip(lower=self.num_lower_, upper=self.num_upper_, axis=1)
        parts = [num]
        for c in self.categorical_cols:
            for v in self.categories_[c]:  # unseen categories at test time -> all zeros
                parts.append(pd.Series((features[c] == v).astype("float32"), name=f"{c}={v}"))
        out = pd.concat(parts, axis=1).astype("float32")
        return out[self.feature_names_]

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)

    # ---- threat-model helpers (used from Day 2) ------------------------------
    def actionable_mask(self) -> np.ndarray:
        """Boolean mask over ``feature_names_``: True where a borrower can act."""
        return np.array([FEATURE_SPECS[n].actionable if n in FEATURE_SPECS else False
                         for n in self.feature_names_])

    def direction_array(self) -> np.ndarray:
        """+1 increase-only, -1 decrease-only, 0 both/immutable (see ``actionable_mask``)."""
        sign = {"increase": 1, "decrease": -1, "both": 0, "none": 0}
        return np.array([sign[FEATURE_SPECS[n].direction] if n in FEATURE_SPECS else 0
                         for n in self.feature_names_])

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TabularPreprocessor":
        return joblib.load(path)


def target_series(raw: pd.DataFrame) -> np.ndarray:
    return raw[TARGET].to_numpy(dtype=np.int64)
