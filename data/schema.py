"""Feature schema and the *financial threat model*.

Each numeric feature is tagged with whether a borrower can plausibly change it
and in which direction. Attacks (Day 2) and recourse (Day 3) will only perturb
``actionable`` features, in ``direction``-consistent ways. This is what makes
the adversarial examples economically meaningful rather than arbitrary L_p noise.

Design choices (document these in your README):
  * ``dti``, ``revol_util``, ``revol_bal``: a borrower can pay down debt        -> decrease
  * ``annual_inc``: real raise OR misreported/inflated income (fraud vector)   -> increase
  * ``loan_amnt``: borrower can ask for less                                    -> both
  * ``open_acc`` / ``total_acc``: open or close accounts (total can only grow)  -> both / increase
  * Credit history facts (delinquencies, inquiries, FICO, history length) are NOT
    directly manipulable on a short horizon -> immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["increase", "decrease", "both", "none"]


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one numeric model feature."""

    name: str
    actionable: bool
    direction: Direction = "none"
    integer: bool = False


NUMERIC_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("loan_amnt", True, "both"),
    FeatureSpec("term_months", False),
    FeatureSpec("annual_inc", True, "increase"),
    FeatureSpec("dti", True, "decrease"),
    FeatureSpec("revol_util", True, "decrease"),
    FeatureSpec("revol_bal", True, "decrease"),
    FeatureSpec("open_acc", True, "both", integer=True),
    FeatureSpec("total_acc", True, "increase", integer=True),
    FeatureSpec("fico_avg", False),
    FeatureSpec("inq_last_6mths", False, integer=True),
    FeatureSpec("delinq_2yrs", False, integer=True),
    FeatureSpec("pub_rec", False, integer=True),
    FeatureSpec("pub_rec_bankruptcies", False, integer=True),
    FeatureSpec("mort_acc", False, integer=True),
    FeatureSpec("credit_history_years", False),
    FeatureSpec("emp_length_years", False),
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("home_ownership", "verification_status", "purpose")

NUMERIC_NAMES: tuple[str, ...] = tuple(f.name for f in NUMERIC_FEATURES)
FEATURE_SPECS: dict[str, FeatureSpec] = {f.name: f for f in NUMERIC_FEATURES}

# Columns read from the raw Kaggle file. Deliberately EXCLUDES post-origination
# leakage (total_pymnt, recoveries, last_pymnt_d, ...) and LendingClub's own
# risk outputs (grade, sub_grade, int_rate, installment) which a borrower cannot
# manipulate and which would leak the incumbent underwriting model.
RAW_COLUMNS: tuple[str, ...] = (
    "loan_status", "issue_d", "application_type", "earliest_cr_line", "term",
    "emp_length", "loan_amnt", "annual_inc", "dti", "revol_util", "revol_bal",
    "open_acc", "total_acc", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "mort_acc",
    "home_ownership", "verification_status", "purpose",
)

TARGET = "default"
