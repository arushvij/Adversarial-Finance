"""Load and filter the LendingClub accepted-loans file."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import get_logger
from data.schema import RAW_COLUMNS, TARGET

log = get_logger(__name__)

_RESOLVED = {"Fully Paid": 0, "Charged Off": 1}


def load_lending_club(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw CSV, keep resolved individual loans, and attach labels.

    Keeps only loans with a final outcome (``Fully Paid`` -> 0, ``Charged Off`` -> 1);
    ``Current`` / ``Late`` / ``In Grace Period`` loans have no label yet.

    Returns:
        DataFrame with the raw columns, a parsed datetime ``issue_d`` and an
        integer ``default`` target, sorted by issue date.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'accepted_2007_to_2018Q4.csv.gz' from "
            "https://www.kaggle.com/datasets/wordsforthewise/lending-club "
            "and place it in data/raw/."
        )
    log.info("Reading %s ...", path)
    df = pd.read_csv(path, usecols=list(RAW_COLUMNS), nrows=nrows, low_memory=False)
    n_raw = len(df)

    df = df[df["loan_status"].isin(_RESOLVED)].copy()
    df = df[df["application_type"] == "Individual"].copy()
    df[TARGET] = df["loan_status"].map(_RESOLVED).astype("int8")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df.sort_values("issue_d").reset_index(drop=True)

    log.info("Kept %d / %d rows | default rate %.3f | %s -> %s",
             len(df), n_raw, df[TARGET].mean(), df["issue_d"].min().date(), df["issue_d"].max().date())
    return df
