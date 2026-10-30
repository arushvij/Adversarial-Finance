"""End-to-end data preparation: load -> temporal split -> preprocess -> cache."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import get_logger
from data.loader import load_lending_club
from data.preprocess import TabularPreprocessor, build_features, target_series
from data.schema import TARGET

log = get_logger(__name__)
SPLIT_NAMES = ("train", "val", "test_id", "ood")


@dataclass
class DatasetBundle:
    """Model-ready arrays (raw economic units) for every split."""

    X: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    issue_d: dict[str, np.ndarray]
    preprocessor: TabularPreprocessor

    @property
    def feature_names(self) -> list[str]:
        return self.preprocessor.feature_names_


def temporal_split(df: pd.DataFrame, cfg: dict[str, Any], seed: int) -> dict[str, pd.DataFrame]:
    """train/val/test_id: stratified random split of the training window.
    ood: a strictly later window, never used for fitting or tuning."""
    d = cfg["data"]
    issue = df["issue_d"]
    dev = df[(issue >= d["train_start"]) & (issue <= d["train_end"])]
    ood = df[(issue >= d["ood_start"]) & (issue <= d["ood_end"])]
    trainval, test_id = train_test_split(dev, test_size=d["test_frac"], stratify=dev[TARGET], random_state=seed)
    train, val = train_test_split(trainval, test_size=d["val_frac"] / (1 - d["test_frac"]),
                                  stratify=trainval[TARGET], random_state=seed)
    return {"train": train, "val": val, "test_id": test_id, "ood": ood}


def prepare_datasets(cfg: dict[str, Any], raw_df: pd.DataFrame | None = None,
                     use_cache: bool = True, cache_name: str = "bundle.joblib") -> DatasetBundle:
    """Build (or load from cache) the full dataset bundle."""
    cache = Path(cfg["data"]["processed_dir"]) / cache_name
    if use_cache and cache.exists():
        log.info("Loading cached bundle %s", cache)
        return joblib.load(cache)

    if raw_df is None:
        raw_df = load_lending_club(cfg["data"]["raw_path"], cfg["data"]["nrows"])
    splits = temporal_split(raw_df, cfg, cfg["seed"])
    feats = {k: build_features(v) for k, v in splits.items()}

    pre = TabularPreprocessor(tuple(cfg["data"]["winsor_quantiles"])).fit(feats["train"])  # TRAIN only
    bundle = DatasetBundle(
        X={k: pre.transform(f).to_numpy(dtype=np.float32) for k, f in feats.items()},
        y={k: target_series(v) for k, v in splits.items()},
        issue_d={k: v["issue_d"].to_numpy() for k, v in splits.items()},
        preprocessor=pre,
    )
    for k in SPLIT_NAMES:
        log.info("%-8s n=%-8d default_rate=%.3f", k, len(bundle.y[k]), bundle.y[k].mean())
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cache)
    return bundle
