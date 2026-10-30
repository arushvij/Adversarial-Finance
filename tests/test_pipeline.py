"""Smoke tests for the Day 1 pipeline (run: pytest -q)."""
from __future__ import annotations

import numpy as np
import pytest

from common import load_config
from data.loader import load_lending_club
from data.pipeline import prepare_datasets
from data.synthetic import make_synthetic_lending_club


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("d")
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_dir"] = str(tmp)
    path = tmp / "raw.csv.gz"
    make_synthetic_lending_club(20_000, 1).to_csv(path, index=False)
    return prepare_datasets(cfg, raw_df=load_lending_club(path), use_cache=False), cfg


def test_no_nans_and_shapes(bundle):
    b, _ = bundle
    for split, X in b.X.items():
        assert not np.isnan(X).any(), split
        assert X.shape[1] == len(b.feature_names)


def test_only_resolved_binary_labels(bundle):
    b, _ = bundle
    for y in b.y.values():
        assert set(np.unique(y)) <= {0, 1}


def test_ood_is_strictly_later_than_train(bundle):
    b, _ = bundle
    assert b.issue_d["ood"].min() > b.issue_d["train"].max()


def test_preprocessor_fit_on_train_only_bounds(bundle):
    b, _ = bundle
    lo, hi = b.preprocessor.lower_, b.preprocessor.upper_
    assert (b.X["train"] >= lo - 1e-4).all() and (b.X["train"] <= hi + 1e-4).all()


def test_actionable_mask_marks_dti(bundle):
    b, _ = bundle
    names = b.feature_names
    mask = b.preprocessor.actionable_mask()
    assert mask[names.index("dti")] and not mask[names.index("fico_avg")]
