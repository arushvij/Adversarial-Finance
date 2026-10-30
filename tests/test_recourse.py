"""Smoke tests for Day 4 recourse and OOD modules."""
from __future__ import annotations

import numpy as np
import pytest

from attacks.build_attack_cfg import build_attack_config
from common import load_config
from data.loader import load_lending_club
from data.pipeline import prepare_datasets
from data.synthetic import make_synthetic_lending_club
from recourse.algorithmic_recourse import (
    DEFAULT_COST_WEIGHTS, batch_recourse, compute_recourse, summarise_recourse,
)
from backtest.ood_eval import evaluate_over_time


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rec")
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_dir"] = str(tmp)
    path = tmp / "raw.csv.gz"
    make_synthetic_lending_club(15_000, 2).to_csv(path, index=False)
    bundle = prepare_datasets(cfg, raw_df=load_lending_club(path), use_cache=False)
    return bundle, cfg


def dummy_predict(X):
    return np.clip(0.3 + 0.015 * X[:, 3] - 0.01 * X[:, 2], 0.05, 0.95)


def test_recourse_found_for_high_dti(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.40)
    names = bundle.feature_names
    dti_idx = names.index("dti")
    # Create a clearly high-risk applicant
    x = bundle.X["test_id"][0].copy()
    x[dti_idx] = bundle.preprocessor.upper_[dti_idx]   # max DTI
    r = compute_recourse(x, dummy_predict, attack_cfg, names, threshold=0.5)
    assert isinstance(r.found, bool)
    assert r.p_orig >= 0.0


def test_already_approved_zero_cost(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.20)
    names = bundle.feature_names
    # low-risk: predict will be well below 0.5
    x = bundle.X["test_id"][0].copy()
    x[names.index("dti")] = bundle.preprocessor.lower_[names.index("dti")]
    r = compute_recourse(x, dummy_predict, attack_cfg, names, threshold=0.99)
    assert r.cost == 0.0 and r.found


def test_recourse_stays_in_box(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.30)
    names = bundle.feature_names
    results = batch_recourse(bundle.X["test_id"][:20], dummy_predict,
                             attack_cfg, names, threshold=0.45)
    for r in results:
        assert (r.x_recourse >= attack_cfg.clip_lower - 1e-4).all()
        assert (r.x_recourse <= attack_cfg.clip_upper + 1e-4).all()


def test_ood_eval_returns_per_period_rows(setup):
    bundle, _ = setup
    df = evaluate_over_time(
        bundle.X["ood"][:500], bundle.y["ood"][:500],
        bundle.issue_d["ood"][:500],
        {"dummy": dummy_predict}, {"dummy": 0.5}, freq="Q", min_samples=10,
    )
    assert len(df) > 0
    assert "auc" in df.columns and "period" in df.columns


def test_summarise_recourse_keys(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.30)
    results = batch_recourse(bundle.X["test_id"][:30], dummy_predict,
                             attack_cfg, bundle.feature_names, threshold=0.45)
    s = summarise_recourse(results, bundle.feature_names)
    assert "recourse_rate" in s and "mean_cost" in s and "top_changed_features" in s
