"""Smoke tests for Day 3 robustness modules."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attacks.base import AttackConfig
from attacks.build_attack_cfg import build_attack_config
from common import load_config
from data.loader import load_lending_club
from data.pipeline import prepare_datasets
from data.synthetic import make_synthetic_lending_club
from models.nn_model import CreditMLP, predict_proba_nn, train_mlp
from robustness.smoothing import SmoothedClassifier, summarise_certification


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r")
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_dir"] = str(tmp)
    cfg["nn"]["max_epochs"] = 3
    cfg["nn"]["patience"] = 2
    path = tmp / "raw.csv.gz"
    make_synthetic_lending_club(10_000, 1).to_csv(path, index=False)
    bundle = prepare_datasets(cfg, raw_df=load_lending_club(path), use_cache=False)
    model = train_mlp(bundle.X["train"], bundle.y["train"],
                      bundle.X["val"], bundle.y["val"], cfg["nn"], 42)
    return bundle, model, cfg


def test_smoothed_classifier_returns_valid_class(setup):
    bundle, model, _ = setup
    sc = SmoothedClassifier(lambda X: predict_proba_nn(model, X),
                            sigma=1.0, n_samples=50, threshold=0.3)
    result = sc.certify(bundle.X["test_id"][0])
    assert result.predicted_class in {-1, 0, 1}
    assert result.certified_radius >= 0.0


def test_certified_rate_between_0_and_1(setup):
    bundle, model, _ = setup
    sc = SmoothedClassifier(lambda X: predict_proba_nn(model, X),
                            sigma=1.0, n_samples=50, threshold=0.3)
    results = sc.certify_dataset(bundle.X["test_id"][:20])
    summary = summarise_certification(results)
    assert 0.0 <= summary["certified_rate"] <= 1.0
    assert summary["mean_certified_radius"] >= 0.0


def test_larger_sigma_gives_larger_radii(setup):
    bundle, model, _ = setup
    X = bundle.X["test_id"][:30]
    radii = {}
    for sigma in [0.5, 2.0]:
        sc = SmoothedClassifier(lambda X: predict_proba_nn(model, X),
                                sigma=sigma, n_samples=80, threshold=0.3)
        res = sc.certify_dataset(X)
        radii[sigma] = np.mean([r.certified_radius for r in res])
    assert radii[2.0] >= radii[0.5] - 0.05   # allow small tolerance


def test_adv_training_runs_one_epoch(setup):
    bundle, model, cfg = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.10)
    adv_params = {"pgd_steps": 2, "pgd_step_size": 0.01, "mix_alpha": 0.5,
                  "max_epochs": 1, "patience": 1, "lr": 1e-3,
                  "weight_decay": 1e-5, "batch_size": 256}
    from robustness.adversarial_training import train_mlp_adversarial
    adv_model = train_mlp_adversarial(
        bundle.X["train"][:500], bundle.y["train"][:500],
        bundle.X["val"][:100], bundle.y["val"][:100],
        model, attack_cfg, cfg["nn"], adv_params, seed=0,
    )
    p = predict_proba_nn(adv_model, bundle.X["test_id"][:10])
    assert p.shape == (10,) and (p >= 0).all() and (p <= 1).all()
