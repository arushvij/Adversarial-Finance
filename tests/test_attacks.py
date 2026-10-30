"""Smoke tests for Day 2 attack modules."""
from __future__ import annotations

import numpy as np
import pytest

from attacks.base import AttackConfig
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from common import load_config
from data.loader import load_lending_club
from data.pipeline import prepare_datasets
from data.synthetic import make_synthetic_lending_club
from attacks.build_attack_cfg import build_attack_config


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("d")
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_dir"] = str(tmp)
    path = tmp / "raw.csv.gz"
    make_synthetic_lending_club(20_000, 1).to_csv(path, index=False)
    bundle = prepare_datasets(cfg, raw_df=load_lending_club(path), use_cache=False)
    return bundle, cfg


def dummy_predict(X):
    """Linear scorer: higher dti (col 3) -> higher default prob."""
    return np.clip(0.3 + 0.01 * X[:, 3], 0, 1)


def test_fgsm_reduces_predicted_prob(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.10)
    X = bundle.X["test_id"][:50]
    fgsm = FGSM(dummy_predict, attack_cfg)
    X_adv = fgsm.perturb(X)
    assert dummy_predict(X_adv).mean() <= dummy_predict(X).mean() + 1e-6


def test_pgd_reduces_predicted_prob(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.10)
    X = bundle.X["test_id"][:30]
    pgd = PGD(dummy_predict, attack_cfg, n_steps=10, step_size=0.01)
    X_adv = pgd.perturb(X)
    assert dummy_predict(X_adv).mean() <= dummy_predict(X).mean() + 1e-6


def test_immutable_features_unchanged(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.20)
    X = bundle.X["test_id"][:50]
    pgd = PGD(dummy_predict, attack_cfg, n_steps=10, step_size=0.02)
    X_adv = pgd.perturb(X)
    immutable = ~attack_cfg.actionable_mask
    np.testing.assert_array_equal(X[:, immutable], X_adv[:, immutable])


def test_adversarial_stays_in_box(setup):
    bundle, _ = setup
    attack_cfg = build_attack_config(bundle, epsilon=0.15)
    X = bundle.X["test_id"][:50]
    pgd = PGD(dummy_predict, attack_cfg, n_steps=10, step_size=0.015)
    X_adv = pgd.perturb(X)
    assert (X_adv >= attack_cfg.clip_lower - 1e-4).all()
    assert (X_adv <= attack_cfg.clip_upper + 1e-4).all()


def test_direction_constraints(setup):
    """dti must only decrease, annual_inc must only increase."""
    bundle, _ = setup
    names = bundle.preprocessor.feature_names_
    dti_idx = names.index("dti")
    inc_idx = names.index("annual_inc")
    attack_cfg = build_attack_config(bundle, epsilon=0.20)
    X = bundle.X["test_id"][:50]
    pgd = PGD(dummy_predict, attack_cfg, n_steps=15, step_size=0.02)
    X_adv = pgd.perturb(X)
    assert (X_adv[:, dti_idx] <= X[:, dti_idx] + 1e-4).all(), "dti should only decrease"
    assert (X_adv[:, inc_idx] >= X[:, inc_idx] - 1e-4).all(), "annual_inc should only increase"
