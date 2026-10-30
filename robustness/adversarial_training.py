"""Adversarial training for the MLP (Madry et al., 2018).

At each mini-batch:
  1. Generate PGD adversarial examples on the CURRENT model weights
  2. Train on the adversarial examples (not the clean ones)

This is the gold-standard empirical defence. We also support mixed training
(alpha fraction adversarial + (1-alpha) clean) which often preserves more
clean accuracy.

Key design: we reuse the same PGD implementation from attacks/pgd.py so the
defence is consistent with the evaluation — no looser inner loop.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks.base import AttackConfig
from attacks.pgd import PGD
from common import get_logger
from models.nn_model import CreditMLP, predict_proba_nn

log = get_logger(__name__)


def _pgd_batch_torch(model: CreditMLP, xb: torch.Tensor,
                     attack_cfg: AttackConfig, n_steps: int, step_size: float) -> torch.Tensor:
    """In-graph PGD for a single training batch (stays on device)."""
    eps_abs = torch.from_numpy(attack_cfg.epsilon * attack_cfg.feature_range()).float()
    act = torch.from_numpy(attack_cfg.actionable_mask).bool()
    lo = torch.from_numpy(attack_cfg.clip_lower).float()
    hi = torch.from_numpy(attack_cfg.clip_upper).float()
    dir_ = torch.from_numpy(attack_cfg.direction).float()

    x_adv = xb.detach().clone()
    alpha = step_size * eps_abs

    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        loss = torch.sigmoid(model(x_adv)).sum()
        loss.backward()
        with torch.no_grad():
            grad = x_adv.grad.detach()
            step = -torch.sign(grad) * alpha          # minimise P(default)
            step = torch.where(act.unsqueeze(0), step, torch.zeros_like(step))
            x_adv = x_adv.detach() + step

            # direction constraints
            inc = act & (dir_ == 1)
            dec = act & (dir_ == -1)
            x_adv = torch.where(inc.unsqueeze(0), torch.maximum(x_adv, xb), x_adv)
            x_adv = torch.where(dec.unsqueeze(0), torch.minimum(x_adv, xb), x_adv)

            # L_inf ball + box
            x_adv = torch.max(torch.min(x_adv, xb + eps_abs), xb - eps_abs)
            x_adv = torch.clamp(x_adv, lo, hi)

    return x_adv.detach()


def train_mlp_adversarial(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    base_model: CreditMLP,
    attack_cfg: AttackConfig,
    params: dict[str, Any],
    adv_params: dict[str, Any],
    seed: int = 42,
) -> CreditMLP:
    """Adversarially retrain a model starting from base_model weights.

    adv_params keys:
        pgd_steps (int): inner PGD steps per batch (7 is standard)
        pgd_step_size (float): fraction of feature range per inner step
        mix_alpha (float): fraction of batch that is adversarial (1.0 = fully adv)
        max_epochs, patience, lr, weight_decay, batch_size (same as nn params)
    """
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deepcopy(base_model).to(dev).train()

    opt = torch.optim.Adam(model.parameters(),
                           lr=adv_params.get("lr", params["lr"]),
                           weight_decay=adv_params.get("weight_decay", params["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=adv_params.get("batch_size", params["batch_size"]), shuffle=True)

    n_steps = adv_params.get("pgd_steps", 7)
    step_size = adv_params.get("pgd_step_size", 0.01)
    mix_alpha = adv_params.get("mix_alpha", 1.0)
    max_epochs = adv_params.get("max_epochs", params["max_epochs"])
    patience = adv_params.get("patience", params["patience"])

    best_auc, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            if mix_alpha >= 1.0:
                xb_train = _pgd_batch_torch(model, xb, attack_cfg, n_steps, step_size)
            else:
                n_adv = int(mix_alpha * len(xb))
                xb_adv = _pgd_batch_torch(model, xb[:n_adv], attack_cfg, n_steps, step_size)
                xb_train = torch.cat([xb_adv, xb[n_adv:]], dim=0)
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(xb_train), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        val_auc = roc_auc_score(y_val, predict_proba_nn(model, X_val))
        log.info("adv epoch %02d | loss %.4f | val_auc %.4f", epoch + 1, total_loss / len(ds), val_auc)
        if val_auc > best_auc + 1e-5:
            best_auc, best_state, bad_epochs = val_auc, deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stop at epoch %d (best val_auc %.4f)", epoch + 1, best_auc)
                break

    model.load_state_dict(best_state)
    return model.cpu().eval()
