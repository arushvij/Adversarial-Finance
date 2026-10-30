"""PyTorch MLP baseline.

The model *contains* its input standardiser, so it consumes features in raw
economic units. Gradients therefore flow back to raw-unit inputs, which is what
gradient-based attacks (FGSM/PGD, Day 2) and randomized smoothing (Day 5) need.
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

from common import get_logger

log = get_logger(__name__)


class Standardizer(nn.Module):
    """Fixed (non-trainable) z-score layer: (x - mean) / std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class CreditMLP(nn.Module):
    """Standardiser -> MLP -> single logit for P(default)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [Standardizer(mean, std)]
        d = mean.numel()
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.hidden_dims, self.dropout = hidden_dims, dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch,)."""
        return self.net(x).squeeze(-1)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba_nn(model: CreditMLP, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """P(default) for a numpy array in raw units."""
    model.eval()
    dev = next(model.parameters()).device
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch_size]).to(dev))).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              params: dict[str, Any], seed: int = 42) -> CreditMLP:
    """Train with Adam + early stopping on validation AUC; returns best model (CPU, eval mode)."""
    torch.manual_seed(seed)
    dev = _device()
    mean = torch.from_numpy(X_train.mean(0))
    std = torch.from_numpy(X_train.std(0))
    model = CreditMLP(mean, std, list(params["hidden_dims"]), params["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=params["batch_size"], shuffle=True)

    best_auc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(params["max_epochs"]):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_auc = roc_auc_score(y_val, predict_proba_nn(model, X_val))
        log.info("epoch %02d | train_loss %.4f | val_auc %.4f", epoch + 1, total / len(ds), val_auc)
        if val_auc > best_auc + 1e-5:
            best_auc, best_state, bad_epochs = val_auc, deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= params["patience"]:
                log.info("Early stopping at epoch %d (best val_auc %.4f)", epoch + 1, best_auc)
                break
    model.load_state_dict(best_state)
    return model.cpu().eval()


def save_mlp(model: CreditMLP, path: str | Path) -> None:
    std_layer = model.net[0]
    torch.save({"state_dict": model.state_dict(), "mean": std_layer.mean, "std": std_layer.std,
                "hidden_dims": model.hidden_dims, "dropout": model.dropout}, path)


def load_mlp(path: str | Path) -> CreditMLP:
    ckpt = torch.load(path, map_location="cpu")
    model = CreditMLP(ckpt["mean"], ckpt["std"], ckpt["hidden_dims"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
