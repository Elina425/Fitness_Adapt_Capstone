from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import BiLSTMMultiHead, freeze_base_model_for_personalization


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 8
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    device: str = "cpu"


def make_loader(x: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y_cls).long(),
        torch.from_numpy(y_reg).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(model, loader, optimizer, cfg: TrainConfig):
    model.train()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total = 0.0
    for xb, yb_cls, yb_reg in loader:
        xb = xb.to(cfg.device)
        yb_cls = yb_cls.to(cfg.device)
        yb_reg = yb_reg.to(cfg.device)
        optimizer.zero_grad()
        logits, pred_reg = model(xb, use_adapter=False)
        cls_loss = ce(logits, yb_cls)
        reg_loss = mse(pred_reg, yb_reg)
        loss = cfg.cls_loss_weight * cls_loss + cfg.reg_loss_weight * reg_loss
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * xb.size(0)
    return total / len(loader.dataset)


def evaluate_model(model, loader, device: str, use_adapter: bool = False) -> Dict[str, float]:
    model.eval()
    all_cls_true, all_cls_pred = [], []
    all_reg_true, all_reg_pred = [], []
    with torch.no_grad():
        for xb, yb_cls, yb_reg in loader:
            xb = xb.to(device)
            logits, pred_reg = model(xb, use_adapter=use_adapter)
            pred_cls = torch.argmax(logits, dim=1).cpu().numpy()
            all_cls_pred.append(pred_cls)
            all_cls_true.append(yb_cls.numpy())
            all_reg_pred.append(pred_reg.cpu().numpy())
            all_reg_true.append(yb_reg.numpy())

    y_cls_t = np.concatenate(all_cls_true)
    y_cls_p = np.concatenate(all_cls_pred)
    y_reg_t = np.concatenate(all_reg_true)
    y_reg_p = np.concatenate(all_reg_pred)

    # Single-class-safe metrics
    acc = float(accuracy_score(y_cls_t, y_cls_p))
    try:
        f1 = float(f1_score(y_cls_t, y_cls_p, average="macro"))
    except ValueError:
        f1 = 1.0 if np.all(y_cls_t == y_cls_p) else 0.0
    mae = float(mean_absolute_error(y_reg_t, y_reg_p))
    try:
        r2 = float(r2_score(y_reg_t, y_reg_p))
    except ValueError:
        r2 = 0.0
    return {"accuracy": acc, "f1_macro": f1, "mae": mae, "r2": r2}


def train_multitask_bilstm(
    x_train: np.ndarray,
    y_cls_train: np.ndarray,
    y_reg_train: np.ndarray,
    x_val: np.ndarray,
    y_cls_val: np.ndarray,
    y_reg_val: np.ndarray,
    *,
    input_dim: int,
    num_classes: int,
    cfg: TrainConfig,
) -> Tuple[BiLSTMMultiHead, Dict[str, float]]:
    model = BiLSTMMultiHead(input_dim=input_dim, num_classes=num_classes).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = make_loader(x_train, y_cls_train, y_reg_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_cls_val, y_reg_val, cfg.batch_size, shuffle=False)

    for _ in range(cfg.epochs):
        _train_one_epoch(model, train_loader, optimizer, cfg)
    metrics = evaluate_model(model, val_loader, cfg.device, use_adapter=False)
    return model, metrics


def run_ablation(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    *,
    cfg: TrainConfig,
    num_classes: int,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for feat_name in ["x_raw", "x_angles"]:
        model, metrics = train_multitask_bilstm(
            train_data[feat_name],
            train_data["y_exercise"],
            train_data["y_quality"],
            val_data[feat_name],
            val_data["y_exercise"],
            val_data["y_quality"],
            input_dim=train_data[feat_name].shape[-1],
            num_classes=num_classes,
            cfg=cfg,
        )
        results[feat_name] = metrics
    return results


def personalize_on_user_windows(
    model: BiLSTMMultiHead,
    user_x: np.ndarray,
    user_y_cls: np.ndarray,
    user_y_reg: np.ndarray,
    *,
    cfg: TrainConfig,
    adapter_steps: int = 40,
) -> Dict[str, float]:
    """
    Task-8 personalization:
    freeze base model and tune only adapter on user session data.
    """
    freeze_base_model_for_personalization(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    loader = make_loader(user_x, user_y_cls, user_y_reg, cfg.batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    model.train()
    step = 0
    while step < adapter_steps:
        for xb, yb_cls, yb_reg in loader:
            xb = xb.to(cfg.device)
            yb_cls = yb_cls.to(cfg.device)
            yb_reg = yb_reg.to(cfg.device)
            opt.zero_grad()
            logits, pred_reg = model(xb, use_adapter=True)
            loss = ce(logits, yb_cls) + mse(pred_reg, yb_reg)
            loss.backward()
            opt.step()
            step += 1
            if step >= adapter_steps:
                break
    return evaluate_model(model, loader, cfg.device, use_adapter=True)

