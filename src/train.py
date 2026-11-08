"""Training / scoring utilities for self-supervised next-step prediction."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 0, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = False) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(windows: dict, batch_size=32, shuffle=True, seed=0) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(windows["X"]).float(),
                       torch.from_numpy(windows["Y"]).float(),
                       torch.from_numpy(windows["M"]).float())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
def next_step_loss(model, xb, yb, mb):
    """Next-step objective: output at t predicts Y at t+1. Missing targets are masked out.

    xb (B, L, Cin), yb (B, L, C), mb (B, L, C) with 1 = missing.
    """
    pred = model(xb[:, :-1])                 # (B, L-1, C) predictions for t = 1..L-1
    target = yb[:, 1:]
    valid = 1.0 - mb[:, 1:]
    se = (pred - target) ** 2 * valid
    return se.sum() / valid.sum().clamp_min(1.0)


# --------------------------------------------------------------------------------------
# Loops
# --------------------------------------------------------------------------------------
def train_one_epoch(model, loader, opt, device, clip=1.0):
    model.train()
    total, n = 0.0, 0
    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        loss = next_step_loss(model, xb, yb, mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        total += next_step_loss(model, xb, yb, mb).item() * len(xb)
        n += len(xb)
    return total / n


def save_checkpoint(path, model, epoch, val_loss, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": getattr(model, "config", None),
                "epoch": epoch, "val_loss": val_loss, "extra": extra or {}}, path)


def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def fit(model, train_loader, val_loader, epochs=20, lr=1e-3, weight_decay=1e-2, clip=1.0,
        ckpt_path="checkpoints/best.pt", patience=None, device="cpu", verbose=True):
    """AdamW + grad clipping; saves the checkpoint with the lowest validation loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "time_s": []}
    best, best_epoch, bad = float("inf"), -1, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, device, clip)
        va = evaluate(model, val_loader, device)
        dt = time.time() - t0
        assert np.isfinite(tr) and np.isfinite(va), f"non-finite loss at epoch {ep}"
        for k, v in zip(history, (ep, tr, va, dt)):
            history[k].append(v)
        flag = ""
        if va < best:
            best, best_epoch, bad = va, ep, 0
            save_checkpoint(ckpt_path, model, ep, va)
            flag = "  * saved"
        else:
            bad += 1
        if verbose:
            print(f"epoch {ep:3d} | train {tr:.4f} | val {va:.4f} | {dt:5.1f}s{flag}")
        if patience is not None and bad >= patience:
            print(f"early stopping (no val improvement for {patience} epochs)")
            break
    history["best_val"], history["best_epoch"] = best, best_epoch
    return history


def overfit_single_batch(model, batch, steps=300, lr=3e-3, device="cpu", log_every=50):
    """Sanity check: a correct model + loss should drive the loss on ONE batch near zero."""
    xb, yb, mb = (t.to(device) for t in batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    for i in range(steps):
        loss = next_step_loss(model, xb, yb, mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and (i % log_every == 0 or i == steps - 1):
            print(f"step {i:4d}  loss {loss.item():.5f}")
    return losses


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_series(model, X, window=512, warmup=128, device="cpu"):
    """Next-step predictions over a whole contiguous split using overlapping windows.

    Each window after the first discards its first `warmup` outputs (state burn-in),
    so every target index is predicted exactly once with >= warmup steps of context.
    Returns preds (T, C) where preds[t] is the prediction OF Y[t] (row 0 is NaN).
    """
    model.eval()
    T = len(X)
    c_out = model.head.out_features
    preds = np.full((T, c_out), np.nan, dtype=np.float32)
    s = 0
    while True:
        e = min(s + window, T - 1)                       # inputs X[s:e] -> targets s+1..e
        out = model(torch.from_numpy(X[s:e]).float().unsqueeze(0).to(device))[0].cpu().numpy()
        keep = 0 if s == 0 else warmup
        preds[s + 1 + keep:e + 1] = out[keep:]
        if e >= T - 1:
            break
        s += window - warmup
    assert not np.isnan(preds[1:]).any(), "some timesteps were not predicted"
    return preds


def anomaly_scores(preds, Y, M):
    """s_t = mean over observed channels of squared next-step error. Also per-channel errors."""
    err = (preds - Y) ** 2
    valid = 1.0 - M
    err_valid = np.where(valid > 0, err, 0.0)
    s = err_valid.sum(1) / np.maximum(valid.sum(1), 1.0)
    s[0] = 0.0                                          # no prediction for the first sample
    s = np.nan_to_num(s, nan=0.0)
    return s, np.nan_to_num(err_valid, nan=0.0)
