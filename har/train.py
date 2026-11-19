"""Training, evaluation and metrics (plan Section 2.4).

Every function that trains or tunes calls require_harth() on all data it receives: HAR70+ is test-only.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import NON_STAIR, STAIRS, Windows, require_harth

N_CLASSES = 7


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5                 # early stopping on validation pooled 7-class macro-F1
    label_smoothing: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Inverse class frequency over the given (training) labels, normalised to mean 1 over the
    classes present; absent classes get weight 0."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.divide(1.0, counts, out=np.zeros(n_classes), where=counts > 0)
    w = w / w[counts > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


# -------------------------------------------------------------------------------------- metrics

def pooled_macro_f1(y, pred, classes: Sequence[int] = tuple(range(N_CLASSES))) -> float:
    """Macro-F1 over all given windows pooled together (classes absent from both y and pred are
    left out of the average)."""
    present = [c for c in classes if (np.asarray(y) == c).any() or (np.asarray(pred) == c).any()]
    return float(f1_score(y, pred, labels=present, average="macro", zero_division=0)) if present else 0.0


def pooled_class_f1(y, pred, classes: Sequence[int] = STAIRS) -> dict:
    """F1 per class over all windows of a group pooled across subjects (default: stairs up, stairs
    down) and their mean. Never computed per subject."""
    f = f1_score(y, pred, labels=list(classes), average=None, zero_division=0)
    out = {f"f1_class_{c}": float(v) for c, v in zip(classes, f)}
    out["f1_mean"] = float(np.mean(f))
    return out


def subject_metrics(y, pred, subject, f1_classes: Sequence[int] = NON_STAIR) -> list[dict]:
    """One dict per subject: n_windows, accuracy (all windows) and macro-F1 over the classes in
    `f1_classes` that are present in that subject's true labels."""
    y, pred, subject = map(np.asarray, (y, pred, subject))
    rows = []
    for s in sorted(set(subject.tolist())):
        m = subject == s
        present = [c for c in f1_classes if (y[m] == c).any()]
        f1 = f1_score(y[m], pred[m], labels=present, average="macro", zero_division=0) if present else np.nan
        rows.append({"subject": s, "n_windows": int(m.sum()), "accuracy": float(accuracy_score(y[m], pred[m])),
                     "non_stair_macro_f1": float(f1), "n_non_stair_classes_present": len(present)})
    return rows


def confusion(y, pred) -> np.ndarray:
    return confusion_matrix(y, pred, labels=list(range(N_CLASSES)))


def f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 from a (summed) confusion matrix, rows = true, columns = predicted."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)


def evaluate(y, pred, subject) -> dict:
    """Window-level metrics for one set of windows (HARTH only in model choice)."""
    per = subject_metrics(y, pred, subject)
    stairs = pooled_class_f1(y, pred)
    acc = np.array([r["accuracy"] for r in per])
    nsf = np.array([r["non_stair_macro_f1"] for r in per], dtype=float)
    return {"macro_f1_pooled7": pooled_macro_f1(y, pred), "accuracy_pooled": float(accuracy_score(y, pred)),
            "accuracy_subject_mean": float(acc.mean()), "accuracy_subject_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "non_stair_macro_f1_subject_mean": float(np.nanmean(nsf)),
            "non_stair_macro_f1_subject_sd": float(np.nanstd(nsf, ddof=1)) if len(nsf) > 1 else 0.0,
            "stairs_up_f1_pooled": stairs["f1_class_2"], "stairs_down_f1_pooled": stairs["f1_class_3"],
            "stairs_f1_pooled_mean": stairs["f1_mean"], "n_windows": int(len(y)), "n_subjects": len(per),
            "per_subject": per, "confusion": confusion(y, pred)}


# ------------------------------------------------------------------------------------- training

@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 1024) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(dev)
        out.append(torch.softmax(model(xb), dim=1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_CLASSES), np.float32)


def train_model(model: nn.Module, train: Windows, val: Windows, cfg: TrainConfig, seed: int,
                device: str = "cpu", log: Callable[[str], None] = print) -> dict:
    """Adam + weighted cross-entropy + CosineAnnealingLR(T_max=max_epochs); after every epoch the
    pooled 7-class macro-F1 on the validation windows; keeps the best state dict and stops after
    `patience` epochs without improvement. The model is left on `device` in eval mode."""
    require_harth(train, val)
    dev = torch.device(device)
    model.to(dev)
    Xtr = torch.from_numpy(train.X).to(dev)
    ytr = torch.from_numpy(train.y).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train.y).to(dev), label_smoothing=cfg.label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    g = torch.Generator().manual_seed(seed)
    n = len(train)
    best_f1, best_state, best_epoch, bad, history = -1.0, None, -1, 0, []
    t_start = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        total, seen = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if len(idx) < 2:          # BatchNorm cannot train on a batch of one
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        sched.step()
        pred = predict_proba(model, val.X, device).argmax(axis=1)
        f1, acc = pooled_macro_f1(val.y, pred), float(accuracy_score(val.y, pred))
        improved = f1 > best_f1
        if improved:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_macro_f1_pooled7": f1,
                        "val_accuracy": acc, "lr": sched.get_last_lr()[0], "seconds": dt})
        eta = dt * (cfg.max_epochs - epoch - 1)
        log(f"    epoch {epoch + 1:2d}/{cfg.max_epochs}  loss {total / max(seen, 1):.4f}  val F1 {f1:.4f}  "
            f"acc {acc:.4f}{'  *best*' if improved else ''}  {dt:.1f}s  (max ETA {eta / 60:.1f} min)")
        if bad >= cfg.patience:
            log(f"    early stop: no improvement for {cfg.patience} epochs")
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_val_macro_f1_pooled7": best_f1, "epochs_run": len(history),
            "train_seconds": time.perf_counter() - t_start, "history": history}


# -------------------------------------------------------------------------------------- XGBoost

# [HARTH Sec 4.1, p12]: 1024 trees, lambda = 1 (L2), alpha = 0 (L1), no bootstrapping (subsample 1)
# and no feature subsampling, loss "multi-class classification error rate" (merror); the same
# settings with learning rate 0.5 in [Ustad Sec 2.3.3, p6]. Early stopping after 30 rounds without
# improvement and min_child_weight 1 (the xgboost default) are our choices.
XGB_DEFAULTS = {"n_estimators": 1024, "early_stopping_rounds": 30, "subsample": 1.0, "colsample_bytree": 1.0,
                "min_child_weight": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0, "eval_metric": "merror"}


def train_xgboost(train: Windows, val: Windows, params: dict, seed: int, device: str = "cpu",
                  log: Callable[[str], None] = print):
    """Native xgboost API on HARTH-style window features, class-weighted like the deep models (plan;
    [HARTH] does not weight classes for XGB), early stopping on the validation multi-class error rate.
    Returns (booster, info)."""
    import xgboost as xgb

    from .features import window_features

    require_harth(train, val)
    p = {**XGB_DEFAULTS, **params}
    t0 = time.perf_counter()
    Ftr, Fva = window_features(train.X), window_features(val.X)
    w = class_weights(train.y).numpy()[train.y]
    dtr = xgb.DMatrix(Ftr, label=train.y, weight=w)
    dva = xgb.DMatrix(Fva, label=val.y)
    booster_params = {"objective": "multi:softprob", "num_class": N_CLASSES, "eval_metric": p["eval_metric"],
                      "tree_method": "hist", "device": device, "seed": seed, "max_depth": int(p["max_depth"]),
                      "eta": float(p["learning_rate"]), "subsample": p["subsample"],
                      "colsample_bytree": p["colsample_bytree"], "min_child_weight": p["min_child_weight"],
                      "lambda": p["reg_lambda"], "alpha": p["reg_alpha"], "verbosity": 0}
    evals_result = {}
    booster = xgb.train(booster_params, dtr, num_boost_round=int(p["n_estimators"]), evals=[(dva, "val")],
                        early_stopping_rounds=int(p["early_stopping_rounds"]), evals_result=evals_result,
                        verbose_eval=False)
    best_it = int(getattr(booster, "best_iteration", p["n_estimators"] - 1))
    pred = xgb_predict_proba(booster, val.X).argmax(axis=1)
    f1 = pooled_macro_f1(val.y, pred)
    log(f"    xgboost: {best_it + 1} trees (early stopping), val F1 {f1:.4f}, {time.perf_counter() - t0:.1f}s")
    metric = p["eval_metric"]
    history = [{"epoch": i, f"val_{metric}": v} for i, v in enumerate(evals_result["val"][metric])]
    return booster, {"best_epoch": best_it, "best_val_macro_f1_pooled7": f1, "epochs_run": len(history),
                     "train_seconds": time.perf_counter() - t0, "history": history, "booster_params": booster_params}


def xgb_predict_proba(booster, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    from .features import window_features

    best_it = getattr(booster, "best_iteration", None)
    rng = (0, best_it + 1) if best_it is not None else (0, 0)
    return booster.predict(xgb.DMatrix(window_features(X)), iteration_range=rng)
