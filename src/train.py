"""train.py — within-subject training loop for EEGNet / CNN-BiLSTM.

Same StratifiedKFold(5) as the SVM baseline so results are directly comparable
in the summary CSV.

NB: small data, single-subject training. Tens of trials per class. Expect:
  - Heavy overfitting risk → strong dropout, early stopping, weight decay
  - Modest GPU/CPU compute — runs on CPU in minutes per fold
"""

from __future__ import annotations

import time
from typing import Any

import mne
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.classifier import CVResult
from src.models import build_model, count_params

mne.set_log_level("ERROR")


def _epochs_to_tensors(
    epochs: mne.EpochsArray,
    labels: np.ndarray,
    use_good_channels_only: bool = True,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """(n_epochs, n_ch, n_times) float32 tensor + integer labels + channel list."""
    if use_good_channels_only:
        ch_idx = [i for i, ch in enumerate(epochs.ch_names) if ch not in epochs.info["bads"]]
    else:
        ch_idx = list(range(len(epochs.ch_names)))
    if not ch_idx:
        raise ValueError("No good channels available")

    channel_names = [epochs.ch_names[i] for i in ch_idx]
    data = epochs.get_data()[:, ch_idx, :]  # in V

    # Per-channel z-score within trial — stabilizes training across noisy subjects
    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True) + 1e-12
    data = (data - mean) / std

    le = LabelEncoder()
    y = le.fit_transform(labels)
    return torch.from_numpy(data.astype(np.float32)), y, channel_names


def train_one_fold(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs_max: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[nn.Module, dict]:
    """Train one fold with early stopping on val balanced accuracy."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    best_bal = -1.0
    best_state: dict | None = None
    bad_epochs = 0
    history = {"train_loss": [], "val_acc": [], "val_bal": []}

    for epoch in range(epochs_max):
        # ---- train ----
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            logits = model(X_val_d)
            preds = logits.argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val_np, preds)
        val_bal = balanced_accuracy_score(y_val_np, preds)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_bal"].append(val_bal)

        if verbose and (epoch % 10 == 0 or epoch == epochs_max - 1):
            print(f"    ep {epoch:3d}: train_loss={train_loss:.4f}  val_acc={val_acc:.3f}  val_bal={val_bal:.3f}")

        if val_bal > best_bal:
            best_bal = val_bal
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} (best val_bal={best_bal:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def cross_val_train(
    epochs: mne.EpochsArray,
    labels: np.ndarray,
    subject: str,
    model_name: str = "eegnet",
    cv_splits: int = 5,
    epochs_max: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    random_state: int = 42,
    verbose: bool = False,
    device: str | None = None,
) -> CVResult:
    """StratifiedKFold deep-learning CV — same protocol as the SVM baseline.

    Returns a CVResult so results land in the same summary CSV.
    """
    X, y_enc, channel_names = _epochs_to_tensors(epochs, labels)
    le = LabelEncoder()
    le.fit(labels)
    n_classes = len(le.classes_)
    n_channels = X.shape[1]
    n_times = X.shape[2]

    # Auto-detect device if not specified — picks CUDA on Colab, MPS on ARM Mac, else CPU
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if verbose:
        print(f"  [{subject}/{model_name}] X={tuple(X.shape)}, n_classes={n_classes}, n_channels={n_channels}, device={device}")

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    fold_accs: list[float] = []
    fold_bal_accs: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    n_train_total = 0
    n_test_total = 0
    histories = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X.numpy(), y_enc)):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train_t = torch.from_numpy(y_enc[train_idx]).long()
        y_val_t = torch.from_numpy(y_enc[test_idx]).long()

        model = build_model(model_name, n_classes=n_classes, n_channels=n_channels, n_times=n_times)
        if fold_idx == 0 and verbose:
            print(f"  [{subject}/{model_name}] params={count_params(model):,}")

        model, hist = train_one_fold(
            model, X_train, y_train_t, X_val, y_val_t,
            epochs_max=epochs_max, batch_size=batch_size, lr=lr,
            device=device, verbose=False,
        )
        histories.append(hist)

        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(device)).argmax(dim=1).cpu().numpy()
        y_val_np = y_val_t.numpy()
        acc = accuracy_score(y_val_np, preds)
        bal = balanced_accuracy_score(y_val_np, preds)
        fold_accs.append(float(acc))
        fold_bal_accs.append(float(bal))
        all_y_true.extend(y_val_np.tolist())
        all_y_pred.extend(preds.tolist())
        n_train_total += len(y_train_t)
        n_test_total += len(y_val_t)
        if verbose:
            print(f"    fold {fold_idx+1}/{cv_splits}: acc={acc:.3f} bal={bal:.3f}")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(n_classes))).tolist()
    f1m = f1_score(all_y_true, all_y_pred, average="macro")

    return CVResult(
        subject=subject,
        task=f"{n_classes}-class_{model_name}",
        n_classes=n_classes,
        classes=list(le.classes_),
        fold_accuracies=fold_accs,
        mean_accuracy=float(np.mean(fold_accs)),
        std_accuracy=float(np.std(fold_accs)),
        fold_balanced_accuracies=fold_bal_accs,
        mean_balanced_accuracy=float(np.mean(fold_bal_accs)),
        std_balanced_accuracy=float(np.std(fold_bal_accs)),
        chance_level=1.0 / n_classes,
        f1_macro=float(f1m),
        confusion_matrix=cm,
        n_train_total=n_train_total,
        n_test_total=n_test_total,
        permutation_pvalue=None,
        n_permutations=0,
        config={
            "model": model_name, "cv_splits": cv_splits,
            "epochs_max": epochs_max, "batch_size": batch_size, "lr": lr,
            "n_channels": n_channels, "n_times": n_times,
            "random_state": random_state,
        },
    )
