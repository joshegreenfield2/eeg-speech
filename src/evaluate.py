"""evaluate.py — aggregation, persistence, and visualization for classifier runs.

- `save_run()` writes a single CVResult to outputs/results/<run_id>/...
- `load_runs()` aggregates across runs into a pandas DataFrame
- `plot_confusion_matrix()`, `plot_per_subject_accuracy()` for visualizations
- `plot_topomap_band_importance()` projects feature importances onto the scalp
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.classifier import CVResult


def save_run(
    result: CVResult,
    out_dir: str | Path = "outputs/results",
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a CVResult + extra metadata to JSON. Returns the directory path."""
    out_dir = Path(out_dir)
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = asdict(result)
    if extra:
        payload["extra"] = extra
    payload["timestamp"] = datetime.now().isoformat(timespec="seconds")

    out_file = run_dir / f"{result.subject}_{result.task}.json"
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    return run_dir


def load_runs(
    out_dir: str | Path = "outputs/results",
    pattern: str = "**/*.json",
) -> pd.DataFrame:
    """Aggregate all run JSONs into one DataFrame for cross-run analysis."""
    out_dir = Path(out_dir)
    rows = []
    for f in out_dir.glob(pattern):
        try:
            data = json.loads(f.read_text())
            rows.append({
                "run_id": f.parent.name,
                "subject": data.get("subject"),
                "task": data.get("task"),
                "n_classes": data.get("n_classes"),
                "mean_accuracy": data.get("mean_accuracy"),
                "std_accuracy": data.get("std_accuracy"),
                "chance_level": data.get("chance_level"),
                "f1_macro": data.get("f1_macro"),
                "perm_pvalue": data.get("permutation_pvalue"),
                "n_train": data.get("n_train_total"),
                "n_test": data.get("n_test_total"),
                "timestamp": data.get("timestamp"),
                "file": str(f),
            })
        except Exception as e:
            print(f"  skipping {f}: {e}")
    return pd.DataFrame(rows)


def summary_csv(
    out_dir: str | Path = "outputs/results",
    csv_path: str | Path = "outputs/summary_results.csv",
) -> pd.DataFrame:
    """Build summary CSV across all saved runs."""
    df = load_runs(out_dir)
    if df.empty:
        print(f"No runs found in {out_dir}")
        return df
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    classes: list[str],
    title: str = "Confusion matrix",
    save_path: str | Path | None = None,
):
    """Plot a confusion matrix; returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    import matplotlib

    cm = np.asarray(cm, dtype=int)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.6), max(5, len(classes) * 0.55)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells with normalized fraction
    threshold = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_per_subject_accuracy(
    df: pd.DataFrame,
    task: str = "11-class",
    save_path: str | Path | None = None,
):
    """Bar plot of per-subject mean accuracy with error bars + chance line."""
    import matplotlib.pyplot as plt

    task_df = df[df["task"] == task].sort_values("subject")
    if task_df.empty:
        print(f"No runs for task '{task}'")
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(task_df) * 0.6), 4))
    x = np.arange(len(task_df))
    bars = ax.bar(x, task_df["mean_accuracy"], yerr=task_df["std_accuracy"],
                  capsize=4, color="steelblue", edgecolor="black")
    chance = task_df["chance_level"].iloc[0]
    ax.axhline(chance, ls="--", color="red", label=f"Chance ({chance:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(task_df["subject"], rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-subject accuracy — {task}")
    ax.set_ylim(0, max(0.5, (task_df["mean_accuracy"] + task_df["std_accuracy"]).max() * 1.15))
    ax.legend()

    for bar, acc in zip(bars, task_df["mean_accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
