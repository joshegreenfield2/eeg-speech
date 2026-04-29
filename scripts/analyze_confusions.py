#!/usr/bin/env python3
"""analyze_confusions.py — phoneme confusion analysis across EEGNet (and SVM) runs.

Aggregates per-subject confusion matrices into a cross-subject view. Then asks:
  - Which phoneme pairs does the model confuse most?
  - Are vowels easier than consonants?
  - Does place-of-articulation (bilabial/alveolar/velar) cluster confusions?
  - Does voicing matter?
  - Do EEGNet and SVM confuse the same things, or different things?

Outputs:
  outputs/figures/confusion_eegnet_avg.png
  outputs/figures/confusion_svm_avg.png
  outputs/figures/top_confusion_pairs.png
  outputs/figures/per_class_accuracy.png
  outputs/results/confusion_analysis.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Linguistic groupings for the 11 KaraOne classes
LINGUISTIC = {
    # phoneme/word: (kind, place, voiced)
    "/iy/":   ("vowel",      "front",     True),
    "/uw/":   ("vowel",      "back",      True),
    "/m/":    ("consonant",  "bilabial",  True),
    "/n/":    ("consonant",  "alveolar",  True),
    "/piy/":  ("CV-syllable","bilabial",  False),
    "/tiy/":  ("CV-syllable","alveolar",  False),
    "/diy/":  ("CV-syllable","alveolar",  True),
    "pat":    ("CVC-word",   "bilabial",  False),  # /p/ initial, voiceless onset
    "pot":    ("CVC-word",   "bilabial",  False),
    "knew":   ("CVC-word",   "alveolar",  True),   # /n/ initial (silent k), voiced onset
    "gnaw":   ("CVC-word",   "alveolar",  True),   # /n/ initial (silent g)
}


def load_eegnet_results(results_dir: Path) -> list[dict]:
    """Load all EEGNet 11-class JSON files (only the 11-class run, skip MM08's 7-class)."""
    runs = []
    for f in sorted(results_dir.glob("phase4_dl_*/[A-Z]*_11-class_eegnet.json")):
        d = json.loads(f.read_text())
        runs.append(d)
    return runs


def load_svm_results(results_dir: Path) -> list[dict]:
    """Load all SVM 11-class JSON files."""
    runs = []
    for f in sorted(results_dir.glob("phase3_batch_*/[A-Z]*_11-class.json")):
        d = json.loads(f.read_text())
        runs.append(d)
    return runs


def average_confusion(runs: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Average confusion matrices, normalized per-row, across runs that share the same class set."""
    # Filter to only runs with the full 11 classes
    full_runs = [r for r in runs if r["n_classes"] == 11]
    if not full_runs:
        raise SystemExit("No 11-class runs found")
    classes = full_runs[0]["classes"]

    cms_norm = []
    for r in full_runs:
        if r["classes"] != classes:
            continue  # skip mismatched class orderings
        cm = np.array(r["confusion_matrix"], dtype=float)
        # Normalize per true-class row
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cms_norm.append(cm / row_sums)

    avg = np.mean(cms_norm, axis=0)
    return avg, classes


def per_class_accuracy(cm: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Diagonal of normalized cm = recall per class."""
    return pd.DataFrame({"class": classes, "accuracy": np.diag(cm)}).sort_values("accuracy", ascending=False)


def top_confusions(cm: np.ndarray, classes: list[str], k: int = 10) -> pd.DataFrame:
    """Top-k off-diagonal cells (strongest confusions)."""
    n = len(classes)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append({"true": classes[i], "predicted": classes[j], "fraction": cm[i, j]})
    return pd.DataFrame(pairs).sort_values("fraction", ascending=False).head(k).reset_index(drop=True)


def grouped_metric(cm: np.ndarray, classes: list[str], grouping: str) -> dict:
    """Test if a linguistic grouping (kind/place/voiced) helps. Returns within vs across rates."""
    idx = {c: i for i, c in enumerate(classes)}
    group_keys = ["kind", "place", "voiced"].index(grouping)

    same_group_mass = 0.0
    diff_group_mass = 0.0
    for c_true in classes:
        if c_true not in LINGUISTIC:
            continue
        i = idx[c_true]
        true_group = LINGUISTIC[c_true][group_keys]
        for c_pred in classes:
            if c_pred == c_true or c_pred not in LINGUISTIC:
                continue
            j = idx[c_pred]
            pred_group = LINGUISTIC[c_pred][group_keys]
            mass = cm[i, j]
            if true_group == pred_group:
                same_group_mass += mass
            else:
                diff_group_mass += mass

    total = same_group_mass + diff_group_mass
    same_pct = 100 * same_group_mass / total if total > 0 else 0
    return {
        "grouping": grouping,
        "same_group_confusion_rate": same_pct,
        "diff_group_confusion_rate": 100 - same_pct,
        "interpretation": (
            f"Of all errors, {same_pct:.1f}% land within the same {grouping}, "
            f"{100-same_pct:.1f}% across — chance for a uniform mistake distribution depends on group sizes (see below)."
        ),
    }


def chance_for_grouping(classes: list[str], grouping: str) -> float:
    """Baseline: if errors were uniformly distributed across non-self classes, what % would land in the same group?"""
    group_keys = ["kind", "place", "voiced"].index(grouping)
    same = 0
    diff = 0
    for c_true in classes:
        if c_true not in LINGUISTIC:
            continue
        true_group = LINGUISTIC[c_true][group_keys]
        for c_pred in classes:
            if c_pred == c_true or c_pred not in LINGUISTIC:
                continue
            pred_group = LINGUISTIC[c_pred][group_keys]
            if true_group == pred_group:
                same += 1
            else:
                diff += 1
    total = same + diff
    return 100 * same / total if total > 0 else 0


def render_confusion_heatmap(cm: np.ndarray, classes: list[str], title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(0.5, cm.max()))
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > 0.3 else "black"
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_top_pairs(top_eegnet: pd.DataFrame, top_svm: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, title in [(axes[0], top_eegnet, "EEGNet"), (axes[1], top_svm, "SVM")]:
        labels = [f"{r['true']} → {r['predicted']}" for _, r in df.iterrows()]
        ax.barh(range(len(df)), df["fraction"][::-1], color="steelblue")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels[::-1])
        ax.set_xlabel("Fraction of true class predicted as other")
        ax.set_title(f"{title} — top 10 confusions (avg across subjects)")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_per_class_accuracy(eegnet_acc: pd.DataFrame, svm_acc: pd.DataFrame, save_path: Path):
    """Side-by-side per-class accuracy comparison."""
    merged = pd.merge(
        eegnet_acc.rename(columns={"accuracy": "EEGNet"}),
        svm_acc.rename(columns={"accuracy": "SVM"}),
        on="class",
    ).sort_values("EEGNet", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(merged))
    ax.barh(y - 0.2, merged["EEGNet"], 0.4, label="EEGNet", color="steelblue")
    ax.barh(y + 0.2, merged["SVM"], 0.4, label="SVM", color="orange")
    ax.set_yticks(y)
    ax.set_yticklabels(merged["class"])
    ax.axvline(1 / 11, ls="--", color="red", alpha=0.6, label="Chance (9.1%)")
    ax.set_xlabel("Per-class recall (avg across subjects)")
    ax.set_title("Per-phoneme accuracy — EEGNet vs SVM")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    results_dir = Path("outputs/results")
    fig_dir = Path("outputs/figures")

    eeg_runs = load_eegnet_results(results_dir)
    svm_runs = load_svm_results(results_dir)
    print(f"Loaded {len(eeg_runs)} EEGNet runs, {len(svm_runs)} SVM runs")

    eeg_cm, classes = average_confusion(eeg_runs)
    svm_cm, _ = average_confusion(svm_runs)

    eeg_acc = per_class_accuracy(eeg_cm, classes)
    svm_acc = per_class_accuracy(svm_cm, classes)
    print("\n=== Per-class accuracy ===")
    print(eeg_acc.merge(svm_acc, on="class", suffixes=("_eegnet", "_svm")).round(3))

    eeg_top = top_confusions(eeg_cm, classes, k=10)
    svm_top = top_confusions(svm_cm, classes, k=10)
    print("\n=== Top 10 EEGNet confusions ===")
    print(eeg_top.round(3))
    print("\n=== Top 10 SVM confusions ===")
    print(svm_top.round(3))

    print("\n=== Linguistic grouping tests (EEGNet) ===")
    eeg_groupings = []
    for g in ["kind", "place", "voiced"]:
        gm = grouped_metric(eeg_cm, classes, g)
        chance = chance_for_grouping(classes, g)
        gm["chance_pct_if_uniform"] = chance
        gm["above_chance_by"] = gm["same_group_confusion_rate"] - chance
        eeg_groupings.append(gm)
        print(f"  {g:8s}: {gm['same_group_confusion_rate']:.1f}% same-group "
              f"(chance {chance:.1f}%, delta {gm['above_chance_by']:+.1f})")

    print("\n=== Linguistic grouping tests (SVM) ===")
    svm_groupings = []
    for g in ["kind", "place", "voiced"]:
        gm = grouped_metric(svm_cm, classes, g)
        chance = chance_for_grouping(classes, g)
        gm["chance_pct_if_uniform"] = chance
        gm["above_chance_by"] = gm["same_group_confusion_rate"] - chance
        svm_groupings.append(gm)
        print(f"  {g:8s}: {gm['same_group_confusion_rate']:.1f}% same-group "
              f"(chance {chance:.1f}%, delta {gm['above_chance_by']:+.1f})")

    render_confusion_heatmap(eeg_cm, classes, "EEGNet — avg confusion (row-normalized) across 13 subjects",
                             fig_dir / "confusion_eegnet_avg.png")
    render_confusion_heatmap(svm_cm, classes, "SVM — avg confusion (row-normalized) across 13 subjects",
                             fig_dir / "confusion_svm_avg.png")
    render_top_pairs(eeg_top, svm_top, fig_dir / "top_confusion_pairs.png")
    render_per_class_accuracy(eeg_acc, svm_acc, fig_dir / "per_class_accuracy.png")

    out = {
        "n_eegnet_runs": len(eeg_runs),
        "n_svm_runs": len(svm_runs),
        "classes": classes,
        "eegnet_per_class_accuracy": eeg_acc.to_dict(orient="records"),
        "svm_per_class_accuracy": svm_acc.to_dict(orient="records"),
        "eegnet_top_confusions": eeg_top.to_dict(orient="records"),
        "svm_top_confusions": svm_top.to_dict(orient="records"),
        "eegnet_linguistic_groupings": eeg_groupings,
        "svm_linguistic_groupings": svm_groupings,
    }
    out_path = results_dir / "confusion_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")
    print(f"Figures: {fig_dir}/confusion_*.png  + top_confusion_pairs.png + per_class_accuracy.png")


if __name__ == "__main__":
    main()
