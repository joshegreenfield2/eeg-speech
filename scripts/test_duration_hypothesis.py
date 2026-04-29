#!/usr/bin/env python3
"""test_duration_hypothesis.py — does SVM's phonetic-kind clustering survive epoch cropping?

Hypothesis: SVM's strong "kind clustering" in confusion matrices (39% of errors land
within the same vowel/consonant/CV-syllable/CVC-word category vs 20% chance) might
come from utterance-duration signal — CVC words occupy more of the 4900ms epoch
than single phonemes, and Welch-averaged band-power picks up this envelope.

Test: re-run SVM on cropped epoch windows.
  full     0.0 → 4.9 s   (baseline — current pipeline)
  early    0.0 → 1.5 s   (motor prep, before most utterances finish)
  late     2.5 → 4.9 s   (only longer utterances are still active here)

If duration is the signal:
  - early window → kind-clustering drops toward chance, accuracy drops
  - late  window → kind-clustering stays high (longer utterances stand out)
If something else is the signal:
  - both windows preserve kind-clustering at similar levels

Runs on 3 representative subjects (MM05, MM12, MM18) for speed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import cross_val_classify
from src.features import DEFAULT_BANDS, band_power_features
from src.loader import EVENT_ID
from src.preprocessor import load_clean_epochs

# Reuse linguistic groupings from analyze_confusions
LINGUISTIC = {
    "/iy/":   ("vowel",      "front",     True),
    "/uw/":   ("vowel",      "back",      True),
    "/m/":    ("consonant",  "bilabial",  True),
    "/n/":    ("consonant",  "alveolar",  True),
    "/piy/":  ("CV-syllable","bilabial",  False),
    "/tiy/":  ("CV-syllable","alveolar",  False),
    "/diy/":  ("CV-syllable","alveolar",  True),
    "pat":    ("CVC-word",   "bilabial",  False),
    "pot":    ("CVC-word",   "bilabial",  False),
    "knew":   ("CVC-word",   "alveolar",  True),
    "gnaw":   ("CVC-word",   "alveolar",  True),
}

WINDOWS = {
    "full":  (None, None),
    "early": (0.0, 1.5),
    "late":  (2.5, None),
}

SUBJECTS = ["MM05", "MM12", "MM18"]


def labels_from_epochs(epochs: mne.EpochsArray) -> np.ndarray:
    inv = {v: k for k, v in EVENT_ID.items()}
    return np.array([inv[e] for e in epochs.events[:, 2]])


def kind_clustering_pct(cm: list[list[int]], classes: list[str]) -> tuple[float, float]:
    """Returns (same-kind error rate, chance baseline). Mirrors analyze_confusions logic."""
    cm = np.asarray(cm, dtype=float)
    # Normalize per-row (per true class)
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    cm_n = cm / rs

    same_mass = 0.0
    diff_mass = 0.0
    same_count = 0
    diff_count = 0
    for i, c_true in enumerate(classes):
        if c_true not in LINGUISTIC:
            continue
        true_kind = LINGUISTIC[c_true][0]
        for j, c_pred in enumerate(classes):
            if i == j or c_pred not in LINGUISTIC:
                continue
            pred_kind = LINGUISTIC[c_pred][0]
            mass = cm_n[i, j]
            if true_kind == pred_kind:
                same_mass += mass
                same_count += 1
            else:
                diff_mass += mass
                diff_count += 1
    total_mass = same_mass + diff_mass
    same_pct = 100 * same_mass / total_mass if total_mass > 0 else 0
    chance_pct = 100 * same_count / (same_count + diff_count) if (same_count + diff_count) > 0 else 0
    return same_pct, chance_pct


def run_one(subject: str, window_name: str, tmin: float | None, tmax: float | None) -> dict:
    epochs, _ = load_clean_epochs(subject)
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    labels = labels_from_epochs(epochs)
    # n_fft must be ≤ n_times. Pick the largest power of 2 that fits.
    n_times = epochs.get_data().shape[2]
    n_fft = 1 << (n_times.bit_length() - 1)  # largest pow2 ≤ n_times
    n_fft = min(n_fft, 2048)
    n_overlap = n_fft // 4
    feats = band_power_features(epochs, labels, bands=DEFAULT_BANDS, n_fft=n_fft, n_overlap=n_overlap)
    t0 = time.time()
    res = cross_val_classify(feats, subject=subject, task=f"11-class_{window_name}", cv_splits=5)
    dur = time.time() - t0

    same_pct, chance_pct = kind_clustering_pct(res.confusion_matrix, res.classes)

    return {
        "subject": subject,
        "window": window_name,
        "tmin": tmin,
        "tmax": tmax,
        "n_epochs": len(epochs),
        "n_classes": res.n_classes,
        "mean_acc": res.mean_accuracy,
        "mean_bal_acc": res.mean_balanced_accuracy,
        "kind_same_pct": same_pct,
        "kind_chance_pct": chance_pct,
        "kind_above_chance": same_pct - chance_pct,
        "duration_s": dur,
    }


def main():
    rows = []
    for subject in SUBJECTS:
        print(f"\n=== {subject} ===")
        for win_name, (tmin, tmax) in WINDOWS.items():
            r = run_one(subject, win_name, tmin, tmax)
            rows.append(r)
            print(f"  {win_name:5s} ({str(tmin):>4}–{str(tmax):>4}s): "
                  f"acc={r['mean_acc']:.3f}  bal={r['mean_bal_acc']:.3f}  "
                  f"kind={r['kind_same_pct']:.1f}% (chance {r['kind_chance_pct']:.1f}%, "
                  f"delta {r['kind_above_chance']:+.1f})  [{r['duration_s']:.0f}s]")

    # Aggregate across subjects
    print("\n=== Cross-subject means ===")
    print(f"{'window':6s} {'mean_acc':>10s} {'mean_bal':>10s} {'kind_same':>11s} {'kind_chance':>13s} {'kind_delta':>11s}")
    print("-" * 64)
    for win in WINDOWS:
        sub = [r for r in rows if r["window"] == win]
        m_acc = np.mean([r["mean_acc"] for r in sub])
        m_bal = np.mean([r["mean_bal_acc"] for r in sub])
        m_kind = np.mean([r["kind_same_pct"] for r in sub])
        m_chance = np.mean([r["kind_chance_pct"] for r in sub])
        m_delta = m_kind - m_chance
        print(f"{win:6s} {m_acc:>10.3f} {m_bal:>10.3f} {m_kind:>10.1f}% {m_chance:>12.1f}% {m_delta:>+10.1f}")

    out = Path("outputs/results/duration_hypothesis_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
