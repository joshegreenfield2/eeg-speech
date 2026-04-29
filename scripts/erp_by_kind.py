#!/usr/bin/env python3
"""erp_by_kind.py — grand-average ERP at central electrodes, grouped by phonetic kind.

Hypothesis from the confusion analysis: phonetic-kind information (vowel /
consonant / CV-syllable / CVC-word) is encoded in the first ~1.5 s of each
trial. If true, we should see visibly different ERPs for the four kinds at
motor-cortex electrodes (Cz, FCz, C3, C4) during that window.

Outputs:
  outputs/figures/erp_by_kind_central.png  — 2×2 panel, one electrode each
  outputs/figures/erp_by_kind_cz_only.png  — single Cz panel for clarity
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loader import EVENT_ID
from src.preprocessor import load_clean_epochs

LINGUISTIC_KIND = {
    "/iy/": "vowel", "/uw/": "vowel",
    "/m/": "consonant", "/n/": "consonant",
    "/piy/": "CV-syllable", "/tiy/": "CV-syllable", "/diy/": "CV-syllable",
    "pat": "CVC-word", "pot": "CVC-word", "knew": "CVC-word", "gnaw": "CVC-word",
}
KIND_ORDER = ["vowel", "consonant", "CV-syllable", "CVC-word"]
KIND_COLORS = {
    "vowel":       "#1f77b4",  # blue
    "consonant":   "#ff7f0e",  # orange
    "CV-syllable": "#2ca02c",  # green
    "CVC-word":    "#d62728",  # red
}

CENTRAL_PICKS = ["CZ", "FCZ", "C3", "C4"]  # KaraOne montage uses uppercase
SUBJECTS = [
    "MM05", "MM08", "MM09", "MM10", "MM11", "MM12",
    "MM14", "MM15", "MM16", "MM18", "MM19", "MM20", "MM21", "P02",
]


def labels_from_epochs(epochs):
    inv = {v: k for k, v in EVENT_ID.items()}
    return np.array([inv[e] for e in epochs.events[:, 2]])


def collect_per_subject_kind_avg(subject: str, picks: list[str], tmax: float = 1.5):
    """Return {kind: {channel: avg_waveform_microvolts}, n_trials_per_kind, sfreq, n_times}."""
    epochs, _ = load_clean_epochs(subject)
    epochs = epochs.copy().crop(tmin=0.0, tmax=tmax)
    sfreq = epochs.info["sfreq"]

    # Filter picks to channels actually present (some are bad/dropped per-subject)
    available = [p for p in picks if p in epochs.ch_names]
    if not available:
        return None
    epochs = epochs.copy().pick(available)

    labels = labels_from_epochs(epochs)
    kinds = np.array([LINGUISTIC_KIND.get(l, "?") for l in labels])

    out = {}
    n_trials_per_kind = {}
    for kind in KIND_ORDER:
        mask = kinds == kind
        if not mask.any():
            continue
        # Get_data: (n_epochs, n_ch, n_times) in volts
        kind_data = epochs.get_data(copy=False)[mask]  # (n, n_ch, n_times)
        kind_avg = kind_data.mean(axis=0)  # (n_ch, n_times)
        out[kind] = {ch: kind_avg[i] * 1e6 for i, ch in enumerate(epochs.ch_names)}  # to µV
        n_trials_per_kind[kind] = int(mask.sum())

    return {
        "subject": subject,
        "kind_avg": out,
        "n_trials_per_kind": n_trials_per_kind,
        "channels_used": list(epochs.ch_names),
        "sfreq": sfreq,
        "n_times": kind_data.shape[2],
    }


def grand_average_across_subjects(per_subject_results: list[dict]):
    """Average each kind's waveform across subjects, separately per electrode."""
    # Gather: {kind: {channel: list_of_per_subject_avgs}}
    pooled = defaultdict(lambda: defaultdict(list))
    sfreq = None
    n_times = None
    for r in per_subject_results:
        if r is None:
            continue
        sfreq = r["sfreq"]
        n_times = r["n_times"]
        for kind, ch_avgs in r["kind_avg"].items():
            for ch, wave in ch_avgs.items():
                pooled[kind][ch].append(wave)

    grand = {}
    for kind, ch_dict in pooled.items():
        grand[kind] = {}
        for ch, waves in ch_dict.items():
            arr = np.array(waves)  # (n_subjects_with_this_ch, n_times)
            grand[kind][ch] = {
                "mean": arr.mean(axis=0),
                "sem":  arr.std(axis=0) / np.sqrt(arr.shape[0]),
                "n_subjects": arr.shape[0],
            }
    return grand, sfreq, n_times


def render_panel(grand, sfreq, n_times, picks, save_path: Path, title_suffix=""):
    times = np.linspace(0, n_times / sfreq, n_times)

    n_picks = len(picks)
    n_cols = 2 if n_picks > 1 else 1
    n_rows = (n_picks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows), squeeze=False)

    for k, ch in enumerate(picks):
        ax = axes[k // n_cols][k % n_cols]
        for kind in KIND_ORDER:
            if kind not in grand or ch not in grand[kind]:
                continue
            d = grand[kind][ch]
            ax.plot(times, d["mean"], label=f'{kind} (n={d["n_subjects"]})',
                    color=KIND_COLORS[kind], lw=1.6)
            ax.fill_between(times, d["mean"] - d["sem"], d["mean"] + d["sem"],
                            color=KIND_COLORS[kind], alpha=0.15, linewidth=0)
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.axvline(0, color="black", lw=0.5, alpha=0.3)
        ax.set_xlabel("Time from trial start (s)")
        ax.set_ylabel("µV (grand-average)")
        ax.set_title(f"{ch}{title_suffix}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)

    # Hide unused axes
    for k in range(n_picks, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle("Grand-average ERP by phonetic kind — 0–1.5s window")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


def report_kind_separation(grand, sfreq, n_times, ch="Cz"):
    """Quick numeric summary: max pairwise amplitude difference across kinds, per electrode."""
    times = np.linspace(0, n_times / sfreq, n_times)
    print(f"\n=== Kind separation at {ch} ===")
    if ch not in grand.get("vowel", {}):
        print(f"  {ch} not present in any kind")
        return

    waves = {kind: grand[kind][ch]["mean"] for kind in KIND_ORDER if kind in grand and ch in grand[kind]}
    print(f"  Time-averaged absolute amplitude (µV) per kind:")
    for kind, w in waves.items():
        print(f"    {kind:12s}: mean={w.mean():+.3f}  std={w.std():.3f}  peak_t={times[np.argmax(np.abs(w))]:.3f}s")

    # Largest pairwise RMS difference
    kinds = list(waves)
    print(f"  Pairwise RMS difference between kind waveforms (µV):")
    for i, k1 in enumerate(kinds):
        for k2 in kinds[i+1:]:
            rms = float(np.sqrt(np.mean((waves[k1] - waves[k2]) ** 2)))
            print(f"    {k1:12s} vs {k2:12s}: {rms:.3f}")


def main():
    fig_dir = Path("outputs/figures")
    print(f"Loading {len(SUBJECTS)} subjects...")
    per_subject = []
    for s in SUBJECTS:
        try:
            r = collect_per_subject_kind_avg(s, CENTRAL_PICKS)
            if r:
                kinds_with_data = list(r["kind_avg"])
                missing_kinds = [k for k in KIND_ORDER if k not in kinds_with_data]
                missing_chs = [c for c in CENTRAL_PICKS if c not in r["channels_used"]]
                msg = f"  {s}: {len(r['channels_used'])} chs"
                if missing_chs:
                    msg += f" (missing {missing_chs})"
                if missing_kinds:
                    msg += f", {len(kinds_with_data)}/4 kinds (missing {missing_kinds})"
                print(msg)
                per_subject.append(r)
            else:
                print(f"  {s}: SKIP (no central electrodes available)")
        except Exception as e:
            print(f"  {s}: ERROR {e}")

    grand, sfreq, n_times = grand_average_across_subjects(per_subject)

    render_panel(grand, sfreq, n_times, CENTRAL_PICKS, fig_dir / "erp_by_kind_central.png")
    render_panel(grand, sfreq, n_times, ["Cz"], fig_dir / "erp_by_kind_cz_only.png", title_suffix=" (motor cortex)")

    for ch in CENTRAL_PICKS:
        report_kind_separation(grand, sfreq, n_times, ch=ch)


if __name__ == "__main__":
    main()
