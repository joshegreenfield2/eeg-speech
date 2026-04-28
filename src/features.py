"""features.py — feature extraction for EEG classification.

Two feature paths supported:

1. **Band-power features** (`band_power_features`)
   Welch PSD → integrate within each band → log-transform.
   Standard pre-deep-learning baseline. ~62 channels × 5 bands = 310 features.

2. **DWT features** (`dwt_features`)
   Daubechies-4 wavelet decomposition → statistical features per sub-band.
   The KaraOne 57% SOTA paper used this. (Phase 4+)

Sklearn-compatible: fit/transform return shape (n_epochs, n_features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mne
import numpy as np
from mne.time_frequency import psd_array_welch


# Frequency bands (Hz) — standard EEG ranges, omitting gamma (>30 Hz) per research
# warning that high-gamma is usually EMG artifact, not brain signal.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "low_beta": (12.0, 20.0),
    "high_beta": (20.0, 30.0),
}


@dataclass
class FeatureSet:
    """Output of a feature extractor."""
    X: np.ndarray              # (n_epochs, n_features)
    y: np.ndarray              # (n_epochs,) class labels
    feature_names: list[str]   # length n_features
    channel_names: list[str]   # length n_channels (used)
    bands: dict[str, tuple[float, float]] | None = None


def band_power_features(
    epochs: mne.EpochsArray,
    labels: np.ndarray,
    bands: dict[str, tuple[float, float]] = DEFAULT_BANDS,
    use_good_channels_only: bool = True,
    log_transform: bool = True,
    n_fft: int = 2048,
    n_overlap: int = 512,
) -> FeatureSet:
    """Compute Welch PSD band-power features per channel per band.

    Returns features shaped (n_epochs, n_channels * n_bands), flattened in
    `[ch1_band1, ch1_band2, ..., chN_bandK]` order.
    """
    sfreq = epochs.info["sfreq"]

    if use_good_channels_only:
        ch_idx = [
            i for i, ch in enumerate(epochs.ch_names)
            if ch not in epochs.info["bads"]
        ]
    else:
        ch_idx = list(range(len(epochs.ch_names)))

    if not ch_idx:
        raise ValueError("No good channels available for feature extraction")

    channel_names = [epochs.ch_names[i] for i in ch_idx]
    data = epochs.get_data()[:, ch_idx, :]  # (n_epochs, n_good_ch, n_times)

    # Compute PSD across all freqs of interest
    fmin = min(b[0] for b in bands.values())
    fmax = max(b[1] for b in bands.values())

    psd, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax,
        n_fft=n_fft, n_overlap=n_overlap,
        verbose=False,
    )  # (n_epochs, n_ch, n_freqs)

    # Integrate within each band
    n_epochs = psd.shape[0]
    n_ch = psd.shape[1]
    n_bands = len(bands)
    band_power = np.zeros((n_epochs, n_ch, n_bands), dtype=np.float32)
    feature_names: list[str] = []

    for b_idx, (band_name, (lo, hi)) in enumerate(bands.items()):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            raise ValueError(
                f"Band {band_name} ({lo}-{hi} Hz) has no PSD bins; "
                f"check fmin/fmax or n_fft (current freqs: {freqs[0]:.1f}-{freqs[-1]:.1f})"
            )
        # Mean power within band (more stable than sum across variable bin counts)
        band_power[:, :, b_idx] = psd[:, :, mask].mean(axis=2)
        for ch in channel_names:
            feature_names.append(f"{ch}_{band_name}")

    if log_transform:
        # Log-transform power features (standard EEG practice — power is log-normal)
        band_power = np.log(band_power + 1e-12)

    # Flatten (n_epochs, n_ch, n_bands) → (n_epochs, n_ch * n_bands) in [ch][band] order
    X = band_power.reshape(n_epochs, n_ch * n_bands).astype(np.float32)
    y = np.asarray(labels)

    return FeatureSet(
        X=X, y=y,
        feature_names=feature_names,
        channel_names=channel_names,
        bands=bands,
    )


def dwt_features(
    epochs: mne.EpochsArray,
    labels: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
    use_good_channels_only: bool = True,
    statistics: tuple[str, ...] = ("mean", "std", "energy", "skew", "kurtosis"),
) -> FeatureSet:
    """Discrete Wavelet Transform features (Daubechies-4 by default).

    Per channel, decomposes signal into `level+1` sub-bands and computes
    statistical features per sub-band. This is the path used by the KaraOne
    57% SOTA paper (Saha et al. 2020).

    Output shape: (n_epochs, n_channels * (level+1) * n_statistics).
    """
    try:
        import pywt
    except ImportError as e:
        raise ImportError("PyWavelets required: `uv add PyWavelets`") from e

    if use_good_channels_only:
        ch_idx = [
            i for i, ch in enumerate(epochs.ch_names)
            if ch not in epochs.info["bads"]
        ]
    else:
        ch_idx = list(range(len(epochs.ch_names)))

    channel_names = [epochs.ch_names[i] for i in ch_idx]
    data = epochs.get_data()[:, ch_idx, :]
    n_epochs, n_ch, _ = data.shape

    # Decompose one signal first to get the number of sub-bands
    sample_coeffs = pywt.wavedec(data[0, 0, :], wavelet=wavelet, level=level)
    n_subbands = len(sample_coeffs)  # = level + 1

    feat_per_subband = len(statistics)
    total_features = n_ch * n_subbands * feat_per_subband
    X = np.zeros((n_epochs, total_features), dtype=np.float32)

    feature_names: list[str] = []
    for ch in channel_names:
        for sb_idx in range(n_subbands):
            sb_label = f"cA{level}" if sb_idx == 0 else f"cD{level - sb_idx + 1}"
            for stat in statistics:
                feature_names.append(f"{ch}_{sb_label}_{stat}")

    for e_idx in range(n_epochs):
        feat_idx = 0
        for c_idx in range(n_ch):
            coeffs = pywt.wavedec(data[e_idx, c_idx, :], wavelet=wavelet, level=level)
            for sb in coeffs:
                for stat in statistics:
                    X[e_idx, feat_idx] = _stat(sb, stat)
                    feat_idx += 1

    return FeatureSet(
        X=X, y=np.asarray(labels),
        feature_names=feature_names,
        channel_names=channel_names,
        bands=None,
    )


def _stat(arr: np.ndarray, name: Literal["mean", "std", "energy", "skew", "kurtosis"]) -> float:
    if name == "mean":
        return float(arr.mean())
    if name == "std":
        return float(arr.std())
    if name == "energy":
        return float(np.sum(arr ** 2))
    if name == "skew":
        # Manual skewness — avoids scipy dep
        m = arr.mean()
        s = arr.std() + 1e-12
        return float(np.mean(((arr - m) / s) ** 3))
    if name == "kurtosis":
        m = arr.mean()
        s = arr.std() + 1e-12
        return float(np.mean(((arr - m) / s) ** 4) - 3.0)
    raise ValueError(f"Unknown statistic: {name}")
