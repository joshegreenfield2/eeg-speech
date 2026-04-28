"""loader.py — KaraOne dataset loader.

KaraOne structure per subject:
  data/raw/{subject}/
    *.set              — EEGLAB raw EEG (MNE reads directly)
    epoch_inds.mat     — trial indices: clearing_inds, thinking_inds, speaking_inds
    kinect_data/
      labels.txt       — one word/phoneme label per trial (actual presentation order)

Epoch types:
  thinking  → imagined speech (what we want for BCI)
  speaking  → vocalized speech
  clearing  → inter-trial baseline
  stimuli   → visual prompt display (every other speaking index)

Non-EEG channels to drop before any processing:
  EMG  (Kinect color sensor — NOT actual EMG)
  M1, M2  (mastoid references)
  EKG, Trigger (or HEOG, VEOG — varies by subject)

Usage:
  from src.loader import load_subject
  epochs, labels = load_subject("MM05", epoch_type="thinking")
"""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path
from typing import Literal

import mne
import numpy as np
import scipy.io

mne.set_log_level("WARNING")

SUBJECTS = [
    "MM05", "MM08", "MM09", "MM10", "MM11", "MM12",
    "MM14", "MM15", "MM16", "MM18", "MM19", "MM20", "MM21", "P02",
]

# Event IDs matching KaraOne word/phoneme prompts
EVENT_ID = {
    "/n/": 1, "/m/": 2, "/uw/": 3, "/iy/": 4,
    "/diy/": 5, "/tiy/": 6, "/piy/": 7,
    "pat": 8, "pot": 9, "gnaw": 10, "knew": 11,
}

TRIAL_MS = 4900  # milliseconds per trial window

# Channels that are not EEG — drop before any processing
NON_EEG_CHANNELS = {
    "EMG",  # Kinect RGB color sensor (mislabeled)
    "M1", "M2",  # Mastoid references
    "EKG",  # Electrocardiogram
    "Trigger",
    "HEOG", "VEOG",  # Horizontal/vertical EOG (kept separately if used for ICA)
}


EpochType = Literal["thinking", "speaking", "clearing", "stimuli"]


def load_subject(
    subject: str,
    raw_data_dir: str | Path = "data/raw",
    epoch_type: EpochType = "thinking",
    verbose: bool = False,
) -> tuple[mne.EpochsArray, np.ndarray]:
    """Load one subject's imagined speech epochs from KaraOne.

    Returns:
        epochs: mne.EpochsArray, shape (n_trials, n_eeg_channels, n_times)
        labels: np.ndarray of str, shape (n_trials,) — word/phoneme strings
    """
    raw_data_dir = Path(raw_data_dir)
    subject_dir = raw_data_dir / subject

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Subject directory not found: {subject_dir}\n"
            f"Run scripts/download_karaone.py first."
        )

    # 1. Load raw EEG from EEGLAB .set file
    raw = _load_raw(subject_dir, verbose=verbose)

    # 2. Drop non-EEG channels (EMG/Kinect, mastoids, EKG)
    raw = _drop_non_eeg(raw, verbose=verbose)

    # 3. Load trial indices and labels
    epoch_inds, labels = _load_epoch_info(subject_dir, subject)

    # 4. Slice raw data into trial windows
    sfreq = raw.info["sfreq"]
    n_times = int(TRIAL_MS * sfreq / 1000)
    raw_data = raw.get_data() * 1e6  # convert V → µV

    inds_key = f"{epoch_type}_inds"
    if inds_key not in epoch_inds:
        raise ValueError(
            f"Epoch type '{epoch_type}' not found in epoch_inds.mat.\n"
            f"Available keys: {[k for k in epoch_inds if not k.startswith('_')]}"
        )

    trial_inds = epoch_inds[inds_key]

    # Handle "stimuli" (every other speaking index, as in the reference code)
    if epoch_type == "stimuli":
        trial_inds = epoch_inds["speaking_inds"][0::2]

    epoched_data, trial_labels = _slice_epochs(
        raw_data, trial_inds, labels, n_times, epoch_type, verbose=verbose
    )

    # 5. Build MNE EpochsArray with events
    events, event_id = _build_events(trial_labels, trial_inds, epoch_type, epoch_inds)

    info = raw.info.copy()
    epochs = mne.EpochsArray(
        epoched_data,
        info=info,
        events=events,
        event_id=event_id,
        tmin=0.0,
        verbose=verbose,
    )

    if verbose:
        print(f"\n{subject} [{epoch_type}]: {epoched_data.shape} — "
              f"{dict(zip(*np.unique(trial_labels, return_counts=True)))}")

    return epochs, trial_labels


def _load_raw(subject_dir: Path, verbose: bool = False) -> mne.io.BaseRaw:
    set_files = list(subject_dir.glob("*.set"))
    if not set_files:
        raise FileNotFoundError(f"No .set file found in {subject_dir}")
    set_file = str(set_files[0])
    raw = mne.io.read_raw_eeglab(set_file, montage_units="mm", preload=True, verbose=verbose)
    return raw


def _drop_non_eeg(raw: mne.io.BaseRaw, verbose: bool = False) -> mne.io.BaseRaw:
    to_drop = [ch for ch in raw.ch_names if ch in NON_EEG_CHANNELS]
    if to_drop:
        raw.drop_channels(to_drop)
        if verbose:
            print(f"Dropped non-EEG channels: {to_drop}")
    return raw


def _load_epoch_info(
    subject_dir: Path, subject: str
) -> tuple[dict, np.ndarray]:
    epoch_inds_file = subject_dir / "epoch_inds.mat"
    if not epoch_inds_file.exists():
        raise FileNotFoundError(f"epoch_inds.mat not found: {epoch_inds_file}")

    try:
        epoch_inds = scipy.io.loadmat(str(epoch_inds_file))
    except Exception as e:
        raise RuntimeError(
            f"scipy.io.loadmat failed on epoch_inds.mat for {subject}: {e}\n"
            f"Try: import mat73; epoch_inds = mat73.loadmat(str(epoch_inds_file))"
        ) from e

    labels_file = subject_dir / "kinect_data" / "labels.txt"
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.txt not found: {labels_file}")

    with open(labels_file, encoding="utf-8") as f:
        labels = np.array(f.read().splitlines())

    return epoch_inds, labels


def _slice_epochs(
    raw_data: np.ndarray,
    trial_inds: np.ndarray,
    labels: np.ndarray,
    n_times: int,
    epoch_type: str,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice continuous EEG into fixed-length trial windows."""
    # trial_inds shape varies — normalize to list of (start, end) pairs
    if hasattr(trial_inds, "shape"):
        flat_inds = trial_inds.flatten() if trial_inds.ndim > 2 else trial_inds

    n_channels = raw_data.shape[0]
    n_epochs = flat_inds.shape[1] if flat_inds.ndim > 1 else len(flat_inds)

    epoched = np.zeros((n_epochs, n_channels, n_times), dtype=np.float32)
    trial_labels = []

    for i in range(n_epochs):
        try:
            if flat_inds.ndim > 1:
                ind_entry = flat_inds[0][i]
                if hasattr(ind_entry, '__iter__'):
                    start_sample = int(ind_entry[0][0]) - 1  # MATLAB 1-indexed → 0-indexed
                else:
                    start_sample = int(ind_entry) - 1
            else:
                start_sample = int(flat_inds[i]) - 1

            end_sample = start_sample + n_times
            if end_sample <= raw_data.shape[1]:
                epoched[i] = raw_data[:, start_sample:end_sample]
            else:
                # Last epoch may be shorter — zero-pad
                available = raw_data.shape[1] - start_sample
                epoched[i, :, :available] = raw_data[:, start_sample:]

            label_idx = i % len(labels)
            trial_labels.append(labels[label_idx])
        except (IndexError, TypeError) as e:
            if verbose:
                print(f"  Warning: epoch {i} index error: {e}")
            trial_labels.append("unknown")

    return epoched, np.array(trial_labels)


def _build_events(
    trial_labels: np.ndarray,
    trial_inds: np.ndarray,
    epoch_type: str,
    epoch_inds: dict,
) -> tuple[np.ndarray, dict]:
    """Build MNE events array from trial labels."""
    events = []
    used_ids = {}

    for i, label in enumerate(trial_labels):
        event_code = EVENT_ID.get(label, len(EVENT_ID) + 1)
        if label not in used_ids:
            used_ids[label] = event_code
        # Sample onset — use trial index i as a proxy sample position
        events.append([i * 100, 0, event_code])

    return np.array(events, dtype=int), {k: v for k, v in EVENT_ID.items() if k in used_ids}


def get_subject_dir(subject: str, raw_data_dir: str | Path = "data/raw") -> Path:
    return Path(raw_data_dir) / subject


def list_available_subjects(raw_data_dir: str | Path = "data/raw") -> list[str]:
    raw_data_dir = Path(raw_data_dir)
    return sorted([
        s for s in SUBJECTS
        if (raw_data_dir / s).exists() and list((raw_data_dir / s).glob("*.set"))
    ])
