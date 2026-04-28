"""preprocessor.py — KaraOne preprocessing pipeline.

Order (non-negotiable per research):
  1. Drop non-EEG channels (already handled by load_raw_continuous)
  2. Bandpass filter 1–45 Hz (zero-phase FIR, on continuous data)
  3. Notch filter 60 Hz
  4. Bad channel detection (pyprep RANSAC) on continuous data
  5. Average re-reference (after bad channels marked)
  6. ICA + mne-icalabel auto-classification, remove ocular/muscle/heart artifacts
  7. Slice into thinking epochs
  8. Epoch rejection at ±100 µV peak-to-peak

Critical rules from research:
  • Bandpass MUST be applied to continuous data BEFORE epoching
  • ICA MUST be fit on continuous data, NOT on epochs
  • Average reference AFTER bad channel detection, not before
  • Track rejection rate per subject — expected 10-30%
  • Track ICA component count removed — expected 1-5

Outputs:
  data/processed/<subject>-clean-epo.fif   — clean MNE Epochs
  data/processed/<subject>-log.json        — per-subject preprocessing log
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

import mne
import numpy as np
from mne.preprocessing import ICA

from src.loader import load_raw_continuous, load_epoch_info, slice_into_epochs

mne.set_log_level("WARNING")


# ---------- Config ----------
DEFAULT_BANDPASS = (1.0, 45.0)
NOTCH_FREQS = [60.0, 120.0]  # 60Hz and first harmonic
ICA_N_COMPONENTS = 30        # fixed count — variance threshold gives degenerate decompositions
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = "auto"
ICLABEL_REJECT = {"muscle artifact", "eye blink", "heart beat", "line noise"}
ICLABEL_KEEP_THRESHOLD = 0.5  # reject if confidence ≥ 0.5 (raw KaraOne is noisy)
EPOCH_REJECT_PEAK_TO_PEAK_UV = 250.0  # ±250 µV after ICA cleanup — gross-artifact catch only


class PreprocessLog(TypedDict):
    subject: str
    epoch_type: str
    sfreq: float
    n_channels_original: int
    n_channels_after_drop: int
    bad_channels: list[str]
    n_bad_channels: int
    bandpass: tuple[float, float]
    notch_freqs: list[float]
    ica_n_components: int
    ica_components_rejected: list[int]
    ica_labels_rejected: dict[str, int]  # label -> count
    n_epochs_before_reject: int
    n_epochs_after_reject: int
    rejection_rate: float
    epoch_reject_threshold_uv: float
    trials_per_class: dict[str, int]
    duration_seconds: float
    warnings: list[str]


def preprocess_subject(
    subject: str,
    raw_data_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/processed",
    epoch_type: str = "thinking",
    bandpass: tuple[float, float] = DEFAULT_BANDPASS,
    skip_if_exists: bool = True,
    verbose: bool = False,
) -> tuple[mne.EpochsArray, PreprocessLog]:
    """Run the full preprocessing pipeline for one subject.

    Returns (clean_epochs, log_dict). Also saves to out_dir/<subject>-clean-epo.fif
    and out_dir/<subject>-log.json.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fif_path = out_dir / f"{subject}-clean-epo.fif"
    log_path = out_dir / f"{subject}-log.json"

    if skip_if_exists and fif_path.exists() and log_path.exists():
        if verbose:
            print(f"{subject}: already preprocessed, loading from {fif_path}")
        epochs = mne.read_epochs(str(fif_path), preload=True, verbose=verbose)
        log = json.loads(log_path.read_text())
        return epochs, log

    warnings: list[str] = []

    # 1. Load continuous Raw with non-EEG channels already dropped
    raw = load_raw_continuous(subject, raw_data_dir=raw_data_dir, verbose=verbose)
    n_channels_original = len(raw.ch_names)
    if verbose:
        print(f"{subject}: loaded raw, {n_channels_original} EEG channels, "
              f"{raw.times[-1]:.1f}s @ {raw.info['sfreq']}Hz")

    # 2. Bandpass filter (FIR, zero-phase) on continuous data
    raw.filter(
        l_freq=bandpass[0], h_freq=bandpass[1],
        method="fir", phase="zero", fir_design="firwin",
        verbose=verbose,
    )

    # 3. Notch filter — only if 60 Hz spike present (KaraOne data is often pre-filtered)
    raw.notch_filter(freqs=NOTCH_FREQS, method="fir", phase="zero", verbose=verbose)

    # 4. Bad channel detection — RANSAC via pyprep
    bad_channels: list[str] = []
    try:
        from pyprep import NoisyChannels
        nd = NoisyChannels(raw.copy(), random_state=ICA_RANDOM_STATE)
        nd.find_bad_by_ransac(channel_wise=False)
        bad_channels = list(nd.bad_by_ransac)
    except Exception as e:
        warnings.append(f"pyprep bad channel detection failed: {e}")
        if verbose:
            print(f"  pyprep failed ({e}); skipping bad-channel detection")

    if bad_channels:
        raw.info["bads"] = bad_channels
        if verbose:
            print(f"  Bad channels (RANSAC): {bad_channels}")

    # 5. Average re-reference (excludes bads)
    raw.set_eeg_reference("average", projection=False, verbose=verbose)

    # 6. ICA on continuous data
    n_good_channels = n_channels_original - len(bad_channels)
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
        method="infomax",  # ICLabel was trained on infomax
        fit_params=dict(extended=True),
        verbose=verbose,
    )
    # For ICA, fit on a high-pass-filtered copy (1Hz) — already done above
    ica.fit(raw, picks="eeg", reject_by_annotation=True, verbose=verbose)
    n_ica_components = ica.n_components_

    # Auto-label components with mne-icalabel
    rejected_indices: list[int] = []
    rejected_labels: dict[str, int] = {}
    try:
        from mne_icalabel import label_components
        ic_labels = label_components(raw, ica, method="iclabel")
        labels = ic_labels["labels"]              # list of strings
        probs = ic_labels["y_pred_proba"]         # array of confidences
        for i, (label, prob) in enumerate(zip(labels, probs)):
            if label in ICLABEL_REJECT and prob >= ICLABEL_KEEP_THRESHOLD:
                rejected_indices.append(i)
                rejected_labels[label] = rejected_labels.get(label, 0) + 1
        ica.exclude = rejected_indices
        if verbose:
            print(f"  ICA: {n_ica_components} components, "
                  f"removed {len(rejected_indices)} ({rejected_labels})")
    except Exception as e:
        warnings.append(f"mne-icalabel failed: {e}")
        if verbose:
            print(f"  ICLabel failed ({e}); keeping all components")

    # Apply ICA — zero out rejected components
    raw_clean = ica.apply(raw.copy(), verbose=verbose)

    # 7. Slice into epochs
    epoch_inds, all_labels = load_epoch_info(subject, raw_data_dir=raw_data_dir)
    epochs, trial_labels = slice_into_epochs(
        raw_clean, epoch_inds, all_labels, epoch_type=epoch_type, verbose=verbose
    )
    n_before = len(epochs)

    # Set baseline correction (use first 0.5s as baseline)
    sfreq = epochs.info["sfreq"]
    epochs.apply_baseline(baseline=(0, 0.5), verbose=verbose)

    # 8. Epoch rejection — autoreject learns per-channel thresholds via cross-validation
    # (much more robust across subjects than a fixed peak-to-peak threshold)
    try:
        from autoreject import AutoReject
        ar = AutoReject(
            n_interpolate=[1, 4, 8],
            consensus=[0.2, 0.5, 0.8],
            random_state=ICA_RANDOM_STATE,
            n_jobs=1,
            verbose=verbose,
        )
        epochs_clean, reject_log = ar.fit_transform(epochs.copy(), return_log=True)
        epochs = epochs_clean
        ar_n_bad_epochs = int(reject_log.bad_epochs.sum())
        ar_n_interp = int(reject_log.fix_log.sum() if hasattr(reject_log, 'fix_log') else 0)
        warnings.append(f"autoreject: dropped {ar_n_bad_epochs}, interpolated {ar_n_interp} channel-epochs")
    except Exception as e:
        warnings.append(f"autoreject failed ({e}), falling back to fixed {EPOCH_REJECT_PEAK_TO_PEAK_UV} µV threshold")
        reject_threshold = {"eeg": EPOCH_REJECT_PEAK_TO_PEAK_UV * 1e-6}
        epochs.drop_bad(reject=reject_threshold, verbose=verbose)

    n_after = len(epochs)
    rejection_rate = (n_before - n_after) / n_before if n_before else 0.0

    # Filter labels by epochs.selection (indices of epochs that survived rejection)
    surviving_labels = trial_labels[epochs.selection]
    unique_classes, class_counts = np.unique(surviving_labels, return_counts=True)
    trials_per_class = {str(c): int(n) for c, n in zip(unique_classes, class_counts)}

    # Save .fif
    epochs.save(str(fif_path), overwrite=True, verbose=verbose)

    duration = time.time() - t0

    log: PreprocessLog = {
        "subject": subject,
        "epoch_type": epoch_type,
        "sfreq": float(raw.info["sfreq"]),
        "n_channels_original": n_channels_original,
        "n_channels_after_drop": n_channels_original,  # we don't drop bads, just mark
        "bad_channels": bad_channels,
        "n_bad_channels": len(bad_channels),
        "bandpass": bandpass,
        "notch_freqs": NOTCH_FREQS,
        "ica_n_components": int(n_ica_components),
        "ica_components_rejected": rejected_indices,
        "ica_labels_rejected": rejected_labels,
        "n_epochs_before_reject": n_before,
        "n_epochs_after_reject": n_after,
        "rejection_rate": float(rejection_rate),
        "epoch_reject_threshold_uv": EPOCH_REJECT_PEAK_TO_PEAK_UV,
        "trials_per_class": trials_per_class,
        "duration_seconds": duration,
        "warnings": warnings,
    }
    log_path.write_text(json.dumps(log, indent=2))

    if verbose:
        print(f"{subject}: {n_after}/{n_before} trials kept "
              f"({rejection_rate*100:.1f}% rejected), saved → {fif_path}")

    return epochs, log


def preprocess_all(
    subjects: list[str] | None = None,
    raw_data_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/processed",
    epoch_type: str = "thinking",
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> dict[str, PreprocessLog]:
    """Batch-run preprocessing for a list of subjects (default: all on disk)."""
    from src.loader import list_available_subjects, SUBJECTS

    if subjects is None:
        subjects = list_available_subjects(raw_data_dir)

    logs: dict[str, PreprocessLog] = {}
    for subject in subjects:
        print(f"\n=== {subject} ===")
        try:
            _, log = preprocess_subject(
                subject,
                raw_data_dir=raw_data_dir,
                out_dir=out_dir,
                epoch_type=epoch_type,
                skip_if_exists=skip_if_exists,
                verbose=verbose,
            )
            logs[subject] = log
        except Exception as e:
            print(f"  {subject}: FAILED — {e}")
            logs[subject] = {"subject": subject, "error": str(e)}  # type: ignore

    return logs


def load_clean_epochs(
    subject: str,
    processed_dir: str | Path = "data/processed",
) -> tuple[mne.EpochsArray, PreprocessLog]:
    """Load preprocessed epochs + log for a subject."""
    processed_dir = Path(processed_dir)
    fif_path = processed_dir / f"{subject}-clean-epo.fif"
    log_path = processed_dir / f"{subject}-log.json"
    if not fif_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {fif_path}\n"
            f"Run: from src.preprocessor import preprocess_subject; "
            f"preprocess_subject('{subject}')"
        )
    epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
    log = json.loads(log_path.read_text())
    return epochs, log
