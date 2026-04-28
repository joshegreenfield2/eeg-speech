"""audio_export.py — Phase 5 side quest: EEG channel → WAV.

The idea: pitch-shift EEG into audible range so you can apply audio-domain
denoising tools (iZotope RX, ffmpeg afftdn) and listen for what's there.
Then re-import the cleaned WAV and compare to MNE's ICA-based denoise.

WHY this is interesting:
  - EEG bands map well to audible frequencies after upsampling:
      delta (1-4 Hz)    × 44 = 44-176 Hz  (bass/sub)
      theta (4-8 Hz)    × 44 = 176-352 Hz (low-mid)
      alpha (8-12 Hz)   × 44 = 352-528 Hz (mid)
      beta  (12-30 Hz)  × 44 = 528-1320 Hz (mid-treble)
      gamma (30-100 Hz) × 44 = 1320-4400 Hz (treble)
  - Eye blinks and muscle artifacts often have audio analogues (clicks, breaths)
    that iZotope RX is built to remove.

NOT a serious denoising path — MNE ICA is the right tool. This is for fun
and as an audio-engineering perspective check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mne
import numpy as np
import scipy.io.wavfile as wavfile

from src.loader import load_raw_continuous

mne.set_log_level("WARNING")


def channel_to_wav(
    subject: str,
    channel: str = "FP1",
    out_path: str | Path | None = None,
    raw_data_dir: str | Path = "data/raw",
    target_sfreq: int = 44100,
    duration_s: float | None = None,
    normalize: bool = True,
    pitch_factor: int | None = None,
) -> Path:
    """Export one EEG channel as a WAV file, pitch-shifted into audible range.

    Args:
        subject: e.g. 'MM05'
        channel: e.g. 'FP1' (frontal — has eye blinks)
        out_path: defaults to outputs/audio/<subject>_<channel>.wav
        target_sfreq: WAV sample rate (default 44100 Hz CD-quality)
        duration_s: trim to first N seconds; None = whole recording
        normalize: scale to [-1, 1] (default True; -3 dB headroom kept)
        pitch_factor: integer up-sampling ratio. If None, computed as
                      target_sfreq // raw_sfreq → "treat samples as audio at
                      target rate" → pitch shifted up by exactly that ratio.
    """
    raw = load_raw_continuous(subject, raw_data_dir=raw_data_dir, drop_non_eeg=False)
    if channel not in raw.ch_names:
        raise ValueError(f"Channel {channel!r} not found. Available: {raw.ch_names[:10]}...")

    data = raw.get_data(picks=[channel])[0]  # (n_times,) in V
    raw_sfreq = int(raw.info["sfreq"])

    if duration_s:
        n_samples = int(duration_s * raw_sfreq)
        data = data[:n_samples]

    # The simplest pitch-shift: write the raw samples at the target rate.
    # No interpolation — just relabel the time axis. This shifts all EEG
    # frequencies up by `target_sfreq / raw_sfreq` ≈ 44x.
    if pitch_factor is None:
        pitch_factor = target_sfreq // raw_sfreq

    if normalize:
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak * 0.7  # -3 dB headroom (avoid clipping)

    # Convert to int16 WAV
    int_data = (data * 32767).astype(np.int16)

    if out_path is None:
        out_path = Path("outputs/audio") / f"{subject}_{channel}.wav"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_path), target_sfreq, int_data)

    duration = len(int_data) / target_sfreq
    print(f"  → {out_path}")
    print(f"    raw: {len(int_data)} samples at {raw_sfreq} Hz = {len(int_data)/raw_sfreq:.1f}s of EEG")
    print(f"    wav: {len(int_data)} samples at {target_sfreq} Hz = {duration:.1f}s of audio")
    print(f"    pitch shift: {pitch_factor}× — EEG frequencies are now audible:")
    for band, lo, hi in [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 12),
                         ("beta", 12, 30), ("gamma", 30, 100)]:
        print(f"      {band:6s} {lo}-{hi} Hz → {lo*pitch_factor}-{hi*pitch_factor} Hz")
    return out_path


def wav_to_channel(
    wav_path: str | Path,
    raw_data_dir: str | Path = "data/raw",
    subject: str = "MM05",
    channel: str = "FP1",
) -> np.ndarray:
    """Re-import a (possibly RX-denoised) WAV back as raw EEG samples for comparison.

    Returns the float64 array in V (matches mne.Raw convention). Caller can
    then plug it back into the channel of a Raw object for QC.
    """
    sample_rate, int_data = wavfile.read(str(wav_path))
    # Convert int16 → float (-1..1) → undo the 0.7 normalization
    float_data = int_data.astype(np.float64) / 32767.0 / 0.7

    # Re-scale back to the original peak amplitude
    raw = load_raw_continuous(subject, raw_data_dir=raw_data_dir, drop_non_eeg=False)
    orig = raw.get_data(picks=[channel])[0]
    orig_peak = np.abs(orig).max()
    return float_data * orig_peak


def export_all_channels_for_inspection(
    subject: str,
    out_dir: str | Path = "outputs/audio",
    duration_s: float = 60.0,
    raw_data_dir: str | Path = "data/raw",
):
    """Quick batch: export several interesting channels for one subject.

    FP1/FPZ/FP2 = frontal poles (eye blinks)
    T7/T8       = temporal (jaw EMG)
    OZ          = occipital (alpha if relaxed)
    """
    interesting = ["FP1", "FPZ", "FP2", "T7", "T8", "OZ"]
    out_dir = Path(out_dir)
    print(f"Exporting {len(interesting)} channels for {subject} ({duration_s:.0f}s each)...")
    for ch in interesting:
        try:
            channel_to_wav(
                subject=subject, channel=ch,
                out_path=out_dir / f"{subject}_{ch}_{int(duration_s)}s.wav",
                duration_s=duration_s, raw_data_dir=raw_data_dir,
            )
        except Exception as e:
            print(f"  {ch}: SKIP — {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("subject", default="MM05", nargs="?")
    p.add_argument("--channel", default="FP1")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--all", action="store_true", help="export 6 interesting channels")
    args = p.parse_args()
    if args.all:
        export_all_channels_for_inspection(args.subject, duration_s=args.duration)
    else:
        channel_to_wav(args.subject, channel=args.channel, duration_s=args.duration)
