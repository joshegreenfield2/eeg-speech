#!/usr/bin/env python3
"""compare_audio_denoise.py — compare a raw EEG-as-audio WAV vs an iZotope-cleaned version.

Inputs are two WAVs of the same length (raw + cleaned). Output:
  - PSD overlay plot (true EEG Hz, post un-pitch-shift)
  - Power-in-band table (delta / theta / alpha / beta / gamma)
  - Spectrogram side-by-side
  - One-line verdict

Usage:
    uv run python scripts/compare_audio_denoise.py \\
        outputs/audio/MM05_FP1_60s_raw.wav \\
        outputs/audio/cleaned/MM05_FP1_60s_rx.wav

    # batch mode — pass a directory of cleaned files; the script pairs by name
    uv run python scripts/compare_audio_denoise.py \\
        --raw-dir outputs/audio \\
        --cleaned-dir outputs/audio/cleaned \\
        --report-dir outputs/audio/reports
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram, welch

BANDS = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 12),
    ("beta", 12, 30),
    ("gamma", 30, 100),
]

# audio_export.py default: WAV written at 44100 Hz from raw EEG, no resampling.
# True EEG sfreq = WAV sfreq / pitch_factor. Detected by reading the WAV's sample
# count vs assumed audio duration; or hard-coded if the user knows the source rate.
DEFAULT_PITCH_FACTOR = 44  # 44100 / 1000 Hz EEG


@dataclass
class BandReport:
    name: str
    lo_hz: float
    hi_hz: float
    raw_power: float
    cleaned_power: float

    @property
    def pct_removed(self) -> float:
        if self.raw_power <= 0:
            return 0.0
        return 100.0 * (self.raw_power - self.cleaned_power) / self.raw_power


def load_wav_as_float(path: Path) -> tuple[int, np.ndarray]:
    """Load WAV → (sample_rate, float64 in [-1, 1])."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]  # take left channel if stereo
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128) / 128.0
    else:
        x = data.astype(np.float64)
    return sr, x


def align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim to the shorter array; iZotope sometimes truncates by a few samples."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


def power_in_band(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrate PSD over [lo, hi] Hz."""
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def compute_band_reports(
    raw: np.ndarray,
    cleaned: np.ndarray,
    audio_sfreq: int,
    pitch_factor: int,
) -> tuple[list[BandReport], np.ndarray, np.ndarray, np.ndarray]:
    """Compute PSDs at audio rate, un-pitch frequency axis, integrate per band."""
    nperseg = min(len(raw), 8192)
    f_raw, psd_raw = welch(raw, fs=audio_sfreq, nperseg=nperseg)
    f_cln, psd_cln = welch(cleaned, fs=audio_sfreq, nperseg=nperseg)
    # Un-pitch: audio Hz / pitch_factor → true EEG Hz
    f_eeg = f_raw / pitch_factor

    reports = []
    for name, lo, hi in BANDS:
        # In true EEG Hz; integrate over the matching audio range.
        lo_audio, hi_audio = lo * pitch_factor, hi * pitch_factor
        p_raw = power_in_band(f_raw, psd_raw, lo_audio, hi_audio)
        p_cln = power_in_band(f_cln, psd_cln, lo_audio, hi_audio)
        reports.append(BandReport(name, lo, hi, p_raw, p_cln))
    return reports, f_eeg, psd_raw, psd_cln


def render_plot(
    raw: np.ndarray,
    cleaned: np.ndarray,
    audio_sfreq: int,
    pitch_factor: int,
    f_eeg: np.ndarray,
    psd_raw: np.ndarray,
    psd_cln: np.ndarray,
    title: str,
    out_path: Path,
):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [2, 2, 1.2]})

    # --- PSD overlay (true EEG Hz, log scale) ---
    ax = axes[0]
    ax.semilogy(f_eeg, psd_raw, color="steelblue", lw=1.4, label="raw (pre-RX)")
    ax.semilogy(f_eeg, psd_cln, color="orange", lw=1.4, label="cleaned (post-RX)")
    ax.set_xlim(0, 60)  # show through gamma
    ax.set_xlabel("EEG frequency (Hz, un-pitched)")
    ax.set_ylabel("PSD (V²/Hz, audio-domain)")
    ax.set_title(f"{title} — PSD comparison")
    ax.legend()
    for _, lo, hi in BANDS:
        ax.axvspan(lo, hi, alpha=0.05, color="gray")
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.5, _, ha="center", fontsize=8, alpha=0.6)
    ax.grid(alpha=0.3)

    # --- Spectrograms side by side (audio-domain Hz, but axis label notes the shift) ---
    ax = axes[1]
    f, t, Sxx = spectrogram(raw, fs=audio_sfreq, nperseg=2048, noverlap=1024)
    f_unpitched = f / pitch_factor
    keep = f_unpitched <= 60
    ax.pcolormesh(t, f_unpitched[keep], 10 * np.log10(Sxx[keep] + 1e-20), shading="auto", cmap="magma")
    ax.set_ylabel("EEG Hz (raw)")
    ax.set_xlabel("Time (s, audio-domain)")
    ax.set_title("Spectrogram — raw")

    # --- Difference signal RMS over time (residual = what RX removed) ---
    ax = axes[2]
    diff = raw - cleaned
    win = max(audio_sfreq // 100, 64)  # 10ms-ish windows
    n_win = len(diff) // win
    rms = np.sqrt(np.mean(diff[: n_win * win].reshape(n_win, win) ** 2, axis=1))
    t_rms = np.arange(n_win) * win / audio_sfreq
    ax.plot(t_rms, rms, color="crimson", lw=0.8)
    ax.set_xlabel("Time (s, audio-domain)")
    ax.set_ylabel("RMS of (raw - cleaned)")
    ax.set_title("Residual envelope — what RX removed over time")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_text_report(reports: list[BandReport], raw: np.ndarray, cleaned: np.ndarray, title: str) -> str:
    rms_diff = float(np.sqrt(np.mean((raw - cleaned) ** 2)))
    rms_raw = float(np.sqrt(np.mean(raw ** 2)))
    pct_overall = 100.0 * rms_diff / rms_raw if rms_raw > 0 else 0.0

    lines = [f"=== {title} ===", ""]
    lines.append(f"{'band':<7} {'range (Hz)':<12} {'raw power':>14} {'cleaned':>14} {'% removed':>12}")
    lines.append("-" * 64)
    most_affected = None
    for r in reports:
        rng = f"{r.lo_hz}-{r.hi_hz}"
        lines.append(
            f"{r.name:<7} {rng:<12} {r.raw_power:>14.3e} {r.cleaned_power:>14.3e} {r.pct_removed:>11.1f}%"
        )
        if most_affected is None or r.pct_removed > most_affected.pct_removed:
            most_affected = r
    lines.append("")
    lines.append(f"overall RMS of residual = {rms_diff:.3e}  ({pct_overall:.1f}% of raw RMS)")

    # Verdict
    if pct_overall < 1.0:
        verdict = "RX barely touched the signal (<1% RMS change). Effectively a no-op."
    elif most_affected and most_affected.pct_removed > 30 and pct_overall < 50:
        verdict = (
            f"Targeted removal — biggest hit on {most_affected.name} band "
            f"({most_affected.pct_removed:.0f}%) without nuking everything. Worth re-importing into MNE."
        )
    elif pct_overall > 50:
        verdict = (
            f"RX removed >{pct_overall:.0f}% of the signal RMS — likely too aggressive, "
            f"will have stripped real EEG."
        )
    else:
        verdict = (
            f"Modest broadband cleanup — {pct_overall:.1f}% RMS reduction across all bands. "
            f"Compare per-band % to judge if this matches an artifact pattern."
        )
    lines.append(f"\nverdict: {verdict}")
    return "\n".join(lines)


def compare_one(raw_path: Path, cleaned_path: Path, report_dir: Path, pitch_factor: int) -> dict:
    sr_raw, raw = load_wav_as_float(raw_path)
    sr_cln, cleaned = load_wav_as_float(cleaned_path)
    if sr_raw != sr_cln:
        raise SystemExit(f"sample rate mismatch: {sr_raw} vs {sr_cln}")
    raw, cleaned = align_lengths(raw, cleaned)

    reports, f_eeg, psd_raw, psd_cln = compute_band_reports(raw, cleaned, sr_raw, pitch_factor)
    title = raw_path.stem.replace("_raw", "")
    plot_path = report_dir / f"{title}_compare.png"
    text = render_text_report(reports, raw, cleaned, title)

    render_plot(raw, cleaned, sr_raw, pitch_factor, f_eeg, psd_raw, psd_cln, title, plot_path)
    txt_path = report_dir / f"{title}_compare.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text)
    print(text)
    print(f"\n  plot:  {plot_path}")
    print(f"  text:  {txt_path}")

    return {
        "title": title,
        "raw_path": str(raw_path),
        "cleaned_path": str(cleaned_path),
        "audio_sfreq": sr_raw,
        "pitch_factor": pitch_factor,
        "n_samples": len(raw),
        "duration_s_audio": len(raw) / sr_raw,
        "duration_s_eeg": len(raw) / (sr_raw / pitch_factor),
        "rms_residual_pct": 100.0 * float(np.sqrt(np.mean((raw - cleaned) ** 2))) / max(float(np.sqrt(np.mean(raw ** 2))), 1e-12),
        "bands": [asdict(r) | {"pct_removed": r.pct_removed} for r in reports],
    }


def pair_files(raw_dir: Path, cleaned_dir: Path) -> list[tuple[Path, Path]]:
    """Match `<X>_raw.wav` in raw_dir to `<X>_rx.wav` (or any non-raw suffix) in cleaned_dir."""
    pairs = []
    for raw in sorted(raw_dir.glob("*_raw.wav")):
        base = raw.stem.replace("_raw", "")
        candidates = list(cleaned_dir.glob(f"{base}_*.wav"))
        candidates = [c for c in candidates if not c.name.endswith("_raw.wav")]
        if not candidates:
            print(f"  skip {raw.name}: no cleaned counterpart in {cleaned_dir}")
            continue
        if len(candidates) > 1:
            print(f"  {raw.name}: multiple cleaned candidates {[c.name for c in candidates]}, using first")
        pairs.append((raw, candidates[0]))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw", nargs="?", help="raw WAV path (single-pair mode)")
    p.add_argument("cleaned", nargs="?", help="cleaned WAV path (single-pair mode)")
    p.add_argument("--raw-dir", help="directory of *_raw.wav files (batch mode)")
    p.add_argument("--cleaned-dir", help="directory of cleaned WAVs (batch mode)")
    p.add_argument("--report-dir", default="outputs/audio/reports", help="where to write plots/text")
    p.add_argument("--pitch-factor", type=int, default=DEFAULT_PITCH_FACTOR,
                   help="EEG×N pitch shift used in audio_export.py (default 44 = 44100/1000)")
    args = p.parse_args()

    report_dir = Path(args.report_dir)
    summaries = []

    if args.raw and args.cleaned:
        summaries.append(compare_one(Path(args.raw), Path(args.cleaned), report_dir, args.pitch_factor))
    elif args.raw_dir and args.cleaned_dir:
        for raw_p, cln_p in pair_files(Path(args.raw_dir), Path(args.cleaned_dir)):
            print(f"\n--- pair: {raw_p.name} vs {cln_p.name} ---")
            summaries.append(compare_one(raw_p, cln_p, report_dir, args.pitch_factor))
    else:
        p.error("provide either positional <raw> <cleaned>, or --raw-dir + --cleaned-dir")

    summary_path = report_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nsummary JSON: {summary_path} ({len(summaries)} pair(s))")


if __name__ == "__main__":
    main()
