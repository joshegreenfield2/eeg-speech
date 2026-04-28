# Handoff to other computer

Saved 2026-04-28. Mid-flight: EEGNet was running on subject 1/14 when paused.

## What's where

| Where | What |
|---|---|
| **GitHub** (https://github.com/joshegreenfield2/eeg-speech) | All code, the SVM baseline summary CSV, README |
| **External drive `/Volumes/Josh G/eeg-speech-data/`** | Raw EEG + preprocessed `.fif` files + saved outputs |
| **Old laptop (paused)** | Just code; data is on the drive |

Everything you need is in those three places. Plug the drive into the new machine and follow the steps below.

## Setup on the new computer

```bash
# 1. Plug in the drive. Confirm it mounted as /Volumes/Josh G
ls "/Volumes/Josh G/eeg-speech-data/"
# expected: outputs/  processed/  raw/

# 2. Clone the repo
cd ~/Projects   # or wherever you keep code
git clone https://github.com/joshegreenfield2/eeg-speech.git
cd eeg-speech

# 3. Install Python deps (uv reads pyproject.toml + uv.lock)
uv sync

# 4. Wire up the data symlinks (same trick as the old laptop)
ln -s "/Volumes/Josh G/eeg-speech-data/raw" data/raw
ln -s "/Volumes/Josh G/eeg-speech-data/processed" data/processed

# 5. Pull outputs from the drive (so we don't redo SVM/audio/etc.)
mkdir -p outputs
cp -R "/Volumes/Josh G/eeg-speech-data/outputs/"* outputs/

# 6. Sanity check — should print 14 preprocessed subjects
ls data/processed/*.fif | wc -l   # → 14
```

## Resume the EEGNet batch

```bash
# Run with caffeinate so it doesn't sleep, -u for unbuffered logs
nohup caffeinate -i uv run python -u scripts/run_dl.py \
  --model eegnet --epochs-max 40 \
  > outputs/batch_dl_eegnet_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor:
tail -f $(ls -t outputs/batch_dl_eegnet_*.log | head -1)
```

ETA: ~1-1.5 hours for all 14 subjects on EEGNet (depends on the new machine's CPU). With `--model both` it'd be ~3 hours; can decide once EEGNet finishes whether to also run CNN-BiLSTM.

## After EEGNet finishes

```bash
# Aggregate all results across SVM + EEGNet runs
jupyter notebook notebooks/03_results.ipynb
# (Run all cells — produces summary plots in outputs/figures/)

# Final commit + push
git add outputs/summary_results.csv outputs/figures/*.png
git commit -m "results(phase-4): EEGNet across 14 subjects + cross-model comparison"
git push
```

## What's already done (so don't redo)

- ✅ All 14 subjects preprocessed (1.83 GB on drive at `eeg-speech-data/processed/`)
- ✅ SVM baseline run + saved (committed to GitHub as `outputs/summary_results.csv`)
- ✅ Audio side-quest WAV files exported (3 channels × 2 durations on drive at `eeg-speech-data/outputs/audio/`)
- ✅ All Phase 1-5 code on GitHub

## Outstanding (in priority order)

1. **EEGNet batch** — ~1.5 hr, just kick it off via the command above
2. **CNN-BiLSTM batch** (optional) — ~2 hr, only if you want the comparison
3. **Notebook 03** — generates the cross-subject summary plots
4. **iZotope experiment** — manual; load the WAVs into iZotope, denoise, compare

## Headline numbers so far (SVM, all 14 subjects)

| Task | Mean balanced accuracy | Chance |
|---|---:|---:|
| 11-class | 16.8% | 9.1% |
| Vowel vs consonant | **68.2%** | 50% |
| Nasal vs non-nasal | 60.9% | 50% |
| Bilabial vs non-bilabial | 60.7% | 50% |

All above chance → real signal across all 14 subjects.

## Issues / known weirdness

- **MM08 loses 4/11 classes** (gnaw/knew/pat/pot) due to autoreject dropping 76% of its epochs. Pipeline still produces above-chance results on the surviving 7 classes. Worth investigating later — maybe loosen autoreject for noisy subjects.
- **pyprep flags 22-35% of channels as bad** on most subjects. High but consistent across runs. Might be worth comparing against published KaraOne pipelines to see if our threshold is too aggressive.
