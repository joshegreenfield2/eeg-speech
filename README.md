# EEG Imagined Speech Decoding

Personal BCI research project. Can a non-invasive EEG + a personalized ML model classify words *I'm only thinking* about? Inspired by the MIT AlterEgo concept — silent communication from thought.

This is a **learning-by-building** project. Hardware (OpenBCI Cyton 8-ch) comes later; right now I'm validating the pipeline on the public KaraOne dataset (14 subjects, 64 channels, imagined + vocalized phonemes/words).

## Status

| Phase | Goal | State |
|-------|------|------|
| 1 | Data access + exploration loader | ✅ |
| 2 | Preprocessing (filter, ICA, autoreject) | ✅ |
| 3 | Baseline SVM on band-power features | ✅ |
| 4 | Deep learning (EEGNet + CNN-BiLSTM) | ✅ Code done; running |
| 5 | Audio-engineering side quest (EEG → WAV → iZotope RX) | ✅ Tooling |

## Headline results (single subject MM05, all-channels, no tuning)

| Task | Mean acc | Balanced acc | Chance |
|------|---------:|-------------:|-------:|
| 11-class (full vocabulary) | 13.3% | 13.3% | 9.1% |
| Vowel vs consonant | 66.1% | **63.7%** | 50% |
| Nasal vs non-nasal | 60.6% | 55.1% | 50% |
| Bilabial vs non-bilabial | 56.4% | 55.7% | 50% |

Both 11-class and the vowel/consonant binary are above chance. The pipeline finds **real signal** — not great signal, but real signal.

Cross-subject SVM and deep-learning numbers will replace this table once the batch finishes.

## Stack

- **Python 3.11** via `uv` (lockfile committed)
- **MNE 1.12** for I/O, filtering, ICA
- **mne-icalabel** + **autoreject** for automatic artifact rejection
- **PyTorch 2.11** for EEGNet / CNN-BiLSTM
- **scikit-learn** for SVM baseline + cross-validation infra

## Repo layout

```
eeg-speech/
├── data/                    # symlinked to external drive — raw .set files NOT committed
├── notebooks/
│   ├── 01_explore_raw.ipynb        # raw signal + PSD checks
│   ├── 02_preprocess_qc.ipynb      # before/after PSD, ICA QC
│   └── 03_results.ipynb            # cross-subject summary + plots
├── scripts/
│   ├── download_karaone.py         # downloads + extracts KaraOne archives
│   ├── run_pipeline.py             # batch: preprocess + SVM all 14
│   └── run_dl.py                   # batch: EEGNet + CNN-BiLSTM all 14
├── src/
│   ├── loader.py                   # KaraOne .set → mne.Epochs (with epoch_inds.mat)
│   ├── preprocessor.py             # 1-45 Hz FIR + notch + ICA + autoreject
│   ├── features.py                 # Welch band power + DWT
│   ├── classifier.py               # SVM + StratifiedKFold(5) + permutation test
│   ├── train.py                    # PyTorch training loop with early stopping
│   ├── models.py                   # EEGNet + CNN-BiLSTM
│   ├── evaluate.py                 # save_run / load_runs / plotting
│   └── audio_export.py             # Phase 5: EEG channel ↔ WAV
└── outputs/
    ├── results/<run_id>/           # per-run JSON + batch summary
    └── figures/                    # PSD comparisons, confusion matrices
```

## Reproducing

```bash
# 1. Install deps
uv sync

# 2. Download KaraOne (~42 GB total; takes ~1 hour)
uv run python scripts/download_karaone.py --delete-archive

# 3. Run preprocessing + SVM baseline on all 14 subjects (~2 hours CPU)
uv run python scripts/run_pipeline.py

# 4. Run EEGNet + CNN-BiLSTM (~30 min CPU)
uv run python scripts/run_dl.py

# 5. View aggregated results
jupyter notebook notebooks/03_results.ipynb
```

## Why this is hard (the methodology rules I'm following)

EEG imagined-speech research has a long history of **inflated published numbers from data leakage**. Following the rules from this project's research notes:

1. **GroupKFold or subject-stratified CV** — never random KFold across mixed-subject data, or sub-window shuffling within trials.
2. **Class imbalance is real** — KaraOne binary subproblems (vowel/consonant) are 30 vs 135. An unweighted SVM will just predict the majority and pretend to win at 81.8%. Always report **balanced accuracy** alongside raw.
3. **Bandpass ≤ 45 Hz** — high-gamma activity is almost always EMG artifact contaminating frontal/temporal channels, not brain signal. Anything reporting big classification gains from gamma should be suspect.
4. **Preprocessing on continuous Raw** — ICA, average reference, and bad-channel detection must run on continuous data BEFORE epoch slicing. Per-epoch ICA is mathematically broken.
5. **Subject-dependent training only** — cross-subject is a research problem (domain adaptation), not a starter task. Each subject is its own classifier.
6. **OpenBCI 8-channel transition is hard** — published 64-channel results don't generalize. Plan for ~10% of subjects to actually work without per-subject re-engineering once we move to consumer hardware.

## Architecture notes

- `data/raw` is a **symlink to an external drive** (`/Volumes/Josh G/eeg-speech-data/raw`). The full dataset is ~42 GB and we want it portable. Loaders read through the symlink transparently.
- KaraOne has two `.set` file layouts depending on subject — some at top level, some only in `set_files/`. The loader handles both.
- Per-epoch peak-to-peak after preprocessing is 76-219 µV on good channels (textbook EEG). pyprep flags 22-35% of channels as bad — high but consistent across subjects, suggesting real recording quality variance, not pyprep over-aggressiveness. Bad channels are excluded from feature extraction, not interpolated.

## License

MIT. Code is mine; the [KaraOne dataset](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) belongs to Zhao & Rudzicz (University of Toronto) and has its own license.
