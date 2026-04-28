# Research Summary — EEG Imagined Speech Pipeline

**Dataset:** KaraOne (64-ch, 14 subjects, 1 kHz, 11-class imagined speech, MATLAB 7.3 .mat, ~24 GB)
**Target:** Beat 57% SOTA on 11-class KaraOne; lay groundwork for eventual OpenBCI hardware transfer

---

## Stack Recommendation

The Python EEG ecosystem has converged around MNE-Python as its uncontested core. MNE handles everything from file I/O through filtering, re-referencing, ICA, epoching, and visualization. Pair it with scikit-learn for baseline classifiers and PyTorch (via braindecode) for deep learning. That trio covers the entire pipeline from raw signal to trained model. Require Python >=3.11 because braindecode 1.4.0 enforces it.

The non-obvious dependency that blocks progress if you miss it: KaraOne distributes MATLAB 7.3 (HDF5-format) .mat files. `scipy.io.loadmat` silently fails on these. The correct loader is `mat73.loadmat`. Install both `mat73` and `h5py`. After loading: mat73 → numpy arrays → `mne.io.RawArray` → MNE ecosystem → save preprocessed epochs as `.fif` for fast reload (~10x faster than re-running preprocessing).

For preprocessing, add three specialized libraries on top of MNE: `pyprep` for robust average referencing and bad channel detection (run first, before ICA), `mne-icalabel` for automatic ICA component classification, and `autoreject` for cross-validated epoch rejection thresholds. Do not use TensorFlow. Do not use `eeglib` or `PyEEG` — both are unmaintained.

**Core stack:**
| Library | Version | Role |
|---------|---------|------|
| mne | >=1.12.1 | EEG signal processing core |
| mat73 | >=0.65 | KaraOne .mat loader (MATLAB 7.3 HDF5) |
| h5py | >=3.x | mat73 dependency |
| pyprep | >=0.6.0 | Robust bad channel detection |
| mne-icalabel | >=0.8.1 | Auto-label ICA components |
| autoreject | >=0.4.3 | Epoch rejection thresholds |
| scikit-learn | >=1.5 | Baseline classifiers, CV |
| torch | >=2.3 | DL backend |
| braindecode | >=1.4.0 | EEGNet and EEG-native DL models |
| numpy | >=1.26 | Arrays |
| scipy | >=1.13 | Signal processing utilities |
| matplotlib | any | Plotting |

---

## Pipeline Architecture

Six discrete stages, each saving a durable artifact to disk. No stage reaches back two levels.

```
Stage 0  data/raw/MM_DD_YYYY/*.mat
         Never modified. Read-only ground truth.

Stage 1  LOADING         (src/loader.py)
         mat73 → numpy → mne.RawArray → mne.Epochs
         → data/interim/subject_XX-epo.fif

Stage 2  PREPROCESSING   (src/preprocessor.py)
         bandpass (1-45 Hz FIR) → notch (60 Hz) → drop non-EEG channels
         → bad channel detection → average reference → ICA → epoch rejection (±100 µV)
         → data/processed/subject_XX-clean-epo.fif

Stage 3  FEATURE EXTRACTION  (src/features.py)
         [sklearn path] Welch PSD per channel per band → features_X.npy + labels_y.npy
         [DL path]      raw 3D epoch array → TensorDataset (no hand-crafted features)
         → data/features/

Stage 4  CLASSIFICATION   (src/classifier.py)
         StandardScaler (fit train only) → SelectKBest → SVM(rbf) → StratifiedKFold(5)
         → outputs/results/results_subjectXX_svm.json

Stage 5  DEEP LEARNING    (src/models.py + src/train.py)   [Phase 2+]
         EEGNet → CNN-BiLSTM
         → outputs/models/ + outputs/results/

Stage 6  EVALUATION       (src/evaluate.py)
         Aggregate JSONs → summary_results.csv + confusion matrices + per-subject plots
         → outputs/figures/ + outputs/summary_results.csv
```

**Key structural decisions:**
- `.fif` as intermediate format — preserves all MNE metadata (channel names, montage, event codes, bad channel list)
- `config.yaml` as single source of truth for all parameters
- Notebooks import from `src/` — they are the development and visualization layer, not the pipeline itself
- Subject-dependent evaluation first; cross-subject is Phase 3+ territory
- SVM baseline must exist before any deep learning — it is the diagnostic benchmark

**Build order:** Loader → Preprocessor → Band-power features + SVM → Evaluation infrastructure → EEGNet → TFR features → CNN-BiLSTM

---

## Table Stakes Features

The pipeline cannot function correctly without all of these:

- **mat73-based KaraOne loader** — mat73.loadmat → numpy → MNE Epochs → .fif
- **Non-EEG channel removal** — drop EMG (it's a Kinect color sensor), M1, M2, EKG, Trigger
- **Correct label source** — use `ID_p.txt` (actual presentation order), never `ID.txt`
- **Correct condition isolation** — use `epoch_inds.mat` to isolate imagined speech trials only
- **Bandpass filter on continuous data before epoching** — 1–45 Hz zero-phase FIR
- **60 Hz notch filter** — powerline artifact dominates spectral features without this
- **Average re-reference** — after bad channel removal, not before
- **ICA-based ocular artifact removal** — 4 EOG channels in KaraOne; auto-label with mne-icalabel
- **Epoch rejection at ±100 µV** — gross artifact catch; track rejection rate per subject
- **GroupKFold cross-validation** — grouped by subject ID; standard KFold inflates accuracy 15–30 pp
- **Scaler fitted on training data only** — never fit_transform on full dataset
- **Per-subject results** — mean ± std + per-subject breakdown; mean alone hides extreme variability
- **Chance-level baseline** — 9.1% theoretical for 11-class; run 1000-shuffle permutation test
- **SVM+RBF baseline before any deep learning**

---

## Critical Warnings

**1. Data leakage from segment-level CV will destroy your results.**
Only 27% of 63 published EEG deep learning papers properly avoided it. A documented case: 53% real accuracy inflated to 99.8% purely from wrong CV splits. Use `GroupKFold(groups=subject_ids)` from day one. Never use `train_test_split` on pre-segmented EEG data.

**2. High-gamma accuracy is probably EMG artifact, not brain signal.**
Imagined speech induces subtle jaw/tongue/facial muscle activity visible in scalp EEG at 30–120 Hz. Papers explicitly note high-gamma gains "might be due to EMG artifacts." Stick to 1–45 Hz bandpass. Verify by checking PSD at temporal channels (T7, T8, FT7, FT8) — a broadband shelf above 30 Hz is the EMG signature.

**3. KaraOne has four silent loader-breaking gotchas.**
(a) Use `ID_p.txt` not `ID.txt` for labels. (b) Channel labeled "EMG" is a Kinect color sensor — drop it. (c) Trial counts vary per subject — do not hardcode 132. (d) Imagined and vocalized trials are interleaved — use `epoch_inds.mat` to separate.

**4. Preprocessing order is non-negotiable.**
Filter → bad channel detection → average reference → ICA (on continuous raw data) → epoch rejection → baseline correction. Deviating causes measurable accuracy degradation.

**5. 8-channel OpenBCI transfer will require near-complete pipeline redesign.**
A 2025 study found 64→8 channel reduction leaves only ~10% of subjects with unchanged accuracy. Simulate it on KaraOne first: select 8 channels matching OpenBCI positions, retrain, measure the accuracy drop. This experiment is worth running in Phase 5 before buying hardware.

---

## Phase Recommendations

**Phase 1 — Data Access and Exploration (Day 1)**
Get one subject's .mat file into MNE correctly, nothing else. `src/loader.py` + `notebooks/01_explore_raw.ipynb`. Stop when you have `(n_epochs, 64, n_times)` array with correct labels.

**Phase 2 — Preprocessing Pipeline (Days 2–3)**
Clean .fif files for all 14 subjects. Implement full preprocessing chain in `src/preprocessor.py`. Stop when rejection rate is 10–30%, ICA removes 1–5 components, PSD shows clean 1/f slope.

**Phase 3 — Baseline Classifier (Days 4–5)**
Band-power features + SVM + GroupKFold in `src/classifier.py`. Run permutation test. Implement binary subproblems (vowel/consonant) as pipeline health check — these should hit 65–70%.

**Phase 4 — Deep Learning Baseline (Week 2)**
EEGNet via braindecode. Compare EEGNet vs. SVM per subject. EEGNet that can't beat SVM has a structural problem.

**Phase 5 — Feature and Model Iteration (Week 3+)**
DWT features, majority voting, Morlet TFR, CNN-BiLSTM. Aim past 57% SOTA.

**Phase 6 — Targeted Experiments (Week 4+)**
Vocalized-to-imagined transfer. 8-channel simulation (before buying OpenBCI). iZotope RX vs. ICA denoising comparison.

**Deferred explicitly:** real-time inference, cross-subject generalization, custom architectures, GAN augmentation, source localization, GUI, multi-modal fusion.

---

## Open Questions to Resolve Early

1. **Exact internal structure of KaraOne .mat files** — read `src/eeg_isr/karaone.py` in AshrithSagar repo against actual data before writing the loader
2. **Is the data pre-epoched or continuous?** — affects ICA strategy
3. **Actual trial count per subject** after Kinect-malfunction exclusions
4. **Trials remaining after preprocessing** — classes need adequate support for StratifiedKFold(5)
5. **Channel name format** — do KaraOne names match a standard MNE montage?
6. **Does ICA converge reliably** with ~650 seconds of data per subject (64 channels)?
7. **Binary subproblem accuracy** — if vowel/consonant classification doesn't reach 65%, the pipeline is broken

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack selection | HIGH | MNE, scikit-learn, braindecode verified on PyPI |
| KaraOne data loading | HIGH | mat73 requirement verified; dataset well-documented |
| Preprocessing order | HIGH | Nature Comms Bio 2025 + multiple studies |
| Data leakage risks | HIGH | Peer-reviewed meta-analysis of 63 studies |
| SOTA accuracy targets | MEDIUM | Incompatible evaluation protocols across papers |
| OpenBCI channel reduction | HIGH | 2025 PMC study with explicit "not recommendable" conclusion |
| iZotope RX side quest | MEDIUM | No published precedent; technically feasible |

---

## Key Sources

- KaraOne database: http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html
- AshrithSagar repo: https://github.com/AshrithSagar/EEG-Imagined-speech-recognition
- Data leakage in EEG DL (PMC 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/
- Hybrid CNN-BiLSTM (PMC 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11595501/
- Electrode reduction (PMC 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12263900/
- Preprocessing effects (Nature Comms Bio 2025): https://www.nature.com/articles/s42003-025-08464-3
- 57% SOTA wavelet+DNN (arXiv 2020): https://arxiv.org/abs/2003.10433
