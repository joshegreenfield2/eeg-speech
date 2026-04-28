# Architecture Research — EEG Imagined Speech Pipeline

**Dataset:** KaraOne (MATLAB .mat, 64-channel EEG, ~12 subjects, imagined speech trials)
**Stack:** MNE-Python, scikit-learn, PyTorch, uv
**Context:** Solo learner, scripts + notebooks, no deployment target
**Researched:** 2026-04-27
**Confidence:** HIGH (MNE patterns from official docs + reviewed 3 real pipeline repos)

---

## Pipeline Stages

Ordered from raw data to final output. Each stage is a discrete transformation with a durable artifact on disk.

```
Stage 0: Raw .mat files (on disk, never modified)
         |
Stage 1: LOADING
         scipy.io.loadmat or h5py → numpy arrays extracted, labels parsed
         |  saved as: -epo.fif (MNE Epochs object per subject)
Stage 2: PREPROCESSING
         MNE Raw → bandpass filter → re-reference → ICA artifact removal → epoch extraction
         |  saved as: -clean-epo.fif (clean Epochs per subject)
Stage 3: FEATURE EXTRACTION
         Epochs array (n_epochs, n_channels, n_times) → feature matrix (n_epochs, n_features)
         |  saved as: features_X.npy + labels_y.npy per subject
Stage 4: CLASSIFICATION (scikit-learn path)
         feature matrix → train/test split → model fit → accuracy/F1
         |  saved as: results_subject_XX.json
Stage 5: DEEP LEARNING (PyTorch path, optional)
         Epochs array (3D) → DataLoader → EEGNet/CNN-LSTM → loss curves → accuracy
         |  saved as: model_subject_XX.pt + results_subject_XX.json
Stage 6: EVALUATION & VISUALIZATION
         results JSONs → aggregate metrics → confusion matrices → plots
         |  saved as: figures/ + summary_results.csv
```

Each stage reads from the previous stage's output file. No stage reaches back two levels. This is the rule that keeps the pipeline navigable.

---

## Component Breakdown

### Component 1: Data Loader (`src/loader.py`)

**Input:** `/data/raw/MM_DD_YYYY/` — raw KaraOne `.mat` files, one directory per subject

**Output:** MNE `Epochs` object saved as `-epo.fif` in `/data/interim/`

**What it does:**
- Detects mat file version. KaraOne files are MATLAB 7.3 (HDF5 format), so use `h5py`, not `scipy.io.loadmat`. `scipy.io.loadmat` silently fails or errors on v7.3.
- Extracts fields: EEG data array (channels x time), channel labels, sampling rate, event markers (trial onset times, class labels for each imagined word).
- Constructs `mne.Info` with channel names and types, then `mne.EpochsArray` directly from the trial-segmented array.
- Writes one `-epo.fif` per subject.

**Key decisions:**
- Always use `h5py.File(path, 'r')` first; fall back to `scipy.io.loadmat` only if h5py raises an error (pre-7.3 files).
- Do not do any signal processing here. Loading is loading.
- KaraOne has 64 EEG channels. Channel names must match a standard 10-20 montage — use `mne.channels.make_standard_montage('easycap-M1')` or the dataset's documented cap layout.

---

### Component 2: Preprocessor (`src/preprocessor.py`)

**Input:** `-epo.fif` per subject from `/data/interim/`

**Output:** `-clean-epo.fif` per subject in `/data/processed/`

**What it does:**
- Bandpass filter: 0.5–40 Hz (preserves theta, alpha, beta, low gamma — all relevant for speech imagery; removes DC drift and high-freq noise)
- Notch filter: 60 Hz (US power line; use 50 Hz if data collected in EU)
- Re-reference: average reference (`raw.set_eeg_reference('average')`)
- ICA: fit on continuous data before epoching if you have the raw signal; otherwise fit on the epoch array. Remove components correlated with eye blinks (EOG-like ICs).
- Epoch rejection: drop epochs where any channel exceeds ±100 µV peak-to-peak.
- Writes clean epochs as `-clean-epo.fif`.

**Key decisions:**
- Do ICA before epoching if the raw continuous signal is available in the .mat file. KaraOne includes pre-epoched data in some formats — if that's the case, skip ICA and rely on amplitude rejection only for the initial pipeline pass.
- Do not baseline-correct until after ICA. Baseline correction before ICA can bias component decomposition.
- Keep a rejection log (how many epochs dropped per subject) in a `.json` sidecar file.

---

### Component 3: Feature Extractor (`src/features.py`)

**Input:** `-clean-epo.fif` per subject

**Output:** `features_X_subjectXX.npy` (shape: n_epochs × n_features) and `labels_y_subjectXX.npy` in `/data/features/`

**What it does:**
- Extracts the feature representation used for classification. Choose one of:

  **Option A — Band power (start here):** Compute power spectral density per channel per band (theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz) using `mne.time_frequency.psd_array_welch`. Concatenate → (n_epochs, n_channels × n_bands). Interpretable, fast, good baseline.

  **Option B — Time-frequency (TFR):** Morlet wavelet transform via `mne.time_frequency.tfr_array_morlet`. Output is 4D (n_epochs, n_channels, n_freqs, n_times) — flatten for sklearn, keep 3D for PyTorch.

  **Option C — Raw epoch array (for deep learning):** Pass the 3D epoch array (n_epochs, n_channels, n_times) directly into PyTorch DataLoader. No hand-crafted features.

- Saves numpy arrays, not MNE objects. Features are stack-agnostic from this point.

**Key decisions:**
- Build Option A first. Band power is the most-cited baseline in imagined speech literature and gives you something to beat. Add B/C in later phases.
- Do not mix feature types in a single run. Keep feature extraction strategies cleanly separated so you can compare them.

---

### Component 4: Classifier (`src/classifier.py`)

**Input:** `features_X_subjectXX.npy` + `labels_y_subjectXX.npy`

**Output:** `results_subjectXX.json` with accuracy, F1-per-class, confusion matrix in `/outputs/results/`

**What it does:**
- Loads feature/label arrays.
- Runs subject-dependent classification (train/test split within one subject — this is the standard for KaraOne evaluation).
- Pipeline: `StandardScaler → SelectKBest(f_classif, k=50) → SVM(rbf)` or `RandomForest`.
- Uses `StratifiedKFold(n_splits=5)` cross-validation.
- Saves per-fold and aggregate metrics to JSON.

**Key decisions:**
- Always use `StratifiedKFold`, not `train_test_split`, for small-N EEG datasets. KaraOne has ~11 subjects with ~100 trials each — a single 80/20 split has high variance.
- SVM with RBF kernel is the most consistently reported baseline for imagined speech (confirmed in the 2024 PMC systematic review). Train this before touching PyTorch.
- `SelectKBest` is essential — raw band-power features for 64 channels × 3 bands = 192 features, many irrelevant.

---

### Component 5: Deep Learning Model (`src/models.py` + `src/train.py`)

**Input:** `-clean-epo.fif` → 3D numpy array loaded into `torch.utils.data.Dataset`

**Output:** `model_subjectXX.pt` + `results_subjectXX_dl.json` in `/outputs/models/` and `/outputs/results/`

**What it does:**
- Wraps epochs array in a `torch.utils.data.TensorDataset`.
- Implements EEGNet (the standard lightweight BCI architecture — 2 depthwise conv layers, ~2000 params, designed for small EEG datasets).
- Training loop: Adam optimizer, cross-entropy loss, 150 epochs, early stopping on val loss.
- Saves model weights and training curves.

**Key decisions:**
- Use EEGNet, not a custom CNN. EEGNet is purpose-built for EEG (handles the (channels, time) structure natively), has a reference implementation, and is a known baseline. Build from scratch using the 2018 paper's architecture — it's 30 lines of PyTorch.
- Subject-dependent training only in early phases. Cross-subject generalization is a research problem, not a starter project.
- Batch size 16–32. KaraOne has ~100 trials per subject — a batch size of 64 means 1–2 batches per epoch, which is too coarse for learning.

---

### Component 6: Evaluator (`src/evaluate.py`)

**Input:** All `results_*.json` from `/outputs/results/`

**Output:** `summary_results.csv` + figures in `/outputs/figures/`

**What it does:**
- Aggregates per-subject metrics into a summary table.
- Computes mean ± std accuracy across subjects.
- Plots: confusion matrix per subject, accuracy bar chart across subjects, loss curves (for DL models).
- Uses `matplotlib`/`seaborn`.

**Key decisions:**
- This is the only component that should produce figures. Plotting scattered throughout notebooks creates ambiguity about which visualization is authoritative.
- Report chance level alongside accuracy. KaraOne has 5 classes (imagined words) → chance = 20%. Any result below 25% needs investigation, not celebration.

---

## Data Flow

```
/data/raw/MM_DD_YYYY/*.mat
    |
    | h5py.File() → numpy arrays
    v
mne.EpochsArray  (n_epochs × 64 channels × n_times)
    |
    | epochs.save()
    v
/data/interim/subject_XX-epo.fif
    |
    | mne.read_epochs() → filter → ICA → reject
    v
mne.Epochs (clean)  (n_epochs × 64 × n_times, fewer epochs after rejection)
    |
    | epochs.save()
    v
/data/processed/subject_XX-clean-epo.fif
    |
    |--- [sklearn path] epochs.get_data() → psd_array_welch() → np.save()
    |        v
    |   /data/features/features_X_subjectXX.npy   (n_epochs × 192)
    |   /data/features/labels_y_subjectXX.npy     (n_epochs,)
    |        |
    |        | StandardScaler → SelectKBest → SVM → StratifiedKFold
    |        v
    |   /outputs/results/results_subjectXX_svm.json
    |
    |--- [pytorch path] epochs.get_data() → TensorDataset → DataLoader
             v
         torch.Tensor  (n_epochs × 1 × 64 × n_times)
             |
             | EEGNet forward pass, Adam, cross-entropy
             v
         /outputs/models/model_subjectXX.pt
         /outputs/results/results_subjectXX_dl.json
             |
             v
         /outputs/figures/ + summary_results.csv
```

**Format summary by stage:**

| Stage | Format | Shape | Library |
|-------|--------|-------|---------|
| Raw | .mat (HDF5) | varies | h5py |
| After loading | -epo.fif | (n_epochs, 64, n_times) | MNE |
| After preprocessing | -clean-epo.fif | (n_epochs_clean, 64, n_times) | MNE |
| Features (sklearn) | .npy pair | (n_epochs, n_features) + (n_epochs,) | numpy |
| Model input (PyTorch) | torch.Tensor | (n_epochs, 1, 64, n_times) | PyTorch |
| Results | .json | flat dict | stdlib |
| Summary | .csv | (n_subjects, n_metrics) | pandas |

---

## Directory Structure

```
eeg-speech/
├── pyproject.toml              # uv-managed dependencies
├── uv.lock
├── .python-version
├── config.yaml                 # all parameters: filter freqs, epoch window, k_best, etc.
│
├── data/
│   ├── raw/                    # original .mat files — NEVER modified
│   │   └── MM_DD_YYYY/         # one dir per subject session (KaraOne naming)
│   ├── interim/                # after loading: -epo.fif files
│   ├── processed/              # after preprocessing: -clean-epo.fif files
│   └── features/               # .npy feature arrays + label arrays
│
├── outputs/
│   ├── results/                # per-subject JSON metrics
│   ├── models/                 # saved .pt weights
│   └── figures/                # all plots
│
├── src/
│   ├── __init__.py
│   ├── loader.py               # .mat → MNE Epochs → .fif
│   ├── preprocessor.py         # filter + ICA + rejection → clean .fif
│   ├── features.py             # Epochs → numpy feature arrays
│   ├── classifier.py           # sklearn pipeline + cross-val
│   ├── models.py               # EEGNet architecture (PyTorch)
│   ├── train.py                # training loop + checkpointing
│   ├── evaluate.py             # aggregate results + figures
│   └── utils.py                # config loader, path helpers, logging
│
└── notebooks/
    ├── 01_explore_raw.ipynb    # sanity-check one subject's .mat file
    ├── 02_preprocessing_dev.ipynb  # develop and tune preprocessing steps
    ├── 03_features_eda.ipynb   # visualize feature distributions, PCA
    ├── 04_classifier_dev.ipynb # prototype sklearn pipeline
    ├── 05_eegnet_dev.ipynb     # prototype EEGNet training loop
    └── 06_results.ipynb        # load all JSONs, produce summary figures
```

**Rules:**
- `data/raw/` is read-only. Never write to it.
- `src/` contains all reusable logic. Notebooks import from `src/`, they do not contain duplicate logic.
- `config.yaml` is the single source of truth for all parameters. No magic numbers in code.
- Each `data/` subdirectory is a cache. If you rerun preprocessing, it overwrites `data/processed/` — that is fine and expected.

---

## Build Order

Build in this order. Each step depends on the previous being stable.

**Phase 1: Data Access (Day 1)**
1. Write `src/loader.py` — get one subject's .mat file loaded into an MNE Epochs object. Print shape. Confirm channel count, sampling rate, trial count.
2. Write `notebooks/01_explore_raw.ipynb` — visualize raw traces for one subject, confirm events parsed correctly, check label distribution.

Stop and verify: you have (n_epochs, 64, n_times) clean data with correct labels before moving on.

**Phase 2: Preprocessing (Day 2–3)**
3. Write `src/preprocessor.py` — filter, re-reference, epoch rejection. No ICA yet.
4. Write `notebooks/02_preprocessing_dev.ipynb` — plot PSD before/after filter, compare epoch counts before/after rejection.
5. Add ICA only after basic preprocessing is confirmed working.

Stop and verify: clean .fif files exist for all subjects, rejection rates are reasonable (losing <30% of epochs is acceptable).

**Phase 3: Features + Baseline Classifier (Day 4–5)**
6. Write `src/features.py` — band power extraction only (Option A).
7. Write `src/classifier.py` — sklearn SVM pipeline with 5-fold CV.
8. Run on all subjects. Confirm you get above-chance accuracy.
9. Write `notebooks/03_features_eda.ipynb` to inspect feature distributions.
10. Write `notebooks/04_classifier_dev.ipynb` for prototyping classifier variations.

Stop and verify: you have a numeric baseline. Every subsequent model must beat this.

**Phase 4: Evaluation Infrastructure (Day 5–6)**
11. Write `src/evaluate.py` — aggregate all subject JSONs, produce summary CSV and figures.
12. Write `notebooks/06_results.ipynb`.

**Phase 5: Deep Learning (Week 2+)**
13. Write `src/models.py` — EEGNet architecture.
14. Write `src/train.py` — training loop.
15. Write `notebooks/05_eegnet_dev.ipynb` — verify training converges on one subject before batching all subjects.

**Phase 6: Iteration**
16. Add TFR features (Option B) to `src/features.py` as an alternate mode.
17. Experiment with cross-subject generalization.
18. Tune EEGNet hyperparameters.

---

## Notebook vs. Script Tradeoffs

| Stage | Use | Why |
|-------|-----|-----|
| Initial .mat exploration | Notebook | One-time, visual, interactive — you're figuring out the data structure |
| Preprocessing development | Notebook | Need to see plots of signals before/after each step |
| Feature EDA | Notebook | PCA plots, feature distributions are visual work |
| Classifier prototyping | Notebook | Fast iteration, want to see confusion matrices inline |
| EEGNet development | Notebook | Training curves, debugging gradient flow |
| Results/figures | Notebook | Final visualization is inherently interactive |
| Loader | Script (`src/`) | Runs on all subjects; needs to be importable, not re-written per subject |
| Preprocessor | Script (`src/`) | Batch operation; called in a loop |
| Feature extraction | Script (`src/`) | Batch operation |
| Classifier | Script (`src/`) | Batch operation; reproducibility matters |
| Training loop | Script (`src/`) | Long-running; should be runnable headless from terminal |
| Evaluation | Script (`src/`) | Deterministic aggregation — no interactivity needed |

**Rule:** If you would run it on more than one subject, it lives in `src/`. If it produces a plot you need to look at, it lives in a notebook that imports from `src/`. Notebooks call `src/` functions — they are not the pipeline themselves.

**Pitfall to avoid:** Do not write your preprocessing logic inside a notebook cell and call it done. It will not be reusable, you will copy-paste it for each subject, and you will debug the same bug 12 times. Write the function in `src/preprocessor.py`, test it in the notebook, then move on.

---

## Integration with AshrithSagar Repo

The [AshrithSagar/EEG-Imagined-speech-recognition](https://github.com/AshrithSagar/EEG-Imagined-speech-recognition) repo is useful as a reference implementation. Use it this way:

**What to borrow (copy and adapt):**
- `workflows/download-karaone.py` — use this verbatim to download the dataset. No need to reinvent.
- `src/eeg_isr/karaone.py` — read this to understand how KaraOne .mat files are structured (field names, trial markers, subject layout). Extract the loading logic into your own `src/loader.py`.
- `src/eeg_isr/features.py` — read to understand what feature extraction choices they made. Use as a reference, not a dependency.
- `config-template.yaml` — copy the karaone section as a starting point for your own `config.yaml`.

**What NOT to do:**
- Do not install it as a dependency (`pip install eeg-isr`). You would be locked into their architecture decisions and their config format.
- Do not clone it and work inside it. Their structure (Sacred experiment tracking, the `workflows/` separation) adds complexity that has no value for a solo learner.
- Do not use Sacred (their experiment tracking framework). It is substantial overhead. Use plain JSON files for results until you need something more.
- Do not use their TFR-first approach as your starting point. Their `tfr.py` path goes deep learning immediately. Build the sklearn baseline first.

**The extraction strategy:**
1. Read `karaone.py` in their `src/eeg_isr/` to understand the .mat file structure — what keys exist, what shape the data array is, how trial labels are stored.
2. Write your own `src/loader.py` using that knowledge, but building on MNE primitives directly.
3. You keep full ownership. They are a reference, not a foundation.

---

## Key Architectural Decisions and Rationale

**Why .fif files as the intermediate format (not pickle or numpy)?**
MNE's .fif preserves all metadata: channel names, montage, sampling rate, event codes, bad channel list, projection operators. If you save a numpy array, you lose all of that and have to pass it around separately. When you load a `-clean-epo.fif` six weeks later, you can inspect it with `mne.read_epochs()` and immediately see what it contains. Pickle is fragile across Python versions and has no schema.

**Why subject-dependent evaluation first?**
KaraOne has 12 subjects. Cross-subject generalization requires domain adaptation techniques that are research-level complexity. Subject-dependent (train/test on same subject) is the established baseline in the literature and gives you something working quickly.

**Why SVM before neural networks?**
SVM with RBF kernel on band-power features is the most frequently reported baseline in imagined speech literature (confirmed in PMC 2024 systematic review). If your EEGNet cannot beat a well-tuned SVM, something is wrong with either the architecture, the data, or the training loop. The SVM baseline is the diagnostic tool.

**Why config.yaml instead of argparse?**
Solo learning project. You want to change filter frequencies and re-run without touching code. A flat YAML config also doubles as documentation of what you tried.
