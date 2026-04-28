# Roadmap — EEG Imagined Speech Pipeline

**5 phases** | **19 v1 requirements** | All covered ✓

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Data Access & Exploration | One subject's raw EEG is loaded into MNE with correct labels and epochs isolated | DATA-01, DATA-02, DATA-03 | 3 criteria |
| 2 | Preprocessing Pipeline | All 14 subjects have clean, artifact-removed epoch files on disk | PREPROC-01, PREPROC-02, PREPROC-03, PREPROC-04 | 4 criteria |
| 3 | Baseline Classifier + Evaluation | SVM classifier with proper CV produces above-chance per-subject results with full logging | BASE-01, BASE-02, BASE-03, BASE-04, EVAL-01, EVAL-02, EVAL-03 | 5 criteria |
| 4 | Deep Learning Models | EEGNet and CNN-BiLSTM trained and compared against SVM baseline per subject | DL-01, DL-02, DL-03 | 4 criteria |
| 5 | Audio Engineering Experiment | iZotope RX denoise compared to MNE ICA with classification accuracy as the metric | AUDIO-01, AUDIO-02 | 3 criteria |

---

## Phase 1: Data Access & Exploration

**Goal:** One subject's raw .mat file is loaded into MNE with correct labels, imagined-speech-only trials isolated, and EEG signal quality visually confirmed.
**Requirements:** DATA-01, DATA-02, DATA-03
**Depends on:** Nothing
**UI hint**: no

### Success Criteria
1. `download-karaone.py` completes without error and all 14 subjects' .mat files exist on local disk
2. `src/loader.py` produces an MNE Epochs object for a single subject with correct shape `(n_epochs, 64, n_times)`, correct word labels from `ID_p.txt`, and imagined-speech trials only (via `epoch_inds.mat`)
3. Raw EEG traces and PSD plots for the loaded subject are renderable from a notebook, visually showing electrode activity and a clean 1/f spectral slope (no obvious 60 Hz spike at this stage)

### Tasks
- [ ] Set up Python environment with `uv` and install full stack (mne, mat73, h5py, pyprep, mne-icalabel, autoreject, scikit-learn, torch, braindecode, numpy, scipy, matplotlib)
- [ ] Run `download-karaone.py` for all 14 subjects; verify directory structure and file sizes
- [ ] Read `src/eeg_isr/karaone.py` in AshrithSagar repo to understand actual .mat internal structure before writing any loader code
- [ ] Write `src/loader.py`: mat73.loadmat → numpy → mne.RawArray → mne.Epochs, using `ID_p.txt` for labels and `epoch_inds.mat` for trial isolation; drop non-EEG channels
- [ ] Verify Epochs shape, label set, and trial count for at least 2 subjects (counts vary — do not hardcode 132)
- [ ] Write `notebooks/01_explore_raw.ipynb`: plot raw traces, compute and plot PSD, count trials per class

---

## Phase 2: Preprocessing Pipeline

**Goal:** All 14 subjects have clean, artifact-removed epoch files saved as `.fif` with per-subject rejection rate logs, ready for feature extraction.
**Requirements:** PREPROC-01, PREPROC-02, PREPROC-03, PREPROC-04
**Depends on:** Phase 1
**UI hint**: no

### Success Criteria
1. Non-EEG channels (EMG/Kinect, M1, M2, EKG, Trigger) are removed before any filtering or referencing step
2. Preprocessing order is enforced: bandpass (1–45 Hz FIR) → notch (60 Hz) → bad channel detection (pyprep) → average re-reference → ICA → epoch rejection (±100 µV) — all on continuous raw data before epoching
3. ICA automatically identifies and removes ocular artifact components using mne-icalabel and KaraOne's 4 EOG channels; each subject's run logs how many components were removed
4. All 14 subjects produce `data/processed/subject_XX-clean-epo.fif` files and a per-subject JSON log showing rejection rate (expected 10–30%), ICA component count removed (expected 1–5), and clean trial counts per class

### Tasks
- [ ] Write `src/preprocessor.py` implementing the full pipeline in correct order using MNE + pyprep + mne-icalabel + autoreject
- [ ] Implement non-EEG channel removal as the first step (before filtering), mapping KaraOne's specific channel names
- [ ] Implement ICA with mne-icalabel auto-labeling; log component labels and removal decisions per subject
- [ ] Implement epoch rejection at ±100 µV as the final step; log rejection rate per subject
- [ ] Batch-run preprocessor across all 14 subjects; save `.fif` files to `data/processed/`
- [ ] Verify PSD of cleaned epochs shows clean 1/f slope with no 60 Hz spike; check temporal channels (T7, T8, FT7, FT8) for absence of broadband shelf above 30 Hz (EMG marker)
- [ ] Write `notebooks/02_preprocess_qc.ipynb`: before/after PSD comparison, ICA component plots, rejection rate summary table

---

## Phase 3: Baseline Classifier + Evaluation Infrastructure

**Goal:** Band-power SVM classifier with GroupKFold CV produces per-subject accuracy results, confirmed above empirical chance by permutation test, with full logging infrastructure in place for all future model runs.
**Requirements:** BASE-01, BASE-02, BASE-03, BASE-04, EVAL-01, EVAL-02, EVAL-03
**Depends on:** Phase 2
**UI hint**: no

### Success Criteria
1. Band-power features (Welch PSD, theta/alpha/beta bands, per channel) are extracted from clean epochs and saved as `features_X.npy` + `labels_y.npy` per subject
2. SVM with RBF kernel runs under GroupKFold CV (never standard KFold or train_test_split); StandardScaler is fitted on training folds only; binary subproblems (vowel/consonant) reach ≥65% accuracy as pipeline sanity check
3. Per-subject results (accuracy, confusion matrix, per-class F1) are saved to `outputs/results/results_subjectXX_svm.json` and aggregated to `outputs/summary_results.csv`
4. Permutation test (1000 shuffles) confirms that accuracy significantly exceeds empirical chance (above 9.1% theoretical baseline for 11-class)
5. Every model run (current and future) automatically logs YAML config + JSON results; no run produces results that can't be reproduced from the log; majority-vote-across-channels benchmark is included in the evaluation suite

### Tasks
- [ ] Write `src/features.py`: Welch PSD computation per channel per band (theta 4–8 Hz, alpha 8–12 Hz, beta 12–30 Hz); output flattened feature vector per epoch
- [ ] Write `src/classifier.py`: StandardScaler → SelectKBest → SVM(rbf) under GroupKFold(5); include binary subproblem runs (vowel/consonant split)
- [ ] Write `src/evaluate.py`: aggregate per-subject JSONs → summary CSV; generate confusion matrix plots and topographic maps per subject per model
- [ ] Implement permutation test (1000 shuffles) in evaluate.py; log p-value per subject
- [ ] Set up `config.yaml` as single source of truth for all parameters (bands, n_components, C, gamma, CV folds, rejection threshold)
- [ ] Implement run logging: every run reads config.yaml, writes outputs/results/run_YYYYMMDD_HHMMSS/ with config copy + results JSON
- [ ] Implement majority voting: train per-channel classifiers, vote across channels, compare to full-feature SVM
- [ ] Write `notebooks/03_baseline_results.ipynb`: per-subject accuracy bar chart, confusion matrices, permutation test p-values

---

## Phase 4: Deep Learning Models

**Goal:** EEGNet and CNN-BiLSTM are trained on raw epoch arrays and benchmarked against the SVM baseline per subject, with DWT features as an alternative feature pathway.
**Requirements:** DL-01, DL-02, DL-03
**Depends on:** Phase 3
**UI hint**: no

### Success Criteria
1. EEGNet (via braindecode) trains on raw 3D epoch arrays without hand-crafted features and produces per-subject accuracy results that are directly comparable to SVM baseline in the summary CSV
2. DWT (Daubechies-4) feature extraction is implemented as a sklearn-compatible transformer, runnable as a drop-in alternative to band-power features in the existing sklearn pipeline
3. CNN-BiLSTM model is implemented, trained, and its results appear alongside EEGNet and SVM in the summary comparison; at least one model configuration exceeds 57% on at least one subject

### Tasks
- [ ] Write `src/models.py`: EEGNet wrapper (braindecode) + CNN-BiLSTM implementation (PyTorch); both must accept `(n_epochs, n_channels, n_times)` input
- [ ] Write `src/train.py`: training loop with early stopping, learning rate schedule, and checkpoint saving to `outputs/models/`; integrate run logging from Phase 3 infrastructure
- [ ] Implement GroupKFold-compatible DL evaluation (no data leakage — same grouping as SVM)
- [ ] Write `src/features.py` DWT path: PyWavelets Daubechies-4 decomposition per channel → feature vector compatible with existing sklearn pipeline
- [ ] Run EEGNet on all 14 subjects; compare per-subject to SVM in summary CSV
- [ ] Run CNN-BiLSTM on all 14 subjects; compare per-subject to EEGNet and SVM
- [ ] Write `notebooks/04_dl_results.ipynb`: three-way comparison table (SVM vs EEGNet vs CNN-BiLSTM), per-subject accuracy plots, training loss curves

---

## Phase 5: Audio Engineering Experiment

**Goal:** A single EEG channel survives WAV round-trip intact, and a three-way denoising comparison (MNE ICA vs. iZotope RX vs. no denoising) is benchmarked using downstream classification accuracy.
**Requirements:** AUDIO-01, AUDIO-02
**Depends on:** Phase 3 (needs baseline classifier to measure denoising impact)
**UI hint**: no

### Success Criteria
1. A single EEG channel exported as 32-bit float mono WAV (resampled to audible range) can be re-imported as an MNE RawArray with signal fidelity confirmed (round-trip error is negligible and documented)
2. Three-way denoising comparison runs end-to-end: raw epochs → MNE ICA path, raw epochs → iZotope RX WAV path, raw epochs → no denoising; each feeds the same SVM classifier and results are logged to the standard run format
3. Comparison output clearly shows whether iZotope RX spectral denoise improves, degrades, or has no effect on classification accuracy relative to MNE ICA baseline

### Tasks
- [ ] Write `src/audio_export.py`: MNE RawArray → resample to 8–44.1 kHz range → 32-bit float WAV using scipy.io.wavfile or soundfile
- [ ] Verify round-trip: WAV → re-import → MNE RawArray; compute and log reconstruction error (MSE, max deviation)
- [ ] Process one subject's single channel through iZotope RX spectral denoise (manual step); document settings used
- [ ] Write `src/audio_import.py`: 32-bit float WAV → resample back to original 1kHz → MNE RawArray
- [ ] Wire iZotope RX output back into preprocessing pipeline as an alternative denoising path (skipping ICA)
- [ ] Run SVM classifier (from Phase 3) on all three denoising conditions; log results using existing run logging infrastructure
- [ ] Write `notebooks/05_audio_experiment.ipynb`: WAV round-trip validation, three-way accuracy comparison table, spectrograms of raw vs. ICA vs. iZotope RX output
