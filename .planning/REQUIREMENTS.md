# Requirements — EEG Imagined Speech Pipeline

*Auto-derived from project description + domain research. All v1 requirements are hypotheses until validated.*

---

## v1 Requirements

### DATA — Data Access and Loading

- [ ] **DATA-01**: User can download all 14 KaraOne subjects' data to local disk using the AshrithSagar download script
- [ ] **DATA-02**: User can load a single subject's .mat file (MATLAB 7.3 HDF5 format) into MNE Epochs using mat73, with correct EEG channels, labels from ID_p.txt, and imagined-speech-only trial isolation via epoch_inds.mat
- [ ] **DATA-03**: User can visualize raw EEG traces and power spectral density for a loaded subject to visually confirm signal quality

### PREPROC — Preprocessing Pipeline

- [ ] **PREPROC-01**: Pipeline applies correct preprocessing order (bandpass filter 1-45 Hz → notch 60 Hz → drop non-EEG channels → bad channel detection → average re-reference → ICA → epoch rejection ±100µV) on continuous raw data before epoching
- [ ] **PREPROC-02**: Pipeline removes non-EEG channels (EMG/Kinect color sensor, M1, M2, EKG, Trigger) before any analysis
- [ ] **PREPROC-03**: ICA automatically labels and removes ocular artifact components using mne-icalabel and KaraOne's 4 EOG channels
- [ ] **PREPROC-04**: Pipeline runs on all 14 subjects and saves clean epochs as .fif files with per-subject rejection rate logs

### BASELINE — Baseline Classifier

- [ ] **BASE-01**: Pipeline extracts band-power features (Welch PSD, theta/alpha/beta bands, per channel) from clean epochs
- [ ] **BASE-02**: SVM classifier with RBF kernel trained using GroupKFold cross-validation (never standard KFold or train_test_split on pre-segmented epochs) with StandardScaler fitted on training data only
- [ ] **BASE-03**: Results aggregated per subject (accuracy, confusion matrix, per-class F1) and saved to JSON; summary CSV generated
- [ ] **BASE-04**: Permutation test (1000 shuffles) run to confirm accuracy significantly exceeds empirical chance level; binary subproblems (vowel/consonant) hit ≥65% as pipeline sanity check

### DL — Deep Learning Models

- [ ] **DL-01**: EEGNet model (via braindecode) trained on raw epoch arrays, no hand-crafted features; compared per-subject against SVM baseline
- [ ] **DL-02**: DWT (Daubechies-4) feature extraction implemented as alternative to band-power, compatible with sklearn pipeline
- [ ] **DL-03**: CNN-BiLSTM model implemented and trained; results compared to EEGNet and SVM across all subjects

### EVAL — Evaluation Infrastructure

- [ ] **EVAL-01**: All model runs log config (YAML) + results (JSON) automatically; summary CSV appended per run — no results lost due to missing logging
- [ ] **EVAL-02**: Confusion matrix and topographic map visualizations generated per subject per model
- [ ] **EVAL-03**: Majority voting across channels implemented and benchmarked (per-channel classifier → vote)

### AUDIO — Audio Engineering Side Quest

- [ ] **AUDIO-01**: Single EEG channel exported as 32-bit float mono WAV resampled to audible range; round-trip import back to MNE RawArray verified
- [ ] **AUDIO-02**: Three-way comparison implemented: MNE ICA denoising vs. iZotope RX spectral denoise vs. no denoising, with downstream classification accuracy as the metric

---

## v2 Requirements (Deferred)

- Vocalized-to-imagined transfer learning (train on vocalized, test on imagined)
- 8-channel simulation experiment (select OpenBCI-matching positions from 64-ch KaraOne, retrain, document accuracy drop)
- Cross-subject generalization (domain adaptation — requires within-subject results first)
- Personal OpenBCI Cyton hardware recording pipeline
- Optuna/Ray Tune hyperparameter optimization
- GAN/VAE data augmentation for low-trial subjects

---

## Out of Scope

- Real-time inference — offline/research only
- Web UI or deployed service
- Multi-modal fusion (EEG + audio + face tracking)
- Source localization
- Custom novel architecture design (EEGNet and CNN-BiLSTM are sufficient for learning)
- Automated hyperparameter search (manual configuration for now)

---

## Traceability

*(Filled by roadmapper)*

| REQ-ID | Phase |
|--------|-------|
| DATA-01 | 1 |
| DATA-02 | 1 |
| DATA-03 | 1 |
| PREPROC-01 | 2 |
| PREPROC-02 | 2 |
| PREPROC-03 | 2 |
| PREPROC-04 | 2 |
| BASE-01 | 3 |
| BASE-02 | 3 |
| BASE-03 | 3 |
| BASE-04 | 3 |
| DL-01 | 4 |
| DL-02 | 4 |
| DL-03 | 4 |
| EVAL-01 | 3 |
| EVAL-02 | 3 |
| EVAL-03 | 3 |
| AUDIO-01 | 5 |
| AUDIO-02 | 5 |
