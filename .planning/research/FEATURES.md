# Features Research — EEG Imagined Speech Pipeline

**Researched:** 2026-04-27
**Dataset:** KaraOne (64-ch, 14 subjects, 1 kHz, 11-class imagined speech)
**Starting repo:** AshrithSagar/EEG-Imagined-speech-recognition
**SOTA target:** 57% (wavelet+DNN, 11-class KaraOne)

---

## Table Stakes (must have)

These are the things the pipeline cannot function without. Without them you will either get garbage results or results you cannot trust.

### 1. Data Loader for KaraOne Format

The raw KaraOne data is distributed as Neuroscan `.cnt` or `.mat` files per subject, with separate epoch markers for the four trial phases (rest, stimulus, imagined speech, vocalized speech). You need a loader that isolates the imagined speech epochs specifically — not the stimulus window or vocalization window.

- Load per-subject, per-session files
- Select only the imagined speech phase (5-second window after stimulus offset)
- Expose subject list and class labels as a consistent interface
- Output: MNE Raw or Epochs objects

Effort: **Medium** (MNE handles the heavy lifting; the work is mapping KaraOne's marker scheme to clean epoch labels)

### 2. Bandpass Filter (1–45 Hz)

The original KaraOne paper applied 1–50 Hz. Most 2024 literature uses 1–45 Hz to cut high-frequency muscle artifact before the notch filter. This is non-negotiable — raw 1 kHz EEG contains massive amounts of noise outside the band of interest (delta through gamma: 1–45 Hz). Imagined speech signal lives primarily in theta (4–8 Hz), alpha (8–13 Hz), and low gamma (30–45 Hz).

Apply before epoching, not after. Use zero-phase FIR (MNE default `mne.filter.filter_data` with `method='fir'`).

Effort: **Low**

### 3. Notch Filter (50/60 Hz)

Line noise. KaraOne was recorded in Canada so 60 Hz. Skip this and your spectral features will be dominated by powerline artifact.

Effort: **Low**

### 4. ICA-Based Ocular Artifact Removal

The KaraOne protocol included 4 EOG electrodes (above/below left eye, lateral both eyes) specifically for this purpose. ICA decomposition isolates eye-blink and lateral eye-movement components, which are the dominant artifact source in imagined speech tasks where subjects are seated and looking at a screen.

Practical approach: fit ICA on the continuous filtered data, auto-label EOG components using MNE's `find_bads_eog()`, subtract them. This is what the original KaraOne paper did ("blind source separation"). Manual inspection of at least one subject's components is worth doing once to sanity-check the auto-detection.

Effort: **Medium** (MNE makes this ~20 lines of code; the time cost is understanding what you're looking at)

### 5. Epoching with Correct Trial Windows

KaraOne's imagined speech phase is 5 seconds. Most literature crops this to 2–4 seconds to avoid late-trial noise and reduce compute. The canonical choice in papers achieving ~57% is to use the full 5-second window but apply windowing with ~50% overlap for feature extraction. You need a clean, reproducible epoching step that:

- Aligns to the imagined speech onset marker
- Applies a baseline correction (subtract mean of a pre-stimulus rest period)
- Rejects epochs with amplitude > 100 µV (gross artifact catch)

Effort: **Low–Medium**

### 6. Feature Extraction (at minimum: power spectral / wavelet)

The 57% SOTA result on 11-class KaraOne used wavelet + temporal domain features with majority voting across channels. Two approaches are table stakes; pick one to start:

**Option A — Band power features:** Compute per-channel power in delta, theta, alpha, beta, low-gamma bands using Welch PSD. Simple, fast, interpretable. Expect ~35–45% accuracy.

**Option B — Discrete Wavelet Transform (DWT) features:** Decompose each epoch with DWT (Daubechies-4 is standard), extract statistical features (mean, variance, energy, entropy, kurtosis) from each sub-band per channel. This is what achieved 57%. More features, slightly more compute.

Both are implemented or easily added to MNE + scipy.

Effort: **Low** (band power) / **Medium** (DWT feature matrix)

### 7. Classifier with Proper Cross-Validation

At minimum: SVM with RBF kernel or Random Forest. At minimum: k-fold cross-validation that does NOT split within a subject's session randomly. The correct approach is leave-one-trial-out or leave-one-block-out, NOT random shuffle of all epochs.

Data leakage from bad CV splits is the #1 way to get inflated results that don't replicate. See Anti-Features section for detail.

Effort: **Low** (scikit-learn handles this; the effort is enforcing the right split strategy)

### 8. Per-Subject Results Reporting

KaraOne has 14 subjects with high inter-subject variability. Reporting only the mean accuracy hides the fact that some subjects are near chance and others are well above. Every result should report mean ± std across subjects, plus per-subject accuracy. This is how the literature reports it and is necessary to understand whether your pipeline is working.

Effort: **Low**

### 9. Chance-Level Baseline

11-class classification has a chance level of 9.1%. 2-class subsets have 50% chance. Every result must be compared against chance, and ideally against a permutation-test baseline (shuffle labels, re-train, check that your real accuracy is significantly above the null distribution).

Effort: **Low**

---

## Standard Research Features (good to have)

These are in most serious EEG research pipelines. They improve trustworthiness of results and enable meaningful comparisons.

### 10. Downsampling to 256 Hz

KaraOne is recorded at 1 kHz. Imagined speech signal is fully captured below 100 Hz, so 1 kHz is 10x oversampled. Downsample after filtering to 256 Hz. This reduces storage by 4x and compute by 4x for feature extraction and model training. MNE `resample()` handles anti-aliasing automatically.

Effort: **Low**

### 11. Bad Channel Detection and Interpolation

Any 64-channel recording will have occasional bad channels per subject (poor contact, electrode pop). MNE's automated bad channel detection (`find_bad_channels_maxwell` or simpler variance-based detection) flags these, and spherical spline interpolation fills them. Without this, one noisy channel contaminates your whole feature matrix.

Effort: **Low–Medium**

### 12. Common Average Reference (CAR)

Subtracting the average of all channels from each channel removes globally shared noise (e.g., movement artifact affecting the whole cap). Standard in EEG research. One line in MNE: `raw.set_eeg_reference('average')`. The original KaraOne paper used a small Laplacian reference instead — either is reasonable, but CAR is simpler.

Effort: **Low**

### 13. Time-Frequency Analysis Visualization (ERSP/ERD)

Event-Related Spectral Perturbation shows you where in time and frequency the brain responds to each stimulus class. This is the fundamental exploratory visualization for EEG. Without it you're flying blind — you don't know if your features are actually capturing anything meaningful.

Use MNE's `tfr_multitaper` or `tfr_morlet` on the epochs. Plot the average ERSP per class. Look for gamma suppression (~30 Hz) and theta/alpha changes in the 200–2000 ms window after imagined speech onset.

Effort: **Medium** (the computation is easy; interpreting what you see takes domain knowledge)

### 14. Confusion Matrix and Per-Class F1

Some of KaraOne's 11 classes are much harder to distinguish than others (e.g., /m/ vs /n/ are both nasals; "pot" vs "pat" differ only in vowel). A confusion matrix reveals which classes your classifier systematically confuses. Per-class F1 is more informative than overall accuracy for an imbalanced 11-class problem.

Effort: **Low**

### 15. Experiment Config Logging (YAML + results CSV)

A solo research project generates many variants: different filter cutoffs, different feature sets, different classifiers. Without logging what you ran and what it produced, you will lose track of your best configuration. The minimal viable system is: a YAML config file per experiment run, a results CSV that appends one row per run (config hash, accuracy, std, date). This is not MLflow — just two files.

Effort: **Low**

### 16. Per-Subject Preprocessing Report

Generate one diagnostic plot per subject showing: raw vs. filtered signal, ICA components removed, epoch rejection rate, channel quality. This takes 30 minutes to build and saves hours of debugging later when one subject's results look strange.

Effort: **Medium**

---

## Differentiators

Things that would push accuracy or insight beyond the baseline for this specific project.

### 17. Majority Voting Across Channels

The 57% SOTA paper achieved its result partly through majority voting: train a separate classifier per channel (or small channel group), then vote across channels for the final prediction. This is ensemble learning applied at the spatial level. It's not standard in all pipelines but is directly validated on KaraOne. Worth implementing as a second-pass experiment after getting baseline accuracy.

Effort: **Medium**

### 18. Channel Selection / Electrode Reduction

2025 research shows 64 channels can be reduced to ~32 with no significant accuracy drop — but the optimal channels are subject-specific, not universal. Implement a simple filter-based channel selection (select top-N channels by Fisher score or mutual information with class labels) and test whether using fewer channels degrades or maintains accuracy. This is also directly relevant to the future OpenBCI goal (16 channels max).

Key finding: relevant electrodes are distributed across the cortex, not limited to left hemisphere speech areas.

Effort: **Medium**

### 19. Topographic Map Visualization

Convert your band-power features into a 2D scalp topography plot per class. This shows where in the brain the discriminative information is coming from. One topographic plot per class, averaged across subjects, is a publishable-quality figure and also helps debug whether your features are picking up genuine neural signal vs. artifact patterns.

MNE's `plot_topomap` handles this.

Effort: **Low** (if features are already per-channel)

### 20. Vocalized vs. Imagined Comparison

KaraOne includes both imagined and vocalized speech trials. Training on vocalized, testing on imagined (or using vocalized as auxiliary training data) is a documented strategy for improving imagined speech accuracy. 2025 research ("From pronounced to imagined") showed this meaningfully improves decoding. This is a natural experiment given the data is already there.

Effort: **Medium**

### 21. Binary Subproblem Decomposition

Rather than tackling 11-class directly, build binary classifiers for the linguistic subproblems the original KaraOne paper studied: vowel vs. consonant, ±nasal, ±bilabial, ±/iy/, ±/uw/. These achieve much higher accuracy (some papers report >90% for binary tasks on KaraOne) and are more interpretable. Use these as validation that your preprocessing is working before attempting 11-class.

Effort: **Low** (just relabeling)

### 22. EEGNet Baseline Model

EEGNet (Lawhern et al., 2018) is a compact CNN specifically designed for EEG that works on raw or minimally processed epochs. It is the de facto baseline DL model in EEG-BCI research — if you can't beat EEGNet, you need to rethink your approach. It trains fast on small datasets and has very few parameters. The 2025 literature compares EEGNet vs. EEG-Conformer vs. ShallowConvNet on imagined speech; EEG-Conformer achieves ~79% while EEGNet gets ~75% on non-KaraOne datasets.

Effort: **Medium** (braindecode library has a ready implementation)

---

## Anti-Features (explicitly don't build)

Scope creep traps for a solo research project. Each of these sounds reasonable but is a rabbit hole.

### DO NOT: Real-Time Inference Engine

You are building a research pipeline, not a BCI system. Real-time processing requires streaming data handling, low-latency model inference, and online artifact detection — completely different engineering than offline batch processing. The KaraOne data is pre-recorded. Defer until you have working offline accuracy and actual OpenBCI hardware in hand.

### DO NOT: Cross-Subject Generalization Model

Training a model that works on unseen subjects (Leave-One-Subject-Out) is a research problem that has its own literature and is significantly harder than within-subject classification. KaraOne has 14 subjects; within-subject is already the hard problem. Cross-subject domain adaptation (Standardization-Refinement Domain Adaptation and similar approaches) is graduate-thesis territory. This is a "later, maybe" after you have strong within-subject results.

### DO NOT: Custom Neural Architecture Search

Do not design your own CNN or Transformer architecture. The field already has EEGNet, EEG-Conformer, ShallowConvNet, and LSTM variants benchmarked on this problem. The value you can add is in preprocessing, feature engineering, and experimental design — not architecture novelty. Use braindecode or a reference implementation and focus compute budget elsewhere.

### DO NOT: GUI or Interactive Dashboard

A research pipeline is run from a script or notebook. You do not need a Streamlit app, a web UI, or an interactive parameter tuner. Every hour spent on UI is an hour not spent on signal processing or understanding the domain. Results go in a CSV and plots go in a folder.

### DO NOT: Automated Hyperparameter Optimization Framework

Optuna, Ray Tune, hyperopt — these are production ML tools. For a 14-subject, ~50-trial-per-class dataset, manual grid search over 3–5 key hyperparameters is sufficient and more interpretable. The risk is spending a week tuning a model that is fundamentally limited by the dataset size.

### DO NOT: Multi-Modal Fusion (EEG + Face Tracking + Audio)

KaraOne provides facial tracking and audio modalities alongside EEG. Some papers achieve >99% binary accuracy by combining all three. This tells you nothing about EEG decoding — you want to know what the brain signal alone can do. Stick to EEG-only until you have a strong baseline, then optionally add audio as an upper bound reference.

### DO NOT: Generative Data Augmentation (GAN/VAE)

GAN-based EEG augmentation is an active research area and sounds appealing for KaraOne's small dataset. In practice it is extremely difficult to validate that synthetic EEG samples are neurologically plausible rather than statistical hallucinations that inflate accuracy. Sliding window augmentation (standard, validated) is sufficient. Generative augmentation is a research project in itself.

### DO NOT: Source Localization / Beamforming

Mapping scalp EEG back to brain source locations requires MRI co-registration, head modeling, and inverse problem solving. It is the next level of sophistication after surface-level classification and requires anatomical data KaraOne does not provide per subject. Not needed to achieve above-chance classification.

---

## The Audio Engineering Side Quest

### Is the iZotope RX vs. MNE ICA comparison feasible?

**Short answer: Yes, but it requires a creative bridge layer. It is a genuine novel experiment — no published work does this comparison.**

### Why It's Interesting

EEG and audio share the same mathematical structure: both are 1D time-series signals sampled at fixed rates, both contain broadband noise with structured artifact components (in audio: hiss, hum, clicks; in EEG: eye blinks, muscle, line noise), and both have been attacked with spectral subtraction and time-frequency masking methods. iZotope RX's Spectral De-noise uses a learned noise model (STFT-based Wiener filtering with adaptive masking) that is conceptually identical to some EEG artifact removal approaches.

The comparison is: does treating each EEG channel as an audio track and running RX's spectral denoiser improve downstream classification, hurt it, or produce the same result as ICA?

### Why It's Not Straightforward

iZotope RX operates on 16-bit or 32-bit PCM audio WAV files at audio sample rates (typically 44.1 kHz or 48 kHz). EEG is recorded at 1 kHz and stored in µV units. The bridge requires:

1. **Export each EEG channel as a mono WAV file** — normalize µV values to float32 in [-1, 1] range, write at 1 kHz sample rate (or resample to 44.1 kHz, process, then resample back)
2. **Apply RX Spectral De-noise** — use the "learn noise" function on a rest-state segment, apply model to the full channel recording
3. **Re-import processed audio as EEG** — read back the WAV, rescale to µV, reconstruct the MNE Raw object

The conceptual mismatch to flag: EEG artifact (especially eye blinks) is not stationary noise — it is a transient, spatially structured event that ICA is specifically designed to handle by exploiting cross-channel independence. RX's spectral denoiser treats each channel independently and models noise as stationary or slowly varying. It will likely handle broadband EMG noise and electrode drift well but struggle with blink artifacts, which are non-stationary and low-frequency.

### Recommended Experiment Design

Run both preprocessing branches and evaluate classification accuracy with the exact same feature extraction and classifier downstream:

- **Branch A:** Bandpass → Notch → ICA (MNE standard)
- **Branch B:** Bandpass → Notch → Per-channel RX Spectral De-noise (audio engineering approach)
- **Branch C:** Bandpass → Notch only (baseline, no artifact removal)

Compare per-subject accuracy across branches. Also compare the power spectral density of cleaned signals to assess how much signal the two methods destroy vs. preserve in the 4–45 Hz band of interest.

The experiment is quantifiable, novel, and directly relevant to the iZotope RX tooling you already know. Publication-worthy as a methods comparison note if results are interesting.

### Practical Constraint

iZotope RX requires manual interaction unless you use the RX Connect or AudioSuite plug-in with a DAW that supports batch processing, or the command-line iZotope RX Batch processor (available in RX Advanced). Check if your RX license includes batch export — if not, processing 14 subjects × 64 channels manually is impractical. The workaround is to use Python's `soundfile` library to write WAVs, process with the RX plug-in in Reaper via a headless batch script, and re-import.

Effort for the audio engineering side quest: **High** (bridge layer engineering + batch processing setup) but the scientific comparison is **Medium** (well-defined, simple to evaluate once data is flowing)

---

## Complexity Ratings Summary

| Feature | Effort | Priority |
|---|---|---|
| 1. KaraOne data loader | Medium | Table stakes |
| 2. Bandpass filter | Low | Table stakes |
| 3. Notch filter | Low | Table stakes |
| 4. ICA artifact removal | Medium | Table stakes |
| 5. Correct epoching + baseline | Low-Medium | Table stakes |
| 6. DWT / band-power features | Low-Medium | Table stakes |
| 7. SVM / RF + proper CV | Low | Table stakes |
| 8. Per-subject results reporting | Low | Table stakes |
| 9. Chance-level baseline | Low | Table stakes |
| 10. Downsample to 256 Hz | Low | Standard |
| 11. Bad channel detection | Low-Medium | Standard |
| 12. Common average reference | Low | Standard |
| 13. ERSP time-frequency visualization | Medium | Standard |
| 14. Confusion matrix + per-class F1 | Low | Standard |
| 15. Experiment config + results logging | Low | Standard |
| 16. Per-subject preprocessing report | Medium | Standard |
| 17. Majority voting across channels | Medium | Differentiator |
| 18. Channel selection / electrode reduction | Medium | Differentiator |
| 19. Topographic map visualization | Low | Differentiator |
| 20. Vocalized vs. imagined comparison | Medium | Differentiator |
| 21. Binary subproblem decomposition | Low | Differentiator |
| 22. EEGNet baseline model | Medium | Differentiator |
| RX vs. ICA audio side quest | High | Side quest |

---

## Sources

- [Systematic Review of EEG-Based Imagined Speech Classification Methods (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679664/)
- [Decoding Imagined Speech: Hybrid Deep Learning (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11595501/)
- [KARA ONE Database — University of Toronto](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html)
- [Decoding Imagined Speech using Wavelet Features and DNNs (arXiv 2020) — 57% result](https://arxiv.org/abs/2003.10433)
- [Data leakage in deep learning EEG studies (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/)
- [Electrode reduction for EEG speech imagery BCIs (Frontiers 2025)](https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2025.1578586/full)
- [MNE-Python ICA documentation](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html)
- [AshrithSagar/EEG-Imagined-speech-recognition (starting repo)](https://github.com/AshrithSagar/EEG-Imagined-speech-recognition)
- [Matt-Golightly/MDS_Kara_One (reference implementation)](https://github.com/Matt-Golightly/MDS_Kara_One)
- [EEGNet / braindecode architecture comparisons (Frontiers 2025)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1668935/full)
- [Data Contamination in Brain-to-Text Decoding (arXiv)](https://arxiv.org/html/2312.10987v2)
- [Transfer Learning for Imagined Speech EEG (arXiv 2025)](https://arxiv.org/html/2502.04132v1)
