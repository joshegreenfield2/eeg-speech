# Pitfalls Research — EEG Imagined Speech

**Domain:** EEG-based imagined speech decoding (BCI)
**Dataset:** KaraOne (64-ch, 14 subjects, ~132 trials/subject, 1kHz, MATLAB .mat)
**Target stack:** MNE-Python, scikit-learn, PyTorch
**Researched:** 2026-04-28
**Overall confidence:** HIGH (multiple peer-reviewed sources, replicated findings)

---

## Data Leakage Traps

This is the single largest source of inflated results in published EEG ML papers. A meta-analysis of 63 published translational EEG deep learning studies found only 17 (27%) properly avoided it. You will almost certainly fall into at least one of these traps without explicit awareness.

### Trap 1: Segment-level cross-validation (the most common mistake)

**What happens:** You slice each ~5-second KaraOne trial into sub-windows (e.g., 500ms segments with 250ms overlap), then randomly shuffle all segments into train/test splits. Segments from the same subject's same trial appear in both sets.

**Why it inflates accuracy:** EEG from one subject is far more similar to other EEG from that same subject than to EEG from a different subject. The model learns each person's idiosyncratic brain "fingerprint" and uses it as the classification signal. The classifier is effectively doing person identification disguised as word classification.

**Concrete numbers from the literature:** An Alzheimer's study got 99.8% accuracy with segment-level splits. With proper subject-based holdout, the same model dropped to 53% — essentially chance. A seizure detection study dropped 14 percentage points (79% to 65%) with the same correction.

**Prevention:** Always use `GroupKFold` or `LeaveOneGroupOut` from scikit-learn with subject ID as the group. Never allow `train_test_split` or standard `KFold` on pre-segmented data unless you have already ensured all segments from one subject go to exactly one fold.

```python
# WRONG — leaks subject identity across folds
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(all_segments, test_size=0.2)

# CORRECT — keeps subjects intact per fold
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
    ...
```

### Trap 2: Overlapping windows straddling the train/test boundary

**What happens:** You create overlapping sliding windows (e.g., 500ms window, 250ms stride). Some windows near the train/test split boundary contain samples from both partitions. The model sees future test data during training.

**Prevention:** Split at the trial level first, then create windows within each partition. Never window the full continuous signal before splitting.

### Trap 3: Normalization/scaling fitted on the full dataset

**What happens:** You compute mean and standard deviation across all epochs (train + test combined), then normalize. The scaler has now "seen" the test set and leaks distributional information into the model.

**Prevention:** Fit the scaler only on training data. Apply (transform only) to test data.

```python
# WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)  # includes test data

# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only
```

### Trap 4: ICA/artifact removal computed on full dataset

**What happens:** You run ICA on all epochs together, identify artifact components, and remove them — then split into train/test. The ICA decomposition has been informed by test-set data.

**Prevention:** In strict pipelines, ICA should be fit on training data only. In practice for research, many papers run ICA on the full continuous recording (before epoching), which is generally considered acceptable because it doesn't use label information. Be explicit about which approach you are using and why.

### Trap 5: Feature selection using all data before splitting

**What happens:** You compute which channels or frequency bands are most informative across the full dataset, then use those features for both training and testing. The test set's labels guided your feature selection.

**Prevention:** All feature selection decisions (channel selection, frequency band selection, dimensionality reduction) must be made using only training data. Wrap feature selection inside your cross-validation loop.

### Trap 6: Hyperparameter tuning against the test set

**What happens:** You try different model architectures or preprocessing parameters and pick whichever gives the best test accuracy. The test set has become a validation set — it is no longer an unbiased estimate.

**Prevention:** Use a three-way split (train / validation / test) or nested cross-validation. The test set is touched exactly once, after all decisions are finalized.

### Trap 7: Temporal autocorrelation in k-fold CV

**What happens:** Standard k-fold CV randomly assigns time-series samples to folds, ignoring that adjacent EEG samples are highly correlated. Training folds "know about" nearby test-fold samples. Research shows this inflates accuracy by up to 25 percentage points in some EEG passive BCI designs.

**Prevention:** Use block-wise cross-validation or trial-based splits where temporal ordering is respected. Report both random-split and block-wise results if comparing to literature.

---

## Preprocessing Mistakes

### Mistake 1: Filtering after epoching

**What happens:** You epoch the continuous data first, then apply bandpass filter to each epoch individually. Edge effects (filter ringing) contaminate the beginning and end of every epoch, and filters applied to short segments have poor frequency resolution.

**Correct order:**
1. Load raw continuous data
2. Apply filters to continuous data
3. Epoch the filtered data (with sufficient pre/post buffer to absorb edge effects)
4. Crop epochs to final analysis window

### Mistake 2: High-pass filter cutoff too aggressive

**What happens:** Using a high-pass filter at 1Hz or higher to remove drift. High-pass filters with cutoffs above ~0.1 Hz can cause temporal smearing — activity from late time points (like the N400 or P600) gets reflected backward into earlier latencies. This creates the appearance of effects that aren't there, or masks real ones.

**Prevention:** For ERPs and slow cortical potentials relevant to speech, use 0.1 Hz high-pass. Only go higher (1 Hz) if you are exclusively working with higher-frequency oscillatory features and have verified the effect on your specific data.

### Mistake 3: Treating high gamma (60-120 Hz) as clean neural signal

**What happens:** You include high-gamma features because papers report them improving accuracy. You may actually be decoding EMG muscle artifacts, not brain signals.

**Why this is critical for imagined speech:** Even imagined speech induces subtle jaw, tongue, and facial muscle artifacts. These appear in EEG at the same frequencies as high-gamma neural activity. Papers in this field have explicitly speculated that their reported high-gamma accuracy gains "might be due to EMG artifacts." Temporal and peripheral electrodes are most affected.

**How to detect it:** Look at the power spectral density (PSD) of your data. Genuine neural gamma activity appears as a narrow-band peak (~20 Hz bandwidth). EMG contamination appears as a broadband shelf spanning 20-300 Hz, maximal at peripheral electrodes (temporal, mastoid-adjacent). A flat spectral shelf above 30 Hz at temporal channels is almost certainly muscle.

**Prevention:** Either avoid frequencies above 50 Hz, or verify artifact removal is effective by checking PSDs at peripheral electrodes before and after ICA. Do not report results using high-gamma features without explicit artifact verification.

### Mistake 4: Wrong epoch window for KaraOne

**What happens:** KaraOne's imagined speech state is 5 seconds. If you epoch too wide (including the cue/prompt period or the overt speech period that follows), your "imagined speech" epochs contain overt speech or cue-processing activity. If you epoch too narrow (just 1-2 seconds), you may miss the bulk of imagined speech signal.

**KaraOne specifics:** The dataset includes both imagined and vocalized conditions. The epoch_inds.mat file contains trial indices. The prompt ordering in ID_p.txt (actual) sometimes differs from ID.txt (intended). Always use the actual presentation order (ID_p.txt), not the intended order.

**Prevention:** Read the KaraOne paper carefully. Verify your epoch window against the experimental protocol. Visualize grand-averaged ERPs per condition to confirm the stimulus-locked response aligns with what you expect.

### Mistake 5: Wrong or missing re-referencing

**What happens:** KaraOne records with a specific reference electrode. Many analyses assume average reference. If you load the .mat file and treat the raw voltage values as-is without re-referencing, your channel values are referenced to whatever the original recording used (linked mastoids or Cz, likely). This does not match most published methods.

**Prevention:** Apply average reference after loading the data and removing bad channels. Do not re-reference before removing bad channels, as bad channels corrupt the average.

```python
raw.set_eeg_reference('average', projection=True)
raw.apply_proj()
```

### Mistake 6: Not removing non-EEG channels before processing

**KaraOne-specific:** The EMG channel in KaraOne contains color sensor data, not EMG. Channels M1, M2, EKG, and Trigger are not EEG channels. If you include them in your EEG analysis pipeline (ICA, common average reference, etc.), you will corrupt your results. Drop these channels explicitly before any preprocessing.

### Mistake 7: Baseline correction on data with slow drift

**What happens:** You apply baseline correction (subtract the pre-stimulus mean from each epoch) expecting it to remove drift. But if there is slow drift within the epoch, subtracting a pre-stimulus window mean does nothing about drift after the stimulus. The baseline correction assumption is that the pre-stimulus period is flat and representative of the "resting" brain state — this is often false for imagined speech where the participant is preparing to imagine the word.

**Prevention:** Address slow drift with proper high-pass filtering (0.1 Hz or detrending) before epoching, not by relying solely on baseline correction.

### Mistake 8: Applying ICA to too little data

**What happens:** ICA needs enough data to reliably decompose independent sources. If you run ICA on individual subject epochs (only ~130 trials × 5 seconds = ~650 seconds), the decomposition may be unstable. Components won't cleanly separate eye and muscle artifacts.

**Prevention:** Run ICA on the continuous raw data (all conditions concatenated) before epoching. Use at least 1 minute of data per component you are estimating, more is better. With 64 channels, you need substantial data for a stable 64-component decomposition.

---

## Evaluation Mistakes

### Mistake 1: Reporting accuracy without the correct chance level

KaraOne has 11 classes (7 phonemic + 4 word prompts). The theoretical chance level for 11-class classification is 9.1% (1/11). For binary classification it is 50%. But reporting accuracy without context is meaningless.

**The complication:** For small datasets like KaraOne (~12 trials per class per subject after filtering), the empirical chance level from a permutation test is higher than the theoretical level. Research shows that for 5-class problems with small datasets, the empirical chance baseline is ~26% versus the theoretical 20%. Always run a permutation test (shuffle labels 1000 times, measure accuracy distribution) and compare your real accuracy to the 95th percentile of the shuffle distribution.

### Mistake 2: Comparing across papers that use different evaluation schemes

Published KaraOne results range from 57% to 85%. This range is mostly explained by:
- Different subset of classes (binary vs. 11-class)
- Different subjects included (some subjects are much easier to decode)
- Different evaluation schemes (within-subject, cross-subject, or wrongly-done segment-based)
- Different epochs windows

Comparison is almost impossible without controlling all of these. Do not interpret your own results against the paper literature without understanding exactly what evaluation protocol each paper used.

### Mistake 3: Subject-dependent evaluation only

**What happens:** You train and test on the same subject, using cross-validation. You achieve 75% accuracy. This tells you nothing about whether your model generalizes to new people. For any practical BCI application — including your OpenBCI hardware — you need cross-subject generalization.

**Prevention:** Always report both:
- Within-subject accuracy (train on subject A, test on subject A)
- Cross-subject accuracy (Leave-One-Subject-Out: train on 13 subjects, test on held-out subject)

The gap between these two numbers tells you how much your model is memorizing individual brain patterns versus learning generalizable speech features. A large gap (>15%) is a red flag.

### Mistake 4: Accuracy is the wrong metric if classes are unequal

For 11 classes with different numbers of trials per class, accuracy is misleading. A model that always predicts the majority class will achieve non-trivial accuracy. Use balanced accuracy or macro-averaged F1.

### Mistake 5: No statistical significance testing

A 2-percentage-point improvement over your baseline might be random noise with n=14 subjects. Do not claim "our method outperforms baseline" without statistical testing (paired t-test across subjects, or permutation test). EEG sample sizes are small — effects that look real often are not.

---

## Hardware Transition Pitfalls

### From 64-channel KaraOne to 8-channel OpenBCI

This transition is the hardest part of the entire project. Be prepared for near-complete invalidation of your KaraOne-trained models.

**The core problem: 8 channels is the research-documented failure zone.**

A 2025 study specifically on electrode reduction for speech imagery BCIs found:
- 64 → 32 channels: performance maintained for ~83% of subjects
- 64 → 8 channels: only ~10% of subjects show unchanged performance. The study authors explicitly stated "this layout cannot be recommended for practical applications."

The reason: optimal electrode positions for imagined speech are highly subject-specific. With 64 channels, you can discover each person's best subset. With only 8 channels in fixed positions, you are gambling that your 8 positions happen to cover each new user's relevant cortex. They usually do not.

### What specifically breaks:

**Spatial coverage loss:** Imagined speech involves distributed cortical networks (frontal language areas, motor cortex, temporal/auditory cortex). 8 electrodes cannot sample all of these simultaneously. You will be forced to choose a subset of regions and lose others entirely.

**No post-hoc channel selection:** With KaraOne, you can identify which of 64 channels were informative after the fact. With OpenBCI's 8 fixed channels, you are locked into whatever spatial sampling you chose before recording.

**Feature extraction methods fail:** Methods that rely on spatial patterns across many channels (Common Spatial Patterns, spatial filters, source localization) degrade severely with only 8 channels.

**ICA quality degrades:** With 8 channels, ICA can only recover 8 components, making reliable artifact rejection nearly impossible. You cannot cleanly separate eye, muscle, and brain components from 8 channels.

### What partially survives:

**Temporal features:** Time-domain features from single channels (band power, ERPs) degrade gracefully rather than catastrophically.

**Simple frequency band power:** Computing theta, alpha, beta power per channel works even at low channel counts. It is just less specific.

**Subject-specific calibration:** If you record a calibration session from each new user (even 10-20 trials), the 8-channel model can adapt. The model will not generalize zero-shot but can be fine-tuned per user.

### Mitigation strategy:

Before buying OpenBCI and testing, run this experiment on KaraOne: select only 8 channels that match OpenBCI's default electrode positions (Fp1, Fp2, C3, C4, P7, P8, O1, O2 or similar), retrain your model, and measure the accuracy drop. This will tell you exactly what you are getting into before investing in hardware.

---

## KaraOne-Specific Gotchas

### Gotcha 1: Trial count varies by subject

Most subjects have 132 trials. One subject has 131. Two subjects have more because the study was longer at the time of their recording. Do not assume a fixed number of trials per subject in your data loading code.

### Gotcha 2: Prompt order discrepancy between ID.txt and ID_p.txt

The intended prompt order (ID.txt) and the actual presented order (ID_p.txt) can differ because the presentation software sometimes repeated prompts. Always use ID_p.txt as your label source. If you use ID.txt, some of your labels are wrong.

### Gotcha 3: The EMG channel is not EMG

The channel labeled "EMG" in KaraOne contains color sensor data from the Kinect sensor, not electromyography. Do not attempt to use it for muscle artifact detection. Also drop M1, M2 (mastoid references), EKG, and Trigger before EEG analysis.

### Gotcha 4: Dataset has imagined AND vocalized conditions

KaraOne includes both imagined speech and overt/vocalized speech trials. These are interleaved. If you accidentally include vocalized speech trials in your imagined speech analysis (or vice versa), your results are meaningless. The epoch_inds.mat structure separates these — verify you are pulling the correct condition.

### Gotcha 5: Some trials were excluded due to Kinect malfunctions

The dataset description notes that some trials were removed due to Kinect sensor malfunctions or experimental problems. The actual usable trial count per subject may be lower than the nominal count. Your data loading code must handle variable-length trial arrays per subject without crashing.

### Gotcha 6: Imagined speech onset is unknown

Unlike overt speech (where audio onset is detectable), imagined speech has no external marker for when the participant actually began imagining. The epoch trigger marks when the prompt was shown, not when imagery began. There is inherent jitter of hundreds of milliseconds to seconds in when different participants start their imagery. This temporal uncertainty blurs any time-locked signal. Do not expect clean ERPs. Frequency-domain features (band power) are more robust to onset uncertainty than time-domain ERP features.

### Gotcha 7: 24GB dataset, MATLAB .mat format

The full dataset is 24GB. The .mat files are MATLAB format. You will need scipy.io.loadmat (or h5py for v7.3 .mat files) and must manually construct MNE RawArray or EpochsArray objects — MNE cannot directly read these .mat files. The data is not in FIF, EDF, or BrainVision format.

The KaraOne .mat files may be MATLAB v7.3 (HDF5-based). If scipy.io.loadmat fails, try:
```python
import h5py
with h5py.File('subject.mat', 'r') as f:
    data = f['data'][:]
```

### Gotcha 8: Extreme inter-subject variability

KaraOne's 14 subjects show massive variability in signal quality and decodability. Some subjects consistently yield 70%+ accuracy with basic methods. Others are near chance regardless of method. A model architecture that works on easy subjects will look impressive but may be learning nothing generalizable. Always report per-subject results, not just mean accuracy. A mean of 65% could mean "12 subjects at 70% and 2 at 30%" — which tells a very different story than the mean implies.

---

## Debugging EEG Data

If your preprocessing went wrong, these are the signs. Check each one before trusting any classification results.

### Visual checks (plot these before running any ML):

**1. Raw channel traces (time domain)**
- What to look for: all 60+ EEG channels should show similar amplitude (roughly 20-100 µV peak-to-peak). A channel that is flat (broken electrode), or 5-10x larger amplitude than others (bridged or high-impedance electrode), or all-zero is a bad channel. Mark and interpolate it before any processing.
- Red flag: more than 5-10 bad channels in a single subject. This subject's data may be unusable.

**2. Power spectral density (PSD) per channel**
- What to look for: you should see a characteristic 1/f dropoff (high power at low frequencies, decreasing at higher). Clear peaks at 50 Hz or 60 Hz indicate line noise (apply notch filter). A flat shelf from 20-200 Hz on temporal/peripheral channels indicates EMG contamination.
- Red flag: any channel showing dramatically more high-frequency power than others, especially T7, T8, FT7, FT8 — these sit over temporal muscles.
- Red flag: if your broadband power is 5-10x larger after ICA removal than before, your ICA went wrong and is adding variance instead of removing it.

**3. ICA component topographies**
- What to look for: eye blink components look like a frontal dipole, strongly weighted on Fp1/Fp2, with characteristic slow waveform in the time course. Muscle components show diffuse, patchy topography at peripheral electrodes with broadband spectral profile. If your ICA components all look like random noise with no interpretable topography, your ICA did not converge properly.
- Red flag: removing more than 10-15 components from 64-channel data is removing too much — you are probably removing brain signal.

**4. Grand average ERP per condition**
- What to look for: after epoching, average all trials of the same class and plot over time. You should see some condition-specific deflection somewhere in the 200-1000ms window. If all conditions look identical, either your labels are wrong or the preprocessing destroyed the signal.
- Red flag: a grand average that looks exactly like zero or like random noise with no structure means your data pipeline is broken, not that "this subject is hard."

**5. Epoch rejection rate**
- What to look for: after applying an amplitude threshold (e.g., ±100µV), a reasonable rejection rate is 10-20% of trials. If you are rejecting 50%+ of trials, your data has major artifact problems. If you reject 0%, your threshold is probably too high and you are including artifact-contaminated trials.

**6. Sanity check: permutation accuracy**
- Before trusting any result above chance, run your full pipeline with shuffled labels. Expected result: accuracy near the theoretical chance level (9.1% for 11 classes, 50% for binary). If shuffled labels give 60% accuracy, you have data leakage — the model is learning something other than the labels.

---

## Common "Fake Good Results" Patterns

These patterns produce numbers that look real and generalize to absolutely nothing.

### Pattern 1: 99%+ accuracy

Almost never legitimate in EEG imagined speech. If you see this, you have data leakage (almost certainly Trap 1 — segment splits without subject grouping). Real SOTA for 11-class KaraOne is 57-85%. Binary classification tops out around 85-90%. Anything above that is an artifact of methodology.

### Pattern 2: Subject-dependent model dramatically outperforms cross-subject

A within-subject model at 80% that drops to 55% cross-subject is not "80% accuracy" — it is a model that has memorized 14 individual brain fingerprints. It is useless for any new user (including your future OpenBCI self). If the cross-subject number is near chance, your model has learned nothing about speech.

### Pattern 3: High accuracy only on easy subjects, never reported per-subject

Papers that report mean accuracy without per-subject breakdown often have 2-3 subjects at 95% (possibly due to EMG contamination or data leakage at the subject level) pulling up a mean that hides 10 subjects at near-chance.

### Pattern 4: Accuracy improves linearly with model complexity on small data

On KaraOne's ~130 trials per subject, a deep learning model with millions of parameters will overfit. If adding more parameters keeps improving validation accuracy, your validation set is contaminated (leaking). On truly held-out data, deep models often perform worse than a linear SVM on this scale of data. If your 12-layer transformer at 80% beats a logistic regression at 79%, the transformer has not actually learned anything more meaningful — it has just found a more complex way to overfit.

### Pattern 5: Results don't degrade when you remove channels

If removing 30 of 64 channels doesn't change your accuracy at all, your model may be ignoring EEG content entirely and learning some other confound (trial index, session order, subject identity). Real speech-relevant neural patterns are spatially distributed — removing channels should degrade performance meaningfully.

### Pattern 6: High-gamma band outperforms all other frequency bands consistently

As described above, this is the EMG contamination signature. Real imagined speech decoding typically draws from theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) bands, with gamma providing modest additional benefit when clean. If gamma alone dominates, you are decoding muscle artifacts.

### Pattern 7: Model trained on overt speech generalizes perfectly to imagined speech

These are fundamentally different neural tasks. A model that achieves 80% on overt speech and transfers without degradation to imagined speech is almost certainly using overt speech motor artifacts (which are enormous) as the signal source. Imagined speech has no mouth movement — there should be degradation when transferring.

---

## Prevention Strategies

| Pitfall | Phase to Address | Prevention Method |
|---------|------------------|-------------------|
| Segment-level data leakage | From day one, before any ML | Use GroupKFold with subject_id as group variable. Make it the default, never deviate. |
| Normalization leakage | From day one, pipeline setup | Wrap scaler.fit() inside cross-validation loop, never outside |
| Feature selection leakage | Feature engineering phase | Use sklearn Pipeline objects to guarantee fit happens inside CV |
| ICA on full dataset | Preprocessing phase | Run ICA on continuous raw data only; document this decision |
| Not re-referencing correctly | Data loading phase | Add average re-reference as first step in raw data processing |
| Bad KaraOne channels included | Data loading phase | Explicitly drop EMG, M1, M2, EKG, Trigger channels before all other steps |
| Wrong label file (ID.txt vs ID_p.txt) | Data loading phase | Hardcode use of ID_p.txt; add assertion to verify label count matches trial count |
| Imagined vs. vocalized trial confusion | Data loading phase | Load condition flags explicitly; assert imagined speech count matches expected |
| High-gamma EMG artifact | Feature engineering phase | Default to 0-50 Hz bandpass; if using gamma, verify PSD at temporal channels |
| No permutation test | Evaluation phase | Run 1000-shuffle permutation test alongside every result; report p-value |
| Subject-dependent results only | Evaluation phase | Always report LOSO cross-subject accuracy alongside within-subject accuracy |
| Wrong chance level reported | Evaluation phase | For 11 classes: theoretical=9.1%, but run empirical permutation test to get actual baseline |
| No per-subject results | Reporting phase | Always show the per-subject breakdown, not just mean ± std |
| 8-channel direct model transfer | Hardware transition phase | Simulate 8-channel by selecting only those positions from 64-ch KaraOne data first |
| Skipping model complexity baseline | ML phase | Always run a linear baseline (LDA or logistic regression) before any deep learning |

---

## Sources

- [Data leakage in deep learning studies of translational EEG (Frontiers/PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/) — HIGH confidence, peer-reviewed. Key finding: segment-based leakage inflated accuracy from 53% to 99.8%.
- [k-Fold Cross-Validation Can Significantly Over-Estimate True Classification Accuracy in Common EEG-Based Passive BCI Experimental Designs (Sensors, 2023)](https://www.mdpi.com/1424-8220/23/13/6077) — HIGH confidence. Up to 25 percentage point inflation from temporal autocorrelation.
- [Decoding Covert Speech From EEG — A Comprehensive Review (Frontiers/PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8116487/) — HIGH confidence, comprehensive review of methodology, artifacts, and pitfalls.
- [A State-of-the-Art Review of EEG-Based Imagined Speech Decoding (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9086783/) — HIGH confidence. Covers mental repetition bias, offline vs. online gaps, vocabulary limits.
- [How low can you go: evaluating electrode reduction methods for EEG-based speech imagery BCIs (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12263900/) — HIGH confidence. 8 channels works for only ~10% of subjects.
- [High-frequency brain activity and muscle artifacts in MEG/EEG: A review and recommendations (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3625857/) — HIGH confidence. Defines EMG spectral characteristics, frequency ranges, topographic identification.
- [How EEG preprocessing shapes decoding performance (Nature Communications Biology, 2025)](https://www.nature.com/articles/s42003-025-08464-3) — HIGH confidence. Systematic study of preprocessing order effects on decoding.
- [The KARA ONE database (official)](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) — HIGH confidence (primary source). Dataset quirks: trial counts, channel issues, prompt order files.
- [Systematic Review of EEG-Based Imagined Speech Classification Methods (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679664/) — HIGH confidence. GroupKFold, LOSO requirements documented.
- [Inflated prediction accuracy of neuropsychiatric biomarkers caused by data leakage in feature selection (Scientific Reports)](https://www.nature.com/articles/s41598-021-87157-3) — HIGH confidence. Feature selection leakage mechanism.
