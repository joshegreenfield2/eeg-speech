<!-- GSD:project-start source:PROJECT.md -->
## Project

**EEG Imagined Speech Decoding**

A personal brain-computer interface research project that classifies imagined words from non-invasive EEG data. Inspired by the MIT AlterEgo concept — silent communication from thought — built by a solo developer learning hands-on. Not a product; a learning-and-experimentation platform with a working ML pipeline as the goal.

**Core Value:** A working pipeline that classifies imagined words from EEG signals above chance accuracy — proving the concept is viable before buying personal hardware.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
| Library | pip package | Version | Role | Why This One |
|---------|-------------|---------|------|--------------|
| MNE-Python | `mne` | `>=1.8, <2` (stable: 1.12.1) | Core EEG I/O, filtering, epoching, ICA, topomap | The de-facto standard for EEG/MEG Python work. Handles everything from raw loading to ICA to visualization. Nothing else comes close for EEG-specific ops. |
| NumPy | `numpy` | `>=1.26` (MNE hard requirement) | Array ops, the carrier format for all EEG tensors | Required by MNE; also the bridge between MNE epochs and PyTorch tensors |
| SciPy | `scipy` | `>=1.13` (MNE hard requirement) | Signal processing: Welch PSD, butter filters, loadmat | `scipy.io.loadmat` for older .mat; `scipy.signal` for manual bandpass/PSD when you need lower-level control than MNE exposes |
| mat73 | `mat73` | `>=0.60` | Load MATLAB 7.3 HDF5 .mat files | KaraOne is MATLAB 7.3 format — `scipy.io.loadmat` silently fails on v7.3. mat73 wraps h5py cleanly into a dict. |
| h5py | `h5py` | `>=3.10` | Direct HDF5 access, backup to mat73 | Needed when mat73's dict interface isn't enough; also future-proofs for any HDF5 output pipelines |
| autoreject | `autoreject` | `0.4.3` | Automated epoch rejection, RANSAC bad channel detection | Learns rejection thresholds per-channel via cross-validation. Standard preprocessing step in 2024-2025 imagined speech papers (e.g., Chisco dataset pipeline). |
| mne-icalabel | `mne-icalabel` | `0.8.1` | Automatic ICA component classification (ICLabel model) | Ports the MATLAB EEGLab ICLabel classifier to Python. Labels components as brain/muscle/eye/heart/line noise automatically — removes the manual "look at every ICA component" step. |
| pyprep | `pyprep` | `0.6.0` | PREP pipeline: robust average reference + bad channel detection | Standard pre-ICA step; detects bad channels before ICA (autoreject handles post-ICA epoch rejection). Requires Python >=3.10. |
| scikit-learn | `scikit-learn` | `>=1.3` | CSP, SVM, LDA baseline classifiers, cross-validation, pipelines | The baseline classifier layer. CSP (Common Spatial Patterns) + SVM is the standard pre-deep-learning EEG classification baseline that every paper benchmarks against. |
| PyTorch | `torch` | `>=2.1` (CPU or CUDA) | CNN + RNN deep learning models | Braindecode, torcheeg, and the majority of current EEG DL research is PyTorch-native. TensorFlow/Keras exists in this space but PyTorch dominates new research code. |
| braindecode | `braindecode` | `1.4.0` | EEGNet, ShallowFBCSP, Deep4Net — pre-built EEG DL architectures | Provides battle-tested EEG-specific architectures (EEGNet especially) as PyTorch modules. EEGNet is the most cited compact BCI architecture. Requires Python >=3.11. |
| matplotlib | `matplotlib` | `>=3.8` (MNE hard requirement) | 2D plots, EEG traces, PSD plots | MNE's visualization backend. All MNE plot functions return matplotlib figures. |
| pandas | `pandas` | `>=2.0` | Trial metadata, subject tables, results aggregation | Managing epoch labels, subject info, and cross-validation result tables |
| joblib | `joblib` | `>=1.3` | Parallel cross-validation, caching fitted transformers | scikit-learn uses it internally; explicit use for caching expensive preprocessing (ICA fits) across runs |
## Signal Processing Layer
### MNE as the Signal Processing Core
- **Raw object** = the raw audio file (continuous multichannel signal at 1 kHz)
- **Epochs object** = the sliced regions (like clip regions in a DAW)
- **Evoked object** = the averaged response (like a static mix render)
- **ICA** = a blind source separation tool (like iZotope RX unmixing artifacts)
# Load (after converting .mat to MNE format — see Data Handling)
# Bandpass filter: 1-40 Hz is the typical imagined speech band
# EEG useful signal lives in delta (0.5-4), theta (4-8), alpha (8-13),
# beta (13-30), gamma (30-100) — imagined speech evidence spans theta-gamma
# Notch filter: remove 60 Hz line noise (North America) + harmonics
# Re-reference to average
# Epoch around stimulus events
### SciPy for Lower-Level Control
# Welch PSD — equivalent of running an FFT with overlap/window averaging
# Think of it like the spectrum analyzer in your DAW, but time-averaged
# Band-specific power (alpha: 8-13 Hz)
### Frequency Band Reference for EEG (audio production framing)
| Band | Range | Analog | Relevance for Imagined Speech |
|------|-------|--------|-------------------------------|
| Delta | 0.5–4 Hz | Sub-bass | Mostly artifacts/drift at scalp; filter out |
| Theta | 4–8 Hz | Bass | Language processing, working memory — HIGH relevance |
| Alpha | 8–13 Hz | Low-mid | Idling/suppression; context for imagery |
| Beta | 13–30 Hz | Mid | Motor planning, active cognition — HIGH relevance |
| Gamma | 30–100 Hz | High-mid | Fine temporal binding — MEDIUM relevance (hard to record cleanly) |
## ML/DL Layer
### Baseline: CSP + SVM (scikit-learn)
# CRITICAL: use GroupKFold with subject IDs to avoid data leakage
# (see Pitfalls section)
### Deep Learning: PyTorch + braindecode
# Conceptual structure — implement in Phase 2+
## Visualization
| Use Case | MNE Method | Notes |
|----------|-----------|-------|
| Topomap (scalp heatmap) | `mne.viz.plot_topomap()` or `evoked.plot_topomap()` | The "spectral analysis view" of EEG — shows voltage distribution across the scalp. Essential for sanity-checking which channels carry your signal. |
| Time-frequency (spectrogram) | `epochs.compute_tfr('morlet', freqs=..., n_cycles=...)` | Morlet wavelet is the EEG equivalent of a constant-Q spectrogram. Prefer this over raw FFT for EEG. |
| PSD across channels | `raw.compute_psd().plot_topomap()` | Shows where power lives on the scalp per band |
| Raw traces | `raw.plot()` | Interactive browser — use to spot bad channels/artifacts before ICA |
| ICA components | `ica.plot_components()` | Before applying mne-icalabel, visually inspect the top N components |
| Epochs image | `epochs.plot_image(picks=['Cz'])` | Heatmap of single-channel response per trial — great for spotting consistent imagined speech ERPs |
## Data Handling
### KaraOne File Format
# Load the raw KaraOne .mat file
# KaraOne structure: data['eeg']['data'] is channels × samples
# Build MNE info object
# Create Raw object
### Format Reference Table
| Format | Extension | How to Load | Notes |
|--------|-----------|------------|-------|
| MATLAB < 7.3 | `.mat` | `scipy.io.loadmat('file.mat')` | Returns nested dict of arrays |
| MATLAB 7.3 | `.mat` | `mat73.loadmat('file.mat')` | HDF5 under the hood; KaraOne uses this |
| EEGLAB | `.set` + `.fdt` | `mne.io.read_raw_eeglab('file.set')` | MNE reads .set directly; .fdt is the binary data blob |
| MNE native | `.fif` | `mne.io.read_raw_fif('file.fif')` | Save preprocessed data here for fast reload |
| EDF/BDF | `.edf`, `.bdf` | `mne.io.read_raw_edf('file.edf')` | Hospital/clinic format; not KaraOne |
| HDF5 generic | `.h5`, `.hdf5` | `h5py.File('file.h5', 'r')` | Direct access when mat73 structs are awkward |
# Save preprocessed epochs — do this once, load fast thereafter
# Fast reload
## What NOT to Use
| Library / Approach | Why to Avoid |
|-------------------|-------------|
| **TensorFlow / Keras** | Both frameworks work, but PyTorch has won EEG research. braindecode (PyTorch-native) is better maintained than any Keras EEG library. Mixing frameworks creates dependency hell. Pick one: pick PyTorch. |
| **EEGLAB (MATLAB)** | You're in Python. EEGLAB's Python bindings are partial and poorly maintained. MNE replicates all the standard EEGLAB operations natively. |
| **MNE-BIDS for KaraOne** | mne-bids is for datasets already in BIDS format. KaraOne is raw MATLAB. Don't add this complexity until/unless you're converting to BIDS for publication. |
| **PyEEG** | Last meaningful commit was 2013. `antropy` (entropy features) and `scipy` cover anything PyEEG offered, with active maintenance. |
| **eeglib** | Small, unmaintained (2021 last activity). Use MNE + scipy feature extraction instead. |
| **scipy.io.loadmat for KaraOne** | Will silently return incomplete data or error on MATLAB 7.3 files. Always check with `mat73.loadmat` first. |
| **Random k-fold CV without subject grouping** | Not a library, but a critical methodology mistake. Random splits across trials from the same subject leak subject identity into the test set — reported accuracy inflates by 15–30%. Use `GroupKFold(groups=subject_ids)` or Leave-One-Subject-Out (LOSO). See the medRxiv data leakage paper. |
| **Applying ICA before autoreject** | ICA is sensitive to bad epochs. Run autoreject first to remove grossly bad epochs, THEN fit ICA on the clean data, THEN run autoreject again post-ICA. Order matters. |
| **Raw Welch/FFT features directly into SVM** | Works but treats all frequencies equally. CSP learns the optimal spatial filters for your specific task. Try CSP features before falling back to manual bandpower features. |
| **BrainFlow** | Real-time acquisition SDK. Useful if you ever connect live hardware, but adds no value for offline dataset analysis (KaraOne). |
## uv Setup
# pyproject.toml (uv manages this)
# Install (uv handles the venv automatically)
# PyTorch: install separately to pick CPU vs CUDA build
# CPU-only (for local dev on M-series Mac):
# CUDA (if you move to a GPU machine for training):
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
## Confidence Notes
| Claim | Confidence | Notes |
|-------|-----------|-------|
| MNE 1.12.1 is current stable | HIGH | Verified directly on PyPI (released 2026-04-20) |
| MNE requires numpy >=1.26, scipy >=1.13 | HIGH | Verified on PyPI |
| braindecode 1.4.0 requires Python >=3.11 | HIGH | Verified on PyPI (released 2026-04-04) |
| mat73 is required for KaraOne .mat files | HIGH | KaraOne is documented as MATLAB 7.3 format; scipy.io.loadmat limitation is documented in SciPy official docs |
| mne-icalabel 0.8.1 is current | MEDIUM | Found via WebSearch referencing PyPI; PyPI page itself had a render error during fetch |
| autoreject 0.4.3 is current | HIGH | Verified on PyPI |
| pyprep 0.6.0 is current | HIGH | Verified on PyPI |
| torcheeg 1.1.3 is current | HIGH | Verified on PyPI (released Dec 2024) |
| CNN-BiLSTM ~77.8% on KaraOne 11-class | MEDIUM | From a 2024 PMC paper (PMC11595501); single study, specific electrode subset |
| PyTorch recommended over TensorFlow for EEG research | MEDIUM | Based on ecosystem analysis — braindecode is PyTorch-only, most recent papers use PyTorch; not a hard community consensus statement |
| GroupKFold / LOSO required to avoid data leakage | HIGH | Multiple 2024-2025 papers explicitly document this as the field's primary reproducibility problem; medRxiv preprint specifically on EEG data leakage |
| CSP+SVM is standard baseline | HIGH | Consistent across systematic reviews and recent papers |
| EEG field moves fast | HIGH | Number of imagined speech papers went from ~3/year to ~35/year recently; new architectures (EEG-Conformer, BENDR) appearing; treat architecture choices as 12-month recommendations |
## Sources
- MNE-Python PyPI: https://pypi.org/project/mne/
- braindecode PyPI: https://pypi.org/project/braindecode/
- autoreject PyPI: https://pypi.org/project/autoreject/
- pyprep PyPI: https://pypi.org/project/pyprep/
- torcheeg PyPI: https://pypi.org/project/torcheeg/
- mat73 PyPI: https://pypi.org/project/mat73/
- Data leakage in EEG deep learning (medRxiv 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/
- Hybrid 3DCNN-BiLSTM imagined speech (PMC 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11595501/
- Chisco dataset preprocessing pipeline: https://www.nature.com/articles/s41597-024-04114-1
- KaraOne database: http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html
- AshrithSagar starting repo: https://github.com/AshrithSagar/EEG-Imagined-speech-recognition
- MNE filtering background: https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html
- SciPy loadmat docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
- Systematic review of EEG imagined speech decoding (MDPI 2024): https://www.mdpi.com/1424-8220/24/24/8168
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
