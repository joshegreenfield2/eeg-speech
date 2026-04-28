# Stack Research — EEG Imagined Speech

**Researched:** 2026-04-27
**Dataset:** KaraOne (64-ch, 14 subjects, 1 kHz, .mat format, MATLAB 7.3 HDF5)
**Starting repo:** AshrithSagar/EEG-Imagined-speech-recognition
**Package manager:** uv

---

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

---

## Signal Processing Layer

### MNE as the Signal Processing Core

MNE is your DAW equivalent for EEG. Think of it this way:

- **Raw object** = the raw audio file (continuous multichannel signal at 1 kHz)
- **Epochs object** = the sliced regions (like clip regions in a DAW)
- **Evoked object** = the averaged response (like a static mix render)
- **ICA** = a blind source separation tool (like iZotope RX unmixing artifacts)

```python
import mne

# Load (after converting .mat to MNE format — see Data Handling)
raw = mne.io.RawArray(data, info)

# Bandpass filter: 1-40 Hz is the typical imagined speech band
# EEG useful signal lives in delta (0.5-4), theta (4-8), alpha (8-13),
# beta (13-30), gamma (30-100) — imagined speech evidence spans theta-gamma
raw.filter(l_freq=1.0, h_freq=40.0, method='fir', fir_window='hamming')

# Notch filter: remove 60 Hz line noise (North America) + harmonics
raw.notch_filter(freqs=[60, 120], method='fir')

# Re-reference to average
raw.set_eeg_reference('average', projection=False)

# Epoch around stimulus events
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.5,
                    baseline=(-0.2, 0), preload=True)
```

### SciPy for Lower-Level Control

Use `scipy.signal` when you need to go below MNE's abstraction — e.g., computing bandpower manually for feature extraction:

```python
from scipy.signal import welch, butter, filtfilt
import numpy as np

# Welch PSD — equivalent of running an FFT with overlap/window averaging
# Think of it like the spectrum analyzer in your DAW, but time-averaged
freqs, psd = welch(epoch_data, fs=1000, nperseg=256, noverlap=128)

# Band-specific power (alpha: 8-13 Hz)
alpha_mask = (freqs >= 8) & (freqs <= 13)
alpha_power = np.trapz(psd[:, alpha_mask], freqs[alpha_mask], axis=-1)
```

### Frequency Band Reference for EEG (audio production framing)

| Band | Range | Analog | Relevance for Imagined Speech |
|------|-------|--------|-------------------------------|
| Delta | 0.5–4 Hz | Sub-bass | Mostly artifacts/drift at scalp; filter out |
| Theta | 4–8 Hz | Bass | Language processing, working memory — HIGH relevance |
| Alpha | 8–13 Hz | Low-mid | Idling/suppression; context for imagery |
| Beta | 13–30 Hz | Mid | Motor planning, active cognition — HIGH relevance |
| Gamma | 30–100 Hz | High-mid | Fine temporal binding — MEDIUM relevance (hard to record cleanly) |

KaraOne at 1 kHz captures all bands. Typical imagined speech pipeline filters 1–40 Hz (drops gamma, which is mostly noise at scalp-level EEG anyway).

---

## ML/DL Layer

### Baseline: CSP + SVM (scikit-learn)

Build this first. It's interpretable, fast to train, and every paper benchmarks against it. If your deep model can't beat CSP+SVM, something is wrong with the deep model.

```python
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipeline = Pipeline([
    ('csp', CSP(n_components=4, log=True, norm_trace=False)),
    ('clf', SVC(kernel='rbf', C=1.0))
])

# CRITICAL: use GroupKFold with subject IDs to avoid data leakage
# (see Pitfalls section)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_epochs, y_labels, cv=cv)
```

### Deep Learning: PyTorch + braindecode

**EEGNet** (Lawhern et al., 2018) is the canonical compact EEG classifier. braindecode provides a clean PyTorch implementation.

```python
from braindecode.models import EEGNetv4
import torch

model = EEGNetv4(
    n_chans=64,          # KaraOne has 64 channels
    n_outputs=11,        # KaraOne has 11 imagined speech classes
    n_times=1500,        # 1.5s trial at 1kHz = 1500 samples
    final_conv_length='auto'
)
```

**CNN+RNN hybrid** (what the roadmap is targeting): Build this yourself in PyTorch after EEGNet baseline. The pattern from 2024-2025 literature:

```python
# Conceptual structure — implement in Phase 2+
class EEGCNNLSTMClassifier(torch.nn.Module):
    def __init__(self, n_chans, n_classes, n_times):
        # Temporal conv block (learn frequency-like filters — like EQ stages)
        # Depthwise spatial conv (learn channel weights — like a spatial mixer)
        # LSTM (capture sequential dependencies across the trial)
        # Classifier head
        pass
```

The 2025 consensus from papers: CNN-BiLSTM outperforms CNN-LSTM for imagined speech (bidirectional LSTM sees both past and future context in the trial window), hitting ~77.8% on 11-class word classification.

**torcheeg** (v1.1.3, Dec 2024) is an alternative to braindecode. It has more model variety (DGCNN, ATCNet, EEG-Conformer) but is less actively maintained and has weaker documentation. Use braindecode for EEGNet; cherry-pick torcheeg models if you need graph neural networks later.

---

## Visualization

All visualization lives inside MNE's matplotlib integration. You don't need a separate viz library for EEG-specific plots.

| Use Case | MNE Method | Notes |
|----------|-----------|-------|
| Topomap (scalp heatmap) | `mne.viz.plot_topomap()` or `evoked.plot_topomap()` | The "spectral analysis view" of EEG — shows voltage distribution across the scalp. Essential for sanity-checking which channels carry your signal. |
| Time-frequency (spectrogram) | `epochs.compute_tfr('morlet', freqs=..., n_cycles=...)` | Morlet wavelet is the EEG equivalent of a constant-Q spectrogram. Prefer this over raw FFT for EEG. |
| PSD across channels | `raw.compute_psd().plot_topomap()` | Shows where power lives on the scalp per band |
| Raw traces | `raw.plot()` | Interactive browser — use to spot bad channels/artifacts before ICA |
| ICA components | `ica.plot_components()` | Before applying mne-icalabel, visually inspect the top N components |
| Epochs image | `epochs.plot_image(picks=['Cz'])` | Heatmap of single-channel response per trial — great for spotting consistent imagined speech ERPs |

For notebook work, use `%matplotlib inline` or `%matplotlib widget`. MNE's interactive `raw.plot()` needs a backend — use `mne.viz.set_browser_backend('qt')` for the full interactive browser in JupyterLab.

**Plotly is not recommended** for EEG-specific plots. It lacks topomap support and electrode layout awareness. Use it only for results tables and accuracy/loss curves if you prefer interactive output.

---

## Data Handling

### KaraOne File Format

KaraOne distributes data as MATLAB .mat files. The format version matters critically:

```
KaraOne .mat files → MATLAB 7.3 (HDF5-based)
scipy.io.loadmat   → ONLY works on MATLAB < 7.3
mat73.loadmat      → works on MATLAB 7.3+ HDF5 format ← USE THIS
```

```python
import mat73
import numpy as np
import mne

# Load the raw KaraOne .mat file
data_dict = mat73.loadmat('MM05.mat')

# KaraOne structure: data['eeg']['data'] is channels × samples
eeg_data = np.array(data_dict['eeg']['data'])  # shape: (64, n_samples)
sfreq = float(data_dict['eeg']['srate'])        # 1000 Hz

# Build MNE info object
ch_names = [str(ch) for ch in data_dict['eeg']['chanlocs']['labels']]
ch_types = ['eeg'] * len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create Raw object
raw = mne.io.RawArray(eeg_data, info)
```

### Format Reference Table

| Format | Extension | How to Load | Notes |
|--------|-----------|------------|-------|
| MATLAB < 7.3 | `.mat` | `scipy.io.loadmat('file.mat')` | Returns nested dict of arrays |
| MATLAB 7.3 | `.mat` | `mat73.loadmat('file.mat')` | HDF5 under the hood; KaraOne uses this |
| EEGLAB | `.set` + `.fdt` | `mne.io.read_raw_eeglab('file.set')` | MNE reads .set directly; .fdt is the binary data blob |
| MNE native | `.fif` | `mne.io.read_raw_fif('file.fif')` | Save preprocessed data here for fast reload |
| EDF/BDF | `.edf`, `.bdf` | `mne.io.read_raw_edf('file.edf')` | Hospital/clinic format; not KaraOne |
| HDF5 generic | `.h5`, `.hdf5` | `h5py.File('file.h5', 'r')` | Direct access when mat73 structs are awkward |

**Workflow**: Load KaraOne .mat via mat73 → build MNE Raw → preprocess → save epochs as `.fif` for fast iteration. The `.fif` reload is ~10x faster than re-running the full preprocessing chain.

```python
# Save preprocessed epochs — do this once, load fast thereafter
epochs.save('subject_MM05-epo.fif', overwrite=True)

# Fast reload
epochs = mne.read_epochs('subject_MM05-epo.fif', preload=True)
```

---

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

---

## uv Setup

```toml
# pyproject.toml (uv manages this)
[project]
name = "eeg-speech"
requires-python = ">=3.11"  # braindecode requires 3.11+; locks you to 3.11-3.13

[project.dependencies]
mne = ">=1.8"
numpy = ">=1.26"
scipy = ">=1.13"
mat73 = ">=0.60"
h5py = ">=3.10"
autoreject = ">=0.4.3"
mne-icalabel = ">=0.8.1"
pyprep = ">=0.6.0"
scikit-learn = ">=1.3"
torch = ">=2.1"          # Install CPU or CUDA variant separately via uv
braindecode = ">=1.4.0"
matplotlib = ">=3.8"
pandas = ">=2.0"
joblib = ">=1.3"

[project.optional-dependencies]
dev = [
    "jupyter",
    "ipykernel",
    "ipympl",           # %matplotlib widget for interactive MNE plots in notebooks
]
```

```bash
# Install (uv handles the venv automatically)
uv sync

# PyTorch: install separately to pick CPU vs CUDA build
# CPU-only (for local dev on M-series Mac):
uv pip install torch torchvision torchaudio

# CUDA (if you move to a GPU machine for training):
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

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

---

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
