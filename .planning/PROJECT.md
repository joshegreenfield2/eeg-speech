# EEG Imagined Speech Decoding

## What This Is

A personal brain-computer interface research project that classifies imagined words from non-invasive EEG data. Inspired by the MIT AlterEgo concept — silent communication from thought — built by a solo developer learning hands-on. Not a product; a learning-and-experimentation platform with a working ML pipeline as the goal.

**Core Value:** A working pipeline that classifies imagined words from EEG signals above chance accuracy — proving the concept is viable before buying personal hardware.

## Context

- **Phase:** Pre-hardware validation — using publicly available labeled EEG datasets first
- **Hardware planned:** OpenBCI Cyton 8-channel + Headband Kit (~$400-500), acquired after pipeline is validated
- **Solo developer:** Self-taught, technical, audio production background — thinks about signals in audio engineering terms

## Core Hypothesis

Even though raw EEG has brutal SNR (microvolt brain signals buried in muscle, electrical, and neural noise), a personalized model trained on enough labeled examples of a specific person thinking specific words should be able to classify them above chance. No need to understand what "the brainwave for hello" looks like — consistent labeled examples + deep learning should find the pattern. SOTA: 57-85% accuracy on similar tasks.

## Primary Dataset

**KaraOne** — Gold standard for imagined speech research
- 64 channels, 14 subjects, 1kHz sampling rate
- Imagined AND vocalized phonemes/words: "pat," "pot," "knew," "gnaw," plus phonemes /iy/, /piy/, /tiy/, /m/, /n/
- Source: http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html
- Download automation: https://github.com/AshrithSagar/EEG-Imagined-speech-recognition

## Starting Point Repository

AshrithSagar/EEG-Imagined-speech-recognition — contains:
- `download-karaone.py` — automated dataset download script
- Preprocessing scripts (MNE-based)
- Baseline classifier scripts

## Signal Processing Concepts Established

- EEG measures voltage over time across multiple scalp electrodes
- Frequency bands: Delta (0-4Hz), Theta (4-8), Alpha (8-12), Beta (12-30), Gamma (30+)
- Both temporal (time) AND spatial (electrode location) info matter
- ICA (Independent Component Analysis) = standard noise separation
- Common artifacts: 60Hz line noise, eye blinks, muscle, jaw tension
- Modern SOTA: hybrid CNN+RNN (3D CNN spatial + RNN temporal)
- Wavelet transforms + DNNs hit ~57% on KaraOne (11-class)
- One paper hit 85% on 9-word classification with CNN+RNN

## Experimental Side Quest

Convert a single EEG channel to WAV (resample into audible range) and run through iZotope RX spectral denoise. Compare to MNE's built-in ICA noise reduction. The relative patterns between words should survive resampling — audio engineering perspective on signal denoising.

## Secondary Datasets (Future)

- "Thinking Out Loud" inner speech dataset (10 participants, 4 mental tasks, 3 conditions)
- FEIS (Fourteen-channel Emotiv, 21 participants, 16 phonemes) — https://zenodo.org/record/3369178
- Chisco (2024 Nature Scientific Data, imagined speech)

## Tech Stack (Planned)

- **Python** package management via `uv`
- **MNE-Python** — EEG signal processing (equivalent to iZotope RX for brainwaves)
- **scikit-learn** — baseline classifiers
- **PyTorch** — deep learning models
- **NumPy / SciPy** — signal processing utilities

## Requirements

### Validated

(None yet — prove the pipeline works first)

### Active

- [ ] Download and store KaraOne dataset locally
- [ ] Load and inspect raw EEG data for one subject
- [ ] Apply standard preprocessing pipeline (filtering, ICA artifact removal)
- [ ] Extract features from imagined speech trials
- [ ] Train and evaluate a baseline classifier on imagined vs. vocalized conditions
- [ ] Implement deep learning model (CNN+RNN) for word classification
- [ ] Achieve above-chance accuracy on held-out test set
- [ ] Compare MNE ICA vs. iZotope RX spectral denoise (experimental)
- [ ] Extend to multiple subjects / cross-subject generalization
- [ ] Build recording pipeline for personal OpenBCI hardware

### Out of Scope

- Real-time BCI inference — research/offline only for now
- Product/UI layer — this is a scripts + notebooks workflow
- Personal hardware recording — after pipeline is validated on public data

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start with KaraOne before buying hardware | Validate pipeline, learn signal processing on real data | — Pending |
| Use AshrithSagar repo as scaffold | Has download script + preprocessing patterns already | — Pending |
| uv for Python package management | Recommended by upstream repo, fast dependency resolution | — Pending |
| MNE-Python for EEG | Industry standard, equivalent to iZotope RX for brainwaves | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-27 after initialization*
