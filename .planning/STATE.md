# Project State — EEG Imagined Speech Pipeline

**Current Phase:** 1
**Status:** planning

---

## Phase Progress

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data Access & Exploration | current |
| 2 | Preprocessing Pipeline | pending |
| 3 | Baseline Classifier + Evaluation | pending |
| 4 | Deep Learning Models | pending |
| 5 | Audio Engineering Experiment | pending |

---

## Current Focus

**Phase 1 goal:** One subject's raw .mat file is loaded into MNE with correct labels, imagined-speech-only trials isolated, and EEG signal quality visually confirmed.

**Next action:** Set up Python environment with `uv` and run `download-karaone.py`.

---

## Decisions Log

| Decision | Rationale | Phase |
|----------|-----------|-------|
| EVAL infrastructure placed in Phase 3 (not 4) | Logging must exist before DL runs, not after; all subsequent phases inherit it | Planning |
| AUDIO placed in Phase 5 (after baseline, before DL results are final) | Needs working SVM classifier as benchmark; self-contained enough to run without DL | Planning |
| DWT features placed in Phase 4 alongside DL | DWT is an alternative to band-power, most useful when comparing feature strategies against DL | Planning |

---

## Accumulated Context

*(Filled during execution — blockers, surprises, open questions resolved)*

---

## Performance Metrics

| Phase | Target | Actual |
|-------|--------|--------|
| 3 | Binary subproblem (vowel/consonant) ≥65% | — |
| 3 | Permutation test p < 0.05 | — |
| 4 | At least one model >57% on at least one subject | — |
