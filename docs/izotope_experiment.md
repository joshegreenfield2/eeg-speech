# iZotope RX denoising experiment — SOP

Run this when you want to test whether RX's audio-domain denoising catches EEG artifacts that MNE-ICA didn't.

## Inputs

`outputs/audio/MM05_<CH>_60s_raw.wav` — three channels × 60s clips (MM05).
- **FP1** (frontal pole) — eye blinks → audible clicks
- **T7** (temporal) — jaw/muscle EMG → audible hiss
- **OZ** (occipital) — alpha bursts + visual noise → tonal hum

EEG was pitch-shifted ×44 (1 Hz EEG → 44 Hz audio) so RX's algorithms can engage with it. After cleaning, the comparison script un-pitches the frequency axis to report findings in true EEG Hz.

## Step 1 — listen first (5 min)

Open all three 60s WAVs in RX. Just play them. Note what you hear.
- Expected on FP1: scattered clicks (blinks), low rumble (alpha leak)
- Expected on T7: continuous noise floor, occasional bursts (jaw clenches)
- Expected on OZ: rhythmic mid-range tone (alpha)

If something else stands out, that's the artifact RX should target.

## Step 2 — module per channel (~30 min)

One module each, lightly tuned. Don't stack — single-module attribution makes the comparison interpretable.

| Source | RX module | Settings hint |
|---|---|---|
| `MM05_FP1_60s_raw.wav` | **De-hum** (free-form) — *not* De-click | A 150-400 ms blink × 44× pitch shift becomes a 6.6-17.6 second "click" — far outside RX's De-click window. The blink envelope is closer to a hum sweep at this scale. Try free-form De-hum on the dominant low-frequency rumble. |
| `MM05_T7_60s_raw.wav` | **Spectral De-noise** | "Learn" the noise profile from a 2-3s region that corresponds to a calm period in the **original** EEG recording (not just an audio-quiet section — see Pitfall 2 below). Threshold ~6, Reduction ~6. |
| `MM05_OZ_60s_raw.wav` | **De-hum** + a touch of **Voice De-noise** | De-hum: free-form, target the dominant alpha tone (8-12 Hz × 44 = 352-528 Hz). Voice De-noise gentle (Adaptive mode, low Reduction). |

Export each cleaned version with this naming, into `outputs/audio/cleaned/`:
- `MM05_FP1_60s_rx.wav`
- `MM05_T7_60s_rx.wav`
- `MM05_OZ_60s_rx.wav`

**Settings that matter (per literature on audio→EEG transfer):**
- **Bit depth: 32-bit float** — *not* 16-bit. 16-bit's ~96 dB quantization noise can exceed EEG SNR (10-20 dB in band-of-interest), introducing a noise floor that overwhelms the signal you're trying to study.
- **Sample rate: 44100 Hz** (matches `audio_export.py`).
- **Phase preservation**: if RX exposes a "preserve phase" or "phase passthrough" mode for any module, **enable it**. Spectral phase carries event-related desynchronization information critical to imagined-speech decoding; magnitude-only cleanup with phase reconstruction (Griffin-Lim) corrupts the signal in a way that looks clean on a PSD but destroys classifier accuracy.

## Step 3 — objective comparison (1 command)

```bash
cd ~/eeg-speech
uv run python scripts/compare_audio_denoise.py \
    --raw-dir outputs/audio \
    --cleaned-dir outputs/audio/cleaned \
    --report-dir outputs/audio/reports
```

Per pair, this writes:
- `<title>_compare.png` — PSD overlay (true EEG Hz), spectrogram, residual envelope
- `<title>_compare.txt` — per-band % power removed + verdict
- `summary.json` — machine-readable totals across pairs

The verdict line uses these bands:
- **<1% RMS change** → no-op, RX did nothing
- **Targeted band hit + <50% overall** → promising, worth the next step
- **>50% RMS removed** → too aggressive, real EEG was stripped
- **Modest broadband cleanup** → check per-band % to judge

## Step 4 — does it actually help classification? (only if Step 3 is promising)

If RX cleaned in a targeted way (say, big alpha-band drop on OZ that matches removed bursts you can see in the spectrogram), the real test is:

1. Re-apply the RX cleanup to **all 62 channels** of MM05's full recording — either by hand (slow), or by writing a small batch wrapper that pipes through `ffmpeg afftdn` with parameters tuned to match what RX did
2. Re-import as MNE Raw via `audio_export.wav_to_channel()`
3. Drop into the SVM pipeline (`scripts/run_pipeline.py`) and rerun for MM05 only
4. Compare new MM05 row in `summary_results.csv` to the existing one

If accuracy goes up, write up the recipe and consider pushing to other subjects. If it goes down or stays flat, RX hasn't added anything ICA didn't already get — interesting null result, document it.

## Pitfalls (from a literature scan of audio→EEG transfer; agent research, 2026-04)

1. **Phase destruction is silent and fatal**. Magnitude-only spectral denoise + phase reconstruction (Griffin-Lim) looks clean on PSD but destroys event-related desynchronization patterns that classifiers depend on. Use modules with phase passthrough wherever possible. **The PSD comparison alone is not sufficient evidence that RX helped — the downstream classifier check (Step 4) is mandatory before drawing conclusions.**

2. **Frequency scaling breaks RX's noise model**. After 44× pitch shift, 60 Hz line noise sits at 2640 Hz — overlapping the audio band where RX's gamma-equivalent EEG content lives. RX's "Learn Noise" can't distinguish them. When learning a noise profile, gate the selection to a region that corresponds to a clean **EEG-silent period in the original recording**, not just an audio-quiet stretch.

3. **De-click is wrong for blinks at this scale**. A 150-400 ms blink × 44 = 6.6-17.6 s "click" — far outside RX's De-click window. The SOP table above uses De-hum for FP1 instead, since the time-stretched blink envelope behaves more like a low-frequency sweep at this pitch ratio. (Original instinct was wrong; this was the most useful catch from the research scan.)

4. **No ground truth for "did this help"**. PSD/power-in-band fidelity is a spectral-fidelity metric, not a task-relevant one. The downstream metric is classification accuracy on the 11-class / vowel-vs-consonant tasks. Plan to complete Step 4 before treating any positive PSD result as a real win.

5. **Subject-specific noise profiles are likely necessary**. iZotope RX is tuned for stationary or quasi-stationary noise. EEG artifacts vary subject-to-subject — a profile learned on MM05's T7 won't generalize automatically. Build the loop for one subject end-to-end before fanning out.

## Architectural notes

- **Don't fight ICA.** The `*-clean-epo.fif` files in `data/processed/` have already been ICA-cleaned. `audio_export.py` reads from `data/raw/` (pre-ICA), so RX is operating as an ICA *alternative*, not an addition. If you ever swap `load_raw_continuous` for the post-ICA epochs, RX has much less left to find.
- **Pitch shift is lossy in interpretation only**, not in math. Audio-domain power values in band reports are correct *relative* measures — % removed is directly comparable across runs.
- **Adjacent prior work** worth knowing about: multi-channel Wiener filter for EEG (Somers 2018, [PubMed 29393057](https://pubmed.ncbi.nlm.nih.gov/29393057/)) is the closest algorithmic analog to RX's Spectral De-noise. NMF-based EEG artifact removal (Damon 2013, [INRIA hal-00958775](https://inria.hal.science/hal-00958775/)) is the closest analog to RX's spectral decomposition. If RX shows a real win, those papers' parameter ranges are a good starting point for tuning.
