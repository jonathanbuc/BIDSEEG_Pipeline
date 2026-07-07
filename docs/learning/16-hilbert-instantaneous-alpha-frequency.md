# 16 — A FOOOF-free per-trial alpha frequency (Hilbert "frequency sliding")

## What I changed

Jonathan flagged that our per-trial individual alpha frequency (IAF) had *"quite some trials with
poor FOOOF fit"* and suggested taking inspiration from Romei & Tarasi, who **don't use FOOOF**. So I
added a third per-trial estimator that reads alpha frequency straight out of the time domain:

- **New module `frequency_sliding.py`** — a pure numpy/scipy function
  `instantaneous_alpha_frequency(data, sfreq, l_freq, h_freq)` implementing the exact method from
  Romei & Tarasi (2026, *Nat Commun* 17:3384, Methods p.12), after Cohen (2014):
  1. **Derivative pre-whitening** of the signal (flattens the 1/f slope so it can't bias frequency).
  2. **Zero-phase Butterworth band-pass** (`scipy.filtfilt` = their MATLAB `filtfilt`) around the band.
  3. **Hilbert transform** → unwrapped instantaneous phase.
  4. **Instantaneous frequency = d(phase)/dt** (× fs/2π).
  5. **Median-filter denoising**: ten window lengths from 10–400 ms, take the median across them.
- **Wired it into `extract_trial_alpha`** (`c_EEGAnalysis_module.py`): two new columns,
  `alpha_cf_hilbert` and `alpha_if_sd_hilbert`. The band-pass is centred on **this subject's alpha
  peak (IAF ± 2 Hz)**, estimated from the far cleaner *trial-averaged* spectrum, falling back to the
  configured `alpha_freq_range` if no peak is found — exactly Romei & Tarasi's individualized band.
- **Built it test-first.** `tests/test_frequency_sliding.py` drives synthetic signals with a known
  frequency: a pure 10.5 Hz sine is recovered to <0.3 Hz, two tones come out ordered and near-truth,
  every epoch returns a finite value, and a large 1/f ramp added underneath doesn't shift the estimate.

## The CS concept — instantaneous frequency, and why a pure function

The headline idea is the **analytic signal**. The Hilbert transform turns a real band-passed signal
`x(t)` into a complex `x(t) + i·x̂(t)` whose angle is the oscillation's phase. If you unwrap that
phase and differentiate, you get **instantaneous frequency** — a frequency value at *every time
sample*, not one number for the whole window. That's the categorical difference from FOOOF: FOOOF is
a **frequency-domain model fit** (find a Gaussian bump in a spectrum), which needs a clean spectrum
and fails on short, noisy single trials; frequency sliding is a **time-domain read-out** that always
returns a value. Two failure modes it guards against: `filtfilt` runs the filter forwards then
backwards so it adds **zero phase distortion** (a one-directional filter would shift the very phase
we measure), and the **multi-window median** suppresses the sharp "phase-slip" spikes that a single
derivative throws off.

The other deliberate choice is architectural: the estimator lives in its **own side-effect-free
module**, not inside `c_EEGAnalysis_module.py`. Every pipeline module runs top-level code at import
(it reads `sys.argv[1]`), so anything defined there can't be imported in a unit test without faking a
run. Pulling the math into `frequency_sliding.py` (numpy/scipy only — no mne, no `inputs.json`) is
what let me do real TDD: **watch the test fail, then write code to pass it**, with synthetic signals
as ground truth. Hard-to-test is usually hard-to-use; the clean function is the payoff.

## The psych/neuro concept — the "alpha clock" and individual alpha frequency

Alpha (~8–12 Hz) isn't a fixed metronome; its **speed varies from person to person and moment to
moment**, and that speed matters. Romei & Tarasi's "alpha clock" account: a faster alpha sweeps
through more phase angles per unit time, so a brief stimulus is more likely to land at an *optimal*
phase for perception — faster alpha → better, faster evidence accumulation (it loads on drift rate
`v`, not on bias `z`). To test that trial-by-trial you need an alpha-frequency value *per trial*,
which is precisely what this estimator provides.

Why their FOOOF-free route is the right tool here: a single ~1.5 s epoch gives a spectrum at only
~0.67 Hz resolution with high variance, so fitting a discrete alpha peak fails on many trials
(hence the NaNs Jonathan saw). Reading frequency from the phase sidesteps peak detection entirely.
The **derivative pre-whitening** step is the clever part — the 1/f aperiodic background would
otherwise pull a naive frequency estimate downward; differentiating removes that tilt without needing
to *model* it the way FOOOF does. That's the sense in which this is genuinely "FOOOF-free."

## Why it helped — before → after

- **Before:** `alpha_cf_fooof` was `NaN` on every trial where FOOOF found no clean alpha peak; even
  the `alpha_cf_cog` "fallback" still leaned on FOOOF's fitted 1/f to work.
- **After:** verified on real sub-001 epochs (336 trials, 250 Hz) — `alpha_cf_hilbert` has **100%
  coverage**, mean **8.71 Hz** (matching the subject's 8.7 Hz average-spectrum IAF), per-trial range
  7.2–10.3 Hz, SD 0.45 Hz. The synthetic-signal tests confirm it recovers a known frequency to
  <0.3 Hz and is unbiased by a strong 1/f trend. There is now an alpha covariate defined on *every*
  trial to regress onto the DDM drift rate.

## Follow-up

`alpha_cf_hilbert` is produced and saved per trial, and `e_HSSM.prep_hssm_data` now includes it in
the alpha-centering tuple, so it gets the same within/between-subject centering (`_gc` / `_wc` /
`_subjmean`) as the other alpha columns. Using it as the drift covariate is then a one-liner in
`inputs.json` — e.g. `"formula_v": "v ~ alpha_cf_hilbert_wc + (1|participant)"` for the pure
trial-to-trial ("does the alpha clock move evidence accumulation") effect, which `load_group_data`
resolves by loading the per-trial alpha table.
