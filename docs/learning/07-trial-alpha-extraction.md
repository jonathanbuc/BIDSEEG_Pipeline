# 07 — Per-trial individual alpha frequency (the neural covariate)

*Teaching note for `extract_trial_alpha()` in `c_EEGAnalysis_module.py` — pulling one
alpha centre frequency out of every single trial, so it can later drive the drift model.*

---

## What I changed

| File | Change |
|------|--------|
| `c_EEGAnalysis_module.py` | New `extract_trial_alpha()` + a call in the per-subject loop (gated by `compute_fooof`) |
| `docs/learning/07-trial-alpha-extraction.md` | This note |

The existing `run_fooof_analysis()` averages the spectrum over **all trials of a condition**
and fits **one** FOOOF model per condition → one alpha number per condition. The new function
keeps every epoch separate and produces **one alpha centre frequency per trial**, written to
`results/groupEEG/trial_alpha/sub-XXX_trial_alpha.csv`.

For each epoch it stores two estimates:
- `alpha_cf_fooof` — centre frequency of the strongest FOOOF peak in 7–14 Hz (NaN if none).
- `alpha_cf_cog` — power-weighted mean frequency in 7–14 Hz after removing the 1/f background
  (always defined).

This is **milestone 1** (extract + validate). Wiring it into HSSM is milestone 2.

---

## The CS concept — going from a group estimate to a per-unit estimate, and the variance cost

The structural change is moving a computation *inside* a loop over individual units. Before,
the averaging happened **first** (mean spectrum over ~100 trials) and the model fit happened
**once**. Now the fit happens **per trial**, on ~1/100th of the data.

That trade is the bias–variance trade-off in concrete form:
- Averaging first → low variance (smooth spectrum), but you've thrown away all trial-to-trial
  information. You can never relate it to a single decision.
- Fitting per trial → high variance (each spectrum is noisy), but you recover the trial-level
  signal you actually need for a trial-by-trial regression.

The whole reliability question — "does per-epoch FOOOF even work?" — is asking how badly the
variance cost bites. The answer here (below) is: not badly, because occipital alpha is a large,
robust signal.

**Two estimators, deliberately.** `alpha_cf_fooof` can fail (no peak ⇒ NaN); `alpha_cf_cog`
cannot. Computing both, then checking how often FOOOF fails and how well the two agree, is a
cheap way to *measure* the variance cost instead of assuming it. This is the same idea as
having a strict parser and a lenient fallback, and logging how often you fall back.

**Frequency resolution is set by window length.** A Welch PSD over a window of length *T* has
bins spaced `1/T` apart. The 1.5 s prediction window → ~0.67 Hz bins. I force a **single** Welch
segment (`n_per_seg = full window`) because splitting into shorter sub-segments would average
the noise down but widen the bins to ~2 Hz — useless for locating a peak that lives in a 7 Hz-wide
band. So per-trial alpha and resolution are in direct tension, and the window length is the knob.

---

## The psych/neuro concept — individual alpha frequency and how you measure it

### Why occipital, why alpha
Alpha (7–14 Hz) is the dominant rhythm over visual cortex; the ROI `O1/Oz/O2` sits right over it,
so the alpha bump is large and easy to fit. Alpha is the band most tied to top-down expectation
and visual inhibition — which is exactly why it's the candidate neural marker for a prior-driven
decision task.

### Two ways to say "where is the alpha peak"
- **FOOOF centre frequency (CF):** fit the parametric model, read the CF of the tallest peak
  inside the band. Precise when a clear peak exists; undefined when it doesn't.
- **Centre of gravity (CoG):** treat the alpha-band spectrum as a distribution over frequency and
  take its mean, weighting each frequency by its power: `CoG = Σ f·P(f) / Σ P(f)`. This is the
  classic Klimesch / Corcoran "individual alpha frequency" estimator. It always returns a value
  and is robust to there being no single sharp peak.

### Why remove the 1/f background before the CoG
Raw power is always higher at the low end of any band because of the 1/f aperiodic slope. If you
weight raw power, the CoG is dragged toward 7 Hz regardless of where the real oscillation sits.
Subtracting the FOOOF-fitted aperiodic background (rebuilt from its `offset`/`exponent`) leaves the
genuine *oscillatory* power, so the weighting reflects the rhythm, not the slope. This is the same
"separate oscillation from background" logic that motivates FOOOF in the first place (see note 05).

### Individual alpha frequency is a real individual difference
Peak alpha is not fixed at 10 Hz — it varies person to person (and trial to trial). In the pilot
data the three subjects' mean IAF came out at ~10.9, ~11.4 and ~9.1 Hz: genuinely different alpha
"clocks." There's growing evidence (the Nature Communications paper Jonathan flagged) that *where*
your alpha sits, and how it shifts trial to trial, tracks cognitive processing — which is why it's
worth carrying per trial into the drift model.

---

## Why it helped / what improved — the validation

Run on the three pilot subjects (1169 cleaned trials total, occipital ROI, 1.5 s window):

| Check | Result | Reading |
|-------|--------|---------|
| **Coverage** (FOOOF peak found) | **92.6%** of trials | per-epoch FOOOF is reliable here; only ~7% need the CoG fallback |
| **Plausibility** | CF mean 10.5 Hz, median 10.9, all within 7–14 | values are physiological, not garbage/edge-pinned |
| **Agreement** FOOOF vs CoG | Pearson **r = 0.81**, bias +0.34 Hz | the two estimators measure the same thing |
| **Per-subject IAF** | 9.1 / 10.9 / 11.4 Hz | sensible individual differences |

Median FOOOF fit quality was r² ≈ 0.68–0.81. The takeaway: **per-trial alpha is usable on this
data.** `alpha_cf_fooof` is the primary (parametric, 92.6% coverage); `alpha_cf_cog` is the
always-available fallback that agrees strongly with it.

### What this unlocks (milestone 2)
The per-trial table joins onto `behavioraldata_hierprior.csv` by
`(participant, block_cond, block_order, thisN)`, after which the HSSM drift formula can read
`v ~ 1 + exp * coh_level + alpha_cf_fooof + (1|participant)` (or a trial-level version), testing
whether the alpha clock modulates evidence-accumulation rate. The ~7% NaN trials drop out of the
fit; if we'd rather keep every trial we use `alpha_cf_cog`.

---

## Key things to know for the PI quiz

- **Why two estimators:** FOOOF CF is precise but can be NaN; CoG is robust but coarser. Reporting
  coverage + their correlation *measures* per-trial reliability instead of assuming it.
- **Why a single Welch segment:** window length sets bin spacing (`1/T`); a 1.5 s window gives
  ~0.67 Hz — splitting it would blur to ~2 Hz and lose the peak.
- **Why subtract the aperiodic before CoG:** the 1/f slope biases a raw-power CoG toward low
  frequencies; removing it makes the weighting reflect true oscillatory power.
- **Coverage number:** 92.6% of trials yielded a FOOOF alpha peak — the headline reliability result.
- **Alignment:** `compute_psd` preserves epoch order, and metadata rows track surviving epochs, so
  the per-trial values line up trial-for-trial with the behavioural table.
