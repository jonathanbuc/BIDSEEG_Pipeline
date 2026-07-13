# HSSM results — full OpenNeuro sample (42 subjects)

**Model:** hierarchical Bayesian DDM (HSSM, `nuts_numpyro`, 2 chains × 1000 draws, GPU ~10 min).
**Data:** 42 usable subjects (sub-001 EEG missing), ~18.8k trials; ~17.4k enter the alpha model
after RT cleaning and coherence/response coding.

**Drift formula:** `v ~ 1 + exp * coh_level + alpha_cf_<est>_wc + (1|participant)`
**Also estimated:** `z ~ 1 + exp + (1|participant)`, `a ~ 1 + (1|participant)`, `t` fixed at 0.2 s.
Coding: `exp` = base / lowlevel / highlevel (base = reference); `coh_level` low→0, medium→1;
response = +1 prior-congruent / −1 incongruent; alpha covariate is within-subject centered.

## 1. Drift rate `v` — prior effects (robust)

| term | mean | 94% HDI | credible? |
|---|---|---|---|
| Intercept (base) | −0.03 | [−0.08, 0.01] | no (≈0) |
| exp[highlevel] | **+0.10** | [0.05, 0.15] | **yes** |
| exp[lowlevel] | **+0.28** | [0.23, 0.33] | **yes** |
| coh_level | +0.01 | [−0.04, 0.06] | no |
| exp:coh_level[lowlevel] | **+0.13** | [0.06, 0.20] | **yes** |
| exp:coh_level[highlevel] | −0.03 | [−0.10, 0.04] | no |

Both priors raise the drift rate toward the expected option, **low-level > high-level**, and the
low-level prior interacts with coherence. Matches your basic 43-subject DDM (drift ↑ with coherence
and prior; low-level effect larger).

## 2. Starting point `z` — dissociation

| term | mean | 94% HDI | credible? |
|---|---|---|---|
| Intercept | +0.01 | [−0.03, 0.05] | no |
| exp[highlevel] | −0.04 | [−0.09, 0.02] | no |
| exp[lowlevel] | **−0.13** | [−0.19, −0.07] | **yes** |

**Only the low-level prior shifts the starting point.** Combined with §1, the full-sample picture is
**`v` ← both priors, `z` ← low-level prior only** — the pre-evidence vs in-evidence bias separation,
now confirmed at n=42 (it was a 3-subject trend before). This holds identically whether or not the
alpha covariate is in the model.

## 3. Trial-by-trial individual alpha frequency → drift

**Estimator (pre-committed): Hilbert instantaneous frequency** ("frequency sliding"), following
**Romei & Tarasi (2026)**. Whole-scalp, within-subject centered.

| term | mean | 94% HDI | credible? |
|---|---|---|---|
| **`v_alpha_cf_hilbert_wc`** | **+0.028** | [−0.001, +0.061] | **no** (grazes 0) |

**No credible trial-level alpha→drift effect** — a weak positive trend whose HDI still includes zero.
We commit to the Hilbert estimator on principled grounds (the Romei & Tarasi method) *before* looking
at which estimator is favourable, so this is the honest answer, not a forking-paths artefact.

Why we expect a null here rather than a hidden effect: a diagnostic showed the two candidate per-trial
IAF estimators (Hilbert vs the spectral centre-of-gravity) are **essentially uncorrelated within
subject** (r ≈ 0.10), while their between-subject agreement is fine (r ≈ 0.59). A ~1.5 s epoch resolves
frequency only to ~0.67 Hz, but the within-subject trial-to-trial IAF SD is ~0.5 Hz — i.e. the
trial-level wobble is at the edge of measurability, so per-trial IAF is noise-dominated at this window
length. The stable between-subject IAF, or a more reliable trial marker (alpha power, aperiodic
exponent, CPP slope), is the more promising route for a trial-by-trial neural covariate.

> Robustness footnote: the centre-of-gravity estimator gives `v_alpha_cf_cog_wc` = −0.018
> [−0.039, +0.002] — opposite sign, also non-credible. The sign flip across estimators is exactly why
> we pre-committed to one.

## Notes / next steps

- Per-trial alpha is extracted **whole-scalp** (all channels), which gives 0 missing trials for COG
  and 100% Hilbert coverage — the extra power you suggested on 6/29 (vs occipital-only).
- Aperiodic exponent/offset are already computed per trial and could be added as covariates.
- Same pipeline, GPU-accelerated: any alternative covariate (alpha power, aperiodic exponent, later
  CPP slope) is a ~10-minute re-fit.

*Outputs:* `results/groupBehavioral/hssm_posterior_summary_{gpu,alpha,alpha_hilbert}.csv`
(+ matching `_*.nc` InferenceData). Figures for the canonical Hilbert model (Fig 4C coefficient
histograms, DDM schematic, v/z ridgelines, trace) are the `*_alpha_hilbert.png` files in the same
folder — regenerate any model's plots with `python plot_hssm.py inputs_openneuro.json <tag>`.
