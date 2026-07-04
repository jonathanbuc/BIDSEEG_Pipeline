# 14 — Seeing the posterior: probability-of-direction, ridgelines, and a corrected diffusion cartoon

## What I changed
Added four plotting functions to `plotting_module.py` (so `e_HSSM_module.py` stays clean,
as Jonathan asked), all fed by the fitted model's ArviZ `InferenceData`:
- **`hssm_trace`** — a readable convergence trace (the old one crammed every variable on
  top of each other so the titles overlapped the axis above).
- **`hssm_posterior_coefficients`** — Romei & Tarasi (2026) Fig 4C: one histogram of each
  regression coefficient's posterior, with a dashed reference line at 0.
- **`hssm_posterior_ridgeline`** — Franzen et al. (2025) Fig 5: stacked posterior densities
  of one parameter across the levels of a factor.
- **`hssm_ddm_schematic`** — the drift-diffusion "anatomy" cartoon adapted from a conference
  colleague's `DDM_Plots_functions.py`, with two bugs fixed.

`e_HSSM_module.py` also now saves `hssm_idata.nc` so all of these can be regenerated without
re-running the (multi-minute) fit.

## The CS concept — separation of concerns, reconstructing derived quantities, link functions
1. **Separation of concerns.** The analysis module fits; the plotting module draws. Keeping
   plotting out of `e_HSSM` means the model code stays about modelling.
2. **A regression posterior stores a *basis*, not every condition.** The model reports an
   intercept plus per-level offsets; a specific condition's value is *reconstructed* by
   summing (`base = intercept`; `lowlevel = intercept + lowlevel_offset`). I select offsets
   by their coordinate *value* (e.g. `"lowlevel"`) rather than the library's auto-generated
   dimension name, so the plots survive formula changes.
3. **Link functions.** HSSM fits bounded/positive parameters on a transformed scale —
   `z` on a logit scale, `a` on a log scale. To plot them in natural units you must apply
   the inverse link: `expit` for `z`, `exp` for `a`. Forgetting this is a *silent*
   correctness bug — the numbers still plot, they're just wrong.

## The psych/neuro concept — what each picture means
- **The coefficient histogram** asks "is this effect credibly non-zero?" The annotated `q`
  is the **probability of direction** — the fraction of the posterior on the wrong side of
  0. It's the Bayesian cousin of a p-value, read directly off the samples. (Here
  `v_exp[lowlevel]` sits almost entirely above 0, `q = 0.002`; the alpha→drift coefficient
  straddles 0, `q = 0.23` — the right direction, but not credible at n=3.)
- **The ridgeline** shows whole posterior distributions per condition side by side, so you
  see overlap and shift rather than a single point estimate — the natural picture for "does
  drift differ between conditions?"
- **The diffusion cartoon** shows the model's anatomy: the boundary `a`, the start point,
  and the drift slope per condition. The fix that mattered: HSSM's `z` is *relative* (0–1),
  so the start point must be drawn at `z·a`, not `z` — otherwise the dot floats at the wrong
  height. (The original code also faked the uncertainty bands by shifting the intercept's
  HDI; here the HDIs come from the real posterior draws.)

## Why it helped
Before: one illegible trace plot and a posterior-summary CSV. After: a readable trace plus
three publication-style figures that make the posteriors interpretable at a glance, all
regenerable from a saved `InferenceData` without refitting. The functions are driven by the
`condition_dict`, so they carry straight over to the Natural Uncertainty dataset.
