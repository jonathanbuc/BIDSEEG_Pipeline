# 19 — Trial-by-trial alpha → drift rate, and why the estimator flips the answer

## What I changed

With the full 42-subject sample processed (`EEG_iaf.csv`, 18.8k trials, per-trial individual
alpha frequency at 0 missing), I fit the **alpha-covariate hierarchical DDM** — the model Jonathan
has wanted since June: drift rate regressed on condition, coherence, *and* a per-trial neural
marker, with each subject partially pooled. Then I ran it **twice**, swapping only the alpha
estimator:

- `alpha_cf_cog_wc` — the frequency-domain **centre of gravity** (power-weighted mean frequency in
  the alpha band, 1/f removed).
- `alpha_cf_hilbert_wc` — the time-domain **instantaneous frequency** ("frequency sliding": the
  derivative of the Hilbert phase inside a subject-specific alpha band).

Both are within-subject centered (`_wc`) so the term isolates the *trial-to-trial* alpha wobble.
Two supporting fixes: the HSSM loader was pointing at a stale path (`groupEEG/trial_alpha/…`) while
module c actually writes `SpectralParameterization/EEG_iaf.csv` — redirected it in both
`e_HSSM_module` and the GPU runner. And the per-trial alpha is computed **whole-scalp**, not
occipital-only (the `.pick(roi)` was already commented out; I fixed the misleading comment).

## The CS concept — robustness checks and researcher degrees of freedom

The headline isn't a coefficient, it's a **robustness check**. The prior→drift and prior→startpoint
effects are identical across the two runs; the alpha→drift term **flips sign** (COG −0.018, Hilbert
+0.028) and **neither credible interval excludes zero**. When a result depends on an analysis choice
that has no single "right" answer — here, how you reduce a noisy per-trial spectrum to one alpha
frequency — that choice is a **researcher degree of freedom**, and reporting only the favourable one
would be a garden-of-forking-paths error. Running the same pipeline end-to-end under both estimators
is the cheap, honest way to expose that fragility. It's only cheap because the fit moved to the GPU
(~10 min each instead of hours): fast, reproducible re-runs are what make a robustness check
practical rather than aspirational. The deterministic pipeline also mattered downstream — when a
process-management slip left several EEG runs going at once, the duplicate rows they produced were
*bit-identical*, so `drop_duplicates()` recovered a clean table with no re-processing.

## The psych/neuro concept — individual alpha frequency as a clock, and two ways to read it

Individual alpha frequency (IAF, ~8–12 Hz) is increasingly treated not as a fixed trait but as a
**trial-varying "clock rate" for perception** — faster alpha is theorised to sample the visual world
in finer temporal slices. The question was whether that clock tracks the **drift rate** `v`, the
speed of evidence accumulation in the DDM. The two estimators answer differently *because they
measure different things*: the COG is a **spectral** summary (where alpha-band power sits, averaged
over the ~1.5 s window), while the Hilbert instantaneous frequency is a **temporal** read (how fast
the phase turns, moment to moment). On short, noisy single trials these need not agree — and here
they don't even agree on sign, which is itself the finding: with this data and these estimators, the
IAF→drift link is not established. Meanwhile the **dissociation is robust** — priors shift *evidence
accumulation* (`v`, both levels) and the low-level prior additionally shifts the *starting point*
(`z`), a pre- vs in-evidence bias separation that holds regardless of the alpha choice.

## Why it helped — before → after

- **Before:** the alpha↔behaviour link had only ever been a plan; there was no full-sample estimate,
  and no sense of whether it was real or estimator-artefact.
- **After:** a definitive n=42 answer — the prior effects replicate cleanly and the alpha→drift term
  is a weak, sign-unstable effect that **can't be claimed without first settling the estimator**.
  That turns a vague open question ("does averaging center frequency make sense?") into a concrete
  next step: decide the IAF estimator on principled grounds (or preregister it) before treating any
  alpha→drift coefficient as evidence.
