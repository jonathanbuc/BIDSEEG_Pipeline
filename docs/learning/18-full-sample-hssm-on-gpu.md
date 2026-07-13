# 18 — Running the full 42-subject HSSM, and putting it on the GPU

## What I changed

Two things, to get the whole OpenNeuro sample (42 usable subjects, ~18.6k trials) through the
hierarchical DDM and to make that fit fast.

1. **Taught `behavdata_prep` a second input schema.** The bundled demo feeds raw PsychoPy logs
   (`motionDir`, `keyMotionResp_2.*`, `trialCoh*`); the OpenNeuro release ships per-subject files
   already in the *processed* schema (`motion_direction`, `response`, `rt`, `coh_level`, …). The
   function now detects which shape it got: raw → the old select+rename; processed → keep the
   canonical columns and drop any precomputed derived columns, then run the **same** downstream
   derivations (RT cleansing, `response_prior`, block filter). One derivation path, either input.
   Two smaller fixes rode along: a diagnostic log line hard-indexed `coh_means['high']`, but this
   dataset's coherence is QUEST-thresholded to two levels (`low`/`medium`) — switched to `.get`;
   and the trait/psychosis-proneness step (a separate research thread, and the OpenNeuro trait file
   lacks the columns it needs) is now wrapped so a missing/mismatched trait file skips instead of
   crashing the module.

2. **Added a GPU path for the HSSM fit.** `hssm_gpu_runner.py` mirrors `e_HSSM_module`'s
   `prep_hssm_data` + `fit_hssm_hierarchical` but imports **only** the HSSM/jax/arviz stack — not
   `utils_module`, which pulls in `mne`/`fooof` that the WSL `hssm-gpu` env doesn't have. It reads
   the group behavioural CSV, samples on the GPU via `jax[cuda12]` under WSL, and writes the same
   posterior summary + InferenceData. Module d still runs first on the Windows `mne-env`.

## The CS concept — data-parallelism, and why it only pays off at scale

NUTS (the sampler) is **sequential**: each leapfrog step depends on the last, so you can't
parallelize *across* steps. What you *can* parallelize is the work *inside* one step — evaluating
the DDM log-likelihood and its gradient over every trial at once. That's a **SIMD / data-parallel**
workload: the same arithmetic applied to 18,614 independent rows, exactly what a GPU's thousands of
lanes are built for. At n=3 (~1.4k trials) there wasn't enough per-step work to hide the GPU's
fixed costs — kernel launches and XLA compilation dominated, and the GPU ran 1.8× *slower*. At
18.6k trials the per-step matrix is ~13× larger, the lanes fill up, and the fixed costs amortize:
**608 s on the GPU vs a 6–7 h pace on the same code on CPU — ~30–40×.** The lesson is the shape of
the curve, not a single number: GPU speedup grows with the parallel work per sequential step, so
the break-even sits somewhere between n=3 and n=42, and "benchmark, don't assume" was the right
call. Running CPU-vs-GPU as the *same program* in the *same env* (`JAX_PLATFORMS=cpu` vs `gpu`)
keeps the comparison honest — only the device changes.

The schema work is a small instance of the **adapter pattern**: rather than force every caller to
know which file shape it holds, one function normalizes both inputs to a single internal
representation and everything downstream stays oblivious.

## The psych/neuro concept — what the fit actually says

The model is a hierarchical drift-diffusion model: each trial's choice+RT is explained by a
**drift rate** `v` (how fast, and toward which boundary, evidence accumulates), a **starting point**
`z` (pre-evidence bias), a **boundary** `a` (caution), and a fixed **non-decision time** `t = 0.2 s`
(sensory+motor latency we don't try to estimate, since the task blocks responses for ~500 ms).
"Hierarchical" means each subject's parameters are drawn from a group distribution, so noisy
per-subject estimates borrow strength from the group. The result reproduces the lab's prior finding:
baseline drift is slightly negative, and **both** a low-level and a high-level prior push the drift
rate up (toward the expected option), with the **low-level prior's effect the larger** — priors act
mainly by biasing *evidence accumulation* (`v`), not the *starting point* (`z`). r_hat = 1.00 and
healthy effective sample sizes say the chains converged, so these aren't sampling artifacts.

## Why it helped — before → after

- **Before:** the pipeline's behavioural stages couldn't read the OpenNeuro files at all (schema
  mismatch → `KeyError`), and the one prior HSSM fit ran on CPU.
- **After:** module d runs clean on all 42 subjects (~6.5 min) producing an 18,614-trial group
  table, and the hierarchical DDM fits on the GPU in **~10 min instead of hours**, with converged
  posteriors that replicate the published direction of effects. Fast iteration is now on the table
  for the heavier alpha-covariate model (Stage 2).
