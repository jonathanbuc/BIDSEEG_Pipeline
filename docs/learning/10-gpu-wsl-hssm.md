# 10 — Can we run HSSM on the GPU? (yes, correct, and slower)

*Teaching note for the WSL + `jax[cuda12]` experiment — getting the hierarchical DDM
onto the RTX 3060, validating it gives the same answer, and discovering it's the
wrong tool for this job.*

---

## What I did

Set up a GPU-capable HSSM environment and benchmarked it against the CPU run. No
pipeline code changed — this was a "can we go faster?" investigation. The artifacts
(the WSL conda env, throwaway fit scripts) live outside the repo; this note is the
deliverable.

The motivating fact: HSSM's `nuts_numpyro` sampler is JAX under the hood, and JAX runs
on NVIDIA GPUs — so in principle the 3060 should help, *for free*, no code edits.

---

## The setup story (and why it's WSL, not Windows)

1. **JAX has no native-Windows GPU build.** The official install docs say the CUDA
   wheels are *"only available on linux."* On Windows the GPU path is WSL2 (a real
   Linux kernel), which exposes the GPU through the Windows driver as
   `/usr/lib/wsl/lib/libcuda.so`.
2. In WSL Ubuntu: a fresh conda env, then `pip install hssm==0.2.10`, then
   `pip install "jax[cuda12]==0.6.2"`. The CUDA support arrives as separate packages
   (`jax-cuda12-plugin`, `jax-cuda12-pjrt`, and the bundled `nvidia-*` wheels);
   **`jaxlib`'s version doesn't change**, only the plugin is added.
3. **Dependency pinning bit hard.** A clean `pip install hssm` resolved to a *newer*
   stack than the working Windows one (jax 0.10, numpyro 0.21, numpy 2.4, pandas 3.0).
   Forcing `jax==0.6.2` for the CUDA wheels then left `numpyro 0.21` (which needs
   jax ≥ 0.7) broken. The fix was to pin the whole stack to the exact Windows versions
   (jax 0.6.2 / numpyro 0.19 / pymc 5.25.1 / numpy 2.2.6 / pandas 2.3.3). Pinning the
   *headline* package isn't enough; the transitive deps have to agree.
4. `jax.default_backend()` → `gpu`, `jax.devices()` → `[CudaDevice(id=0)]`. Confirmed.

---

## The CS concept — why a GPU loses at sequential MCMC

The benchmark, identical model, full 1000-draw fit:

| Backend | Wall time | `v_alpha_cf_cog_wc` | r̂ |
|---------|-----------|---------------------|----|
| CPU (Windows `mne-env`) | **1:46** | 0.061 | 1.0 |
| GPU (RTX 3060, WSL)     | 3:08 | 0.059 | 1.0 |

**Correct but ~1.8× slower.** The posteriors match within Monte-Carlo error (0.061 vs
0.059) — so the GPU stack produces *correct* inference; it's purely a speed question.

Why slower? Two compounding reasons:

1. **NUTS is sequential.** Each leapfrog step depends on the previous one, so the
   sampler's trajectory can't be parallelized. A GPU only parallelizes the work
   *inside* a step — the DDM likelihood over 1093 trials. That's a tiny matmul,
   nowhere near enough to fill thousands of CUDA cores. So every step pays GPU
   kernel-launch + host↔device sync latency and *loses* to a CPU core that just runs
   the small computation directly. This is Amdahl's law with the serial fraction
   pinned near 1.
2. **Compile overhead.** The first `sample()` call JIT-compiles CUDA kernels (~100 s
   here). It amortizes over more draws, but at this problem size it never pays back.

The general rule this illustrates: **GPUs win on wide, parallel work; they lose on
narrow, sequential work.** A small hierarchical DDM on 3 subjects is about as narrow
and sequential as Bayesian inference gets.

---

## The psych/neuro angle — when *would* the GPU pay off?

Nothing about the alpha/DDM science changed; what changes with the GPU is only
feasibility at scale. The GPU's per-step likelihood gets *relatively* cheaper as the
data grows, and numpyro can run many chains **vectorized** (all at once on the
device). So the regime where the 3060 should start winning is the real study:

- the full **~43-subject** dataset (≈14× the trials → a per-step likelihood big enough
  to actually occupy the GPU), and/or
- **many chains** run vectorized (e.g. 20–50) instead of 2.

Even then it's not guaranteed — NUTS stays sequential — so the honest plan is to
**re-benchmark on the full dataset**, not assume. For the n = 3 proof-of-concept, the
CPU is simply the right tool, and that's what the pipeline uses.

---

## Key things to know for the PI quiz

- **Why WSL:** JAX ships CUDA wheels for Linux only; native Windows is CPU-only.
- **How GPU support installs:** a `jax-cuda12-plugin` next to an unchanged `jaxlib`;
  it's a plugin, not a different jaxlib.
- **The benchmark:** GPU was correct (posteriors matched CPU) but ~1.8× slower at
  n = 3.
- **Why slower:** NUTS is sequential (can't parallelize the trajectory) and a 1093-
  trial DDM likelihood is too small to saturate the GPU; kernel-launch + compile
  overhead dominate. Amdahl's law.
- **When to revisit:** the full ~43-subject set with vectorized chains — then
  re-benchmark rather than assume a speedup.
