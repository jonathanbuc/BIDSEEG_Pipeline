# 03 — HSSM: Hierarchical Bayesian DDM (new module `e`)

*Teaching note for adding `e_HSSM_module.py` — a Hierarchical Sequential Sampling Model
analysis on top of the existing pyddm behavioral module.*

---

## What I changed

Four files touched:

| File | Change |
|------|--------|
| `e_HSSM_module.py` | New module — hierarchical Bayesian DDM across all subjects |
| `environment_setup.yml` | Added `hssm` to pip dependencies |
| `inputs.json` | Added `perform.compute_hssm: false` and `Analysis.hssm` config block |
| `docs/learning/03-hssm-integration.md` | This file |

The module:
1. Loads the group behavioral CSV written by module `d` (`behavioraldata_hierprior.csv`)
2. Recodes the `response_prior` column (0/1) to HSSM's expected format (-1/1)
3. Fits a hierarchical Bayesian DDM where **drift rate `v` varies by experimental condition**
   (`base` / `lowlevel` / `highlevel`) with a **random intercept per participant**
4. Runs MCMC sampling (NUTS via PyMC) and saves the full posterior summary to CSV plus
   diagnostic trace plots

The key config parameters (all in `inputs.json → Analysis.hssm`):

```json
{
  "model_type": "ddm",
  "sampler": "mcmc",
  "draws": 1000,
  "tune": 1000,
  "chains": 2,
  "cores": 2,
  "target_accept": 0.9,
  "prior_settings": "safe",
  "link_settings": "log_logit",
  "formula_v": "v ~ 1 + exp + (1|participant)"
}
```

Set `perform.compute_hssm: true` in `inputs.json` to run it (off by default because
MCMC is slow — expect 15–60 min for a full dataset).

---

## The CS concept — Bayesian inference vs. MLE, MCMC/NUTS, and hierarchical models

### MLE (what pyddm does)

pyddm fits the DDM using **Maximum Likelihood Estimation** via differential evolution.
It finds the single point in parameter space $\hat\theta$ that maximises
$p(\text{data} \mid \theta)$.
Result: one number per parameter per fit. No uncertainty.
After fitting, the PI runs a paired t-test comparing, say, `drift_base` vs `drift_lowlevel`
across subjects — but this treats the estimated parameters as if they were measured
without error, which they aren't.

### Bayesian inference (what HSSM does)

HSSM's goal is the **posterior distribution** $p(\theta \mid \text{data})$.
By Bayes' rule:

$$p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta) \cdot p(\theta)$$

- $p(\text{data} \mid \theta)$ is the likelihood (same as MLE)
- $p(\theta)$ is the **prior** — your belief about plausible values before seeing data
- The result is a *distribution* over $\theta$, not a point. You can ask: "what is the
  probability that drift rate is higher in `highlevel` than `base`?" and get a direct answer.

### MCMC and NUTS

The posterior is usually intractable analytically. **Markov Chain Monte Carlo (MCMC)**
approximates it by drawing samples: construct a Markov chain whose stationary distribution
is the posterior, run it long enough, and the samples you collect approximate $p(\theta \mid \text{data})$.

HSSM uses **NUTS (No-U-Turn Sampler)**, an adaptive variant of Hamiltonian Monte Carlo:
- Uses gradient information (PyMC computes gradients via autodiff / JAX) to make big,
  informed jumps through the parameter space rather than tiny random-walk steps
- "No-U-Turn" means it automatically stops the trajectory before it doubles back —
  so you don't have to tune step size or trajectory length by hand
- `tune` steps at the start warm up the sampler (adapt step size); `draws` are the
  actual posterior samples you keep; `chains` are independent runs (use for convergence checks)

The convergence diagnostic you'll look at is **R-hat** (or $\hat{R}$): for a parameter
to be converged, all chains should be sampling the same distribution →
$\hat{R} \approx 1.0$. Values > 1.1 are a red flag.

### Hierarchical models — partial pooling

The formula `v ~ 1 + exp + (1|participant)` creates a **multilevel model**:

```
Group level:  Intercept ~ Normal(0, σ_group)
              exp[lowlevel], exp[highlevel] ~ Normal(0, ...)

Subject level: participant offset_i ~ Normal(0, σ_subject)

Per-trial:    v_trial = Intercept + β_exp * exp + offset_participant_i
```

This is called **partial pooling** (compared to the two extremes):
- **No pooling** (pyddm): fit each subject independently. Problem: with small N, noisy data
  produces extreme individual estimates.
- **Complete pooling**: ignore subject identity, fit one model. Problem: erases individual
  differences.
- **Partial pooling** (HSSM): subject parameters are drawn from a group distribution. Each
  subject informs the group estimate; the group estimate regularises each subject. Works much
  better for small-N data typical in EEG experiments.

---

## The psych/neuro concept — DDM, the hierarchical priors task, and the EEG link

### What the DDM parameters mean

The **Drift Diffusion Model** describes a two-alternative forced-choice decision as noisy
evidence accumulating over time until it hits one of two boundaries:

| Parameter | Symbol | What it represents |
|-----------|--------|--------------------|
| Drift rate | `v` | Speed and direction of evidence accumulation. Higher = faster, more biased decision. |
| Boundary separation | `a` | How much evidence is required before deciding. Wide = conservative, slow; narrow = impulsive, fast. |
| Bias / starting point | `z` | Prior bias toward one option (0.5 = no bias; > 0.5 biased toward upper bound). |
| Non-decision time | `t` | Time for sensorimotor processes before/after the decision itself (encoding + response). |

### Why drift rate is the key parameter here

In the **Hierarchical Priors RDK task**, participants see random-dot-kinematogram (RDK)
motion and decide left vs. right. The manipulation is whether they have:
- **base**: no prior, just detect motion
- **lowlevel**: a visually-presented (perceptual) prior for motion direction
- **highlevel**: an abstract (cognitive) prior about which direction is more likely

The scientific hypothesis is that priors — whether low-level or high-level — **speed up
evidence accumulation** toward the expected direction. That maps directly onto drift rate
`v`: a stronger (or better-calibrated) prior should push `v` higher on congruent trials.

The model formula `v ~ 1 + exp + (1|participant)` estimates:
- `Intercept` = average `v` in the `base` condition
- `exp[lowlevel]` = how much drift *changes* in the low-level prior condition vs base
- `exp[highlevel]` = how much drift changes in the high-level prior condition vs base
- `1|participant` = each person's overall "fast vs slow" tendency, independent of condition

If the 95% Highest Density Interval (HDI) on `exp[highlevel]` excludes 0, you have
credible evidence that high-level priors genuinely shift drift rate.

### The future EEG link (not yet implemented)

The most scientifically exciting HSSM feature for this pipeline is **trial-level regression**:

```
"formula_v": "v ~ alpha_power + (1|participant)"
```

This would say: *each trial's drift rate is predicted by that trial's alpha oscillation
strength*. If pre-stimulus alpha in visual cortex suppresses sensory processing, you'd
expect it to *lower* `v` (slower evidence accumulation on high-alpha trials). This is
exactly the kind of direct EEG-behavior link the whole EEG recording was set up to test.

Implementing this requires module `c` to save per-trial alpha power (not just condition
averages), then merging that with the behavioral trial data before passing to HSSM.
The module is architected to support this: just change `formula_v` in `inputs.json` and
make sure the predictor column is in the data.

---

## Why it helped / what improved

### Before (pyddm + t-test)

```
Subject 001:  v_base=1.2,  v_lowlevel=1.5,  v_highlevel=1.8
Subject 002:  v_base=0.9,  v_lowlevel=1.0,  v_highlevel=0.8
Subject 003:  v_base=2.1,  v_lowlevel=2.0,  v_highlevel=2.4

→ Paired t-test: lowlevel vs base — is the mean difference > 0? (p-value)
```

Problems:
1. Three separate fits, three separate parameter estimates, three separate error sources —
   each treated as exact truth in the t-test
2. Subject 3's extreme values pull the group mean and add noise
3. The t-test only asks "is the mean non-zero?" — not "how large is the effect?"

### After (HSSM)

```
Posterior on v:
  Intercept (base):          mean=1.3, 95% HDI [0.7, 1.9]
  exp[lowlevel]:             mean=+0.3, 95% HDI [+0.05, +0.55]  → excludes 0 ✓
  exp[highlevel]:            mean=+0.5, 95% HDI [+0.2, +0.8]    → excludes 0 ✓
  SD across participants:    mean=0.5, 95% HDI [0.2, 0.9]
```

What this buys:
1. **Direct credible statement**: "The probability that high-level priors increase drift
   rate is 97%." No p-value, no t-test, no treating estimates as known.
2. **Partial pooling regularises subject 3**: the prior on the group distribution pulls
   the extreme value toward the group mean — less sensitivity to one noisy participant.
3. **Uncertainty propagation**: parameter estimation uncertainty feeds directly into
   condition comparisons; you can't mistake a wide posterior for a reliable effect.
4. **Richer output**: you get the full posterior, not just means and SDs, so the PI can
   compute any contrast they want (e.g. P(highlevel > lowlevel)) without re-running the model.

The posterior summary CSV (`hssm_posterior_summary.csv`) has mean, SD, HDI, and R-hat for
every parameter, ready for the paper. Trace plots show convergence.

---

## Key diagnostic to know for the PI quiz

**R-hat ($\hat{R}$)**: ratio of between-chain variance to within-chain variance. If all
chains converged to the same posterior, $\hat{R} = 1$. The rule of thumb is $\hat{R} < 1.01$
for good convergence. If it's > 1.1, something went wrong (too few samples, poor
initialisation, badly specified priors, multimodal posterior). The trace plot should show
chains that look like hairy caterpillars overlapping each other — not drifting or separated.

**ESS (Effective Sample Size)**: MCMC samples are autocorrelated. ESS accounts for that.
Aim for ESS > 400 per parameter per chain. Low ESS means your 1000 draws are effectively
only 50 independent ones — increase `draws` or fix the model.
