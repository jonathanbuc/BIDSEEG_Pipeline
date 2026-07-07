# 06 — Adding motion coherence to the HSSM drift formula

*Teaching note for putting sensory evidence strength (motion coherence) into the
hierarchical Bayesian DDM, including the prior × coherence interaction and the
mean-centering question.*

---

## What I changed

| File | Change |
|------|--------|
| `e_HSSM_module.py` → `prep_hssm_data()` | Recode `coh_level` (`low`/`medium` → `0`/`1`); add centered versions of continuous `coh` |
| `e_HSSM_module.py` (load-time default) | `formula_v` default now `v ~ 1 + exp * coh_level + (1|participant)` |
| `inputs.json → Analysis.hssm.formula_v` | Same default, so the formula is config-driven |
| `docs/learning/06-coherence-in-hssm.md` | This file |

Two coherence columns already live in the behavioral CSV:

- **`coh`** — continuous, the *actual* coherence fraction on each trial (e.g. `0.0562`,
  `0.12595`, `0.2519`). This is the physical stimulus: the proportion of dots moving
  coherently. It differs **between subjects** because each person was calibrated with a
  QUEST staircase (see below).
- **`coh_level`** — a string, `low` / `medium`. The *threshold-relative* difficulty band
  for that subject. Recoded to `0` / `1` so it enters the formula as a numeric slope
  (coefficient = "medium minus low").

The default formula moved from main-effects-only:

```
v ~ 1 + exp + (1|participant)            # before — prior condition only
v ~ 1 + exp * coh_level + (1|participant)  # after — prior × coherence
```

`exp * coh_level` is [Wilkinson/Bambi shorthand](https://bambinos.github.io/bambi/) that
expands to `exp + coh_level + exp:coh_level`: both main effects **plus** their interaction.

---

## The CS concept — design matrices, interactions, and centering

### How a formula becomes numbers

A formula like `v ~ 1 + exp + coh_level` is compiled into a **design matrix** `X`: one row
per trial, one column per predictor. The model estimates a coefficient vector `β` so that
`v_trial = X · β`. Categorical factors like `exp` (3 levels) are expanded into dummy
columns via **treatment coding**: `base` becomes the reference (folded into the intercept),
and `exp[lowlevel]`, `exp[highlevel]` are 0/1 indicator columns. So the intercept is "drift
in base at coherence-level low," and each coefficient is a *contrast* against that baseline.

### What an interaction term is

`exp:coh_level` adds **product columns** — `exp[lowlevel] × coh_level` and
`exp[highlevel] × coh_level`. Without it, the model assumes the coherence effect is the
*same size* in every prior condition (parallel lines). With it, each prior condition gets
its own coherence slope (lines can fan out). That is the whole scientific point: we want to
know whether priors reweight drift **differently depending on how good the sensory evidence
is** — you cannot see that with main effects alone.

### Why centering matters (the part you asked about)

A continuous predictor like `coh` has an arbitrary zero. Coherence is never literally 0 in
this experiment (the lowest value is ~0.056), so the raw intercept — "drift when `coh = 0`"
— is an extrapolation to a stimulus that never occurred. Two problems follow:

1. **Interpretability.** In a model *with an interaction*, the `exp` main effect is the
   prior effect *at `coh = 0`*. Nonsensical anchor → nonsensical main effect. Centering moves
   the zero to a meaningful place (the mean), so `exp` reads as "the prior effect at average
   coherence."
2. **Sampling geometry / collinearity.** Product terms are strongly correlated with their
   parents when the parent isn't centered. That correlation tilts and stretches the
   posterior, which makes NUTS work harder (lower effective sample size, occasional
   divergences). Centering largely decorrelates them and the sampler mixes better.

So **mean-centering `coh` is valid and standard.** The real question — the one you flagged —
is *which mean*.

### Global vs. within-subject centering

This is a genuine multilevel-modeling decision, and it matters **because of the QUEST
design**: each subject's `coh` values are individually calibrated, so a subject with a high
threshold has systematically higher `coh` across all their trials.

- **Grand-mean (global) centering** — `coh_gc = coh − mean(coh over all trials)`. The
  centered value now mixes two different things: "this trial was harder/easier *than typical
  for me*" **and** "I'm a high-threshold person." A coefficient on `coh_gc` is a blurred
  average of a within-person effect and a between-person effect. This is the classic
  multilevel trap (an ecological / Simpson's-paradox confound).
- **Within-subject (cluster-mean) centering** — `coh_wc = coh − mean(coh for this subject)`.
  Now the predictor is purely "was this trial's coherence above or below *my own* average,"
  cleanly isolating the **within-subject** trial-to-trial effect — which is exactly the
  drift-diffusion question ("does fluctuating sensory evidence change my accumulation rate?").

The textbook recommendation (Enders & Tofighi 2007; Bell, Fairbrother & Jones) for a
predictor that varies *within* a cluster is **within-subject centering**, optionally adding
the subject mean back as a separate between-subject predictor (the "within–between" or
Mundlak decomposition) when you also care about whether high-threshold people differ overall.

I implemented three derived columns so we can actually test this rather than argue it:

| Column | Definition | Isolates |
|--------|-----------|----------|
| `coh_gc` | `coh − grand mean` | within + between, blurred |
| `coh_wc` | `coh − participant mean` | pure within-subject |
| `coh_subjmean` | participant mean of `coh` | pure between-subject |

`coh_level`, by contrast, is *already* threshold-relative by construction — QUEST defines
"low" and "medium" relative to each subject's own threshold — so it needs no centering. That
is part of why the PI suspects the categorical version is the cleaner predictor.

---

## The psych/neuro concept — coherence as sensory precision

### What motion coherence is

In a **random-dot kinematogram (RDK)**, a fraction of dots move coherently (left or right)
while the rest move randomly. **Coherence** is that fraction. High coherence = an obvious,
low-uncertainty motion signal; low coherence = a weak, noisy signal you have to integrate
harder to read. In drift-diffusion terms, coherence is the **quality / precision of the
sensory evidence** feeding the accumulator.

### Why it belongs in the drift rate

Drift rate `v` *is* the rate of evidence accumulation, and that rate is governed by how
strong the evidence is. More coherent motion → steeper, more reliable drift toward the
correct bound → faster, more accurate decisions. Leaving coherence out of the `v` formula
forces the model to absorb a large, known source of trial-to-trial drift variation into
noise — biasing every other estimate. This is *the* standard covariate in perceptual DDM
work; omitting it is the conspicuous gap, which is why the PI flagged it first.

### QUEST thresholding — why `coh` differs per subject

Before the main task, each subject ran a **QUEST adaptive staircase** that homes in on the
coherence at which they hit a target accuracy. The experiment then presents stimuli at
*threshold-relative* levels (`low` / `medium`) rather than fixed physical coherence — so two
people doing "medium" trials may be seeing different physical `coh`. That's the source of the
between-subject variation in `coh`, and exactly why within-subject centering (above) is the
principled choice for the continuous version.

### The interaction — precision-weighted priors

The Bayesian-brain story is that the influence of a prior should scale with the
*(im)precision* of the sensory evidence: when the stimulus is ambiguous (low coherence), a
prior should pull the decision more; when the evidence is crisp (high coherence), the prior
should matter less. `exp:coh_level` is the term that lets the model express precisely this —
a different prior-driven drift shift at low vs. medium coherence. A credible non-zero
interaction would be direct evidence of **precision-weighting** in this dataset.

---

## Why it helped / what improved

**Before:** `v ~ 1 + exp + (1|participant)` attributed all drift variation to prior
condition and subject identity. The large, systematic effect of stimulus difficulty was
unmodeled — it leaked into residual noise and inflated uncertainty on every coefficient.

**After:** the model accounts for sensory evidence strength and can ask the precision-
weighting question directly. Concretely we now read:

- `Intercept` — drift in `base`, at low coherence, for the average subject
- `exp[lowlevel] / exp[highlevel]` — prior-driven drift shift *at low coherence*
- `coh_level` — extra drift going from low → medium coherence
- `exp[...]:coh_level` — how the prior effect *changes* with coherence ← the new science

**Empirical comparison (3 pilot subjects, 1329 trials, 1000 draws × 2 chains each).**
Five formulas were fit and compared on convergence and which drift coefficients had a 94%
HDI excluding 0:

| Model | Formula (drift) | max R-hat | min ESS | divergences | coherence effect credible? |
|-------|-----------------|-----------|---------|-------------|----------------------------|
| M0 | `exp` | 1.00 | 679 | 8 | — (no coherence term) |
| M1 | `exp * coh_level` | 1.00 | 626 | 2 | no (`coh_level` ≈ 0.01) |
| **M2** | `exp * coh_wc` | 1.00 | **1056** | **0** | no (`coh_wc` ≈ 0.12) |
| M3 | `exp * coh_wc + coh_subjmean` | 1.01 | 514 | 2 | no; `coh_subjmean` unidentified |
| M4 | `exp * coh_gc` | 1.00 | 788 | 2 | no (`coh_gc` ≈ 0.12) |

Three things came out of this, two of which directly answer the centering question:

1. **The prior effect is robust and is the real signal.** `exp[lowlevel]` ≈ **+0.34 to +0.40**
   with an HDI excluding 0 in *every* model; `exp[highlevel]` ≈ +0.18–0.20. Adding `coh_level`
   (M1) actually *sharpened* `highlevel` to credibly exclude 0, where the no-coherence baseline
   (M0) left it touching 0 — accounting for difficulty tightened the prior estimate.
2. **Within-subject centering beats global, empirically.** `coh_wc` (M2) gave the best
   sampling geometry — **highest ESS (1056) and zero divergences** — while global `coh_gc` (M4)
   sampled worse *and* lost the `highlevel` detection. This is exactly the prediction from the
   decorrelation argument above: cluster-mean centering removes the within/between confound the
   sampler was struggling with.
3. **The Mundlak between-subject term is unidentified at n=3.** `coh_subjmean` (M3) has a wide
   HDI around 0 and dragged min ESS down to 514 / R-hat to 1.01 — three subjects give three
   data points for a between-subject slope, so it can't be estimated. Revisit only with the
   full ~43-subject dataset.

**No coherence main effect or interaction is credible yet** — all coherence HDIs straddle 0,
with the most suggestive being `exp:coh_level[lowlevel]` = +0.19 (HDI −0.06 to +0.46). This is
expected: with three subjects this is a *plumbing / proof-of-concept* validation, not a power
test. The coherence and precision-weighting hypotheses are for the full dataset.

**Decisions taken from this:**
- **Default stays `v ~ 1 + exp * coh_level + (1|participant)`** — needs no centering, is the
  most interpretable for the paper, matches the PI's "use the threshold-relative level"
  intuition, and is where the (weak) interaction signal sits.
- **For the continuous option, use `coh_wc` (within-subject centered), not `coh_gc`** — the
  comparison confirms it samples better and detects more.
- **Skip the Mundlak `coh_subjmean` term until n is large enough** to estimate a
  between-subject slope.

### How to switch formulas (config only, no code edit)

| Goal | `formula_v` |
|------|-------------|
| Categorical level + interaction (default) | `v ~ 1 + exp * coh_level + (1\|participant)` |
| Continuous coherence, within-subject centered | `v ~ 1 + exp * coh_wc + (1\|participant)` |
| Within–between (Mundlak) decomposition | `v ~ 1 + exp * coh_wc + coh_subjmean + (1\|participant)` |
| Global-centered (to contrast against `coh_wc`) | `v ~ 1 + exp * coh_gc + (1\|participant)` |
| Both predictors, no interaction | `v ~ 1 + exp + coh_wc + coh_level + (1\|participant)` |

---

## Key things to know for the PI quiz

- **Why center at all:** with an interaction, the main effect is read at the predictor's
  zero; raw coherence has a meaningless zero, so center to make `exp` interpretable and to
  decorrelate the product term (better NUTS mixing).
- **Global vs within-subject:** because QUEST makes `coh` vary between subjects, grand-mean
  centering confounds within- and between-subject effects; within-subject (cluster-mean)
  centering isolates the trial-level effect that the DDM is actually about.
- **Why `coh_level` needs no centering:** it is already threshold-relative (0/1 around each
  subject's own calibration).
- **The interaction is the hypothesis:** `exp:coh_level` tests precision-weighted priors —
  whether expectations bias drift *more* when sensory evidence is weak.
