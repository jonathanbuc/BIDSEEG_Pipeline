# 02 — Paper summaries: the science behind the HSSM work

These are plain-language summaries of the three documents read this session, written so I can
both *understand* the work and *brief Jonathan and Guido* on it. Two are published/preprint papers;
the third is the lab's own preregistration for the next study. For each I give the one-sentence
version, what they did, what they found, the key EEG + modelling methods, and **why it matters for
our pipeline**.

> **Reality check before we start.** Everything our pipeline has *produced* so far is from the
> **3-subject Hierarchical-Priors RDK demo dataset**. Those numbers are plumbing and validation —
> they show the code runs and the figures render correctly — **not scientific results**. Don't
> interpret anything below ~10 observations. The papers here are the *targets* we're learning to
> reproduce; the demo is just us checking our instruments against them.

## A 60-second glossary (drift-diffusion model terms)

Both papers model a two-choice decision as **noisy evidence piling up over time until it hits a
boundary**. The drift-diffusion model (DDM) gives that process four interpretable knobs:

| Term | Symbol | Plain meaning |
|------|--------|---------------|
| **Drift rate** | `v` | How *fast and cleanly* evidence accumulates toward the correct boundary. Higher `v` = better signal quality / faster, more accurate decisions. This is the **"gain"** knob. |
| **Boundary separation** | `a` | How much evidence you demand before committing. Wider `a` = more cautious, slower but more accurate. |
| **Start point** | `z` | Where accumulation *begins* between the two boundaries, **before any evidence arrives**. An off-centre `z` is a pre-evidence bias. This is the **"origin"** knob. |
| **Non-decision time** | `t` | Time spent on stimulus encoding + motor response, *not* on deciding. |
| **Drift criterion / drift bias** | `dc` | A *constant pull* added to the drift during accumulation that favours one choice regardless of the stimulus — a bias that acts *during* evidence processing rather than before it. |

Two more terms from signal detection theory (SDT), used as a simpler parallel analysis:
- **Sensitivity (`d'`)** — how well a person separates signal from noise (perceptual acuity).
- **Criterion (`c`)** — how biased they are toward saying "yes/present" vs "no/absent".

And the Bayesian vocabulary, since both papers fit their DDMs *Bayesianly*:
- **Posterior** — the full probability distribution of a parameter *after* seeing the data, rather
  than a single best-fit number. You get a whole cloud of plausible values.
- **HDI (highest-density interval)** — the band containing the most credible X% (usually 95%) of
  that posterior. If two conditions' HDIs don't overlap, the difference is credible.
- **`q`-value / probability of direction** — the fraction of posterior samples on the "wrong" side
  of zero. It's the Bayesian cousin of a p-value: `q` near 0 means the effect is credibly non-zero
  and points consistently in one direction. (Romei & Tarasi report `q`, not `p`.)
- **Random intercept** — in a mixed model, a per-subject baseline offset (written `(1|subject)`).
  It lets each participant have their own level while still estimating one shared group effect — the
  "hierarchical" in hierarchical DDM. It's why these models behave well with modest sample sizes.

---

## 1. Romei & Tarasi (2026) — *Alpha frequency shapes perceptual sensitivity by modulating optimal phase likelihood*
*Nature Communications 17:3384.*

### In one sentence
The moment-to-moment **speed of your occipito-parietal alpha rhythm** (instantaneous individual
alpha frequency, IAF) predicts how *accurately* you perceive — faster alpha → better detection —
and in their DDM that benefit lands specifically on the **drift rate `v`**, not on bias.

### What they did
- A large visual **contrast-detection** task (n ≈ 125; ~116 with full behavioural data). On each
  trial participants reported whether faint grey circles were present in a briefly flashed
  black-and-white checkerboard.
- The stimulus lasted only **59 ms** — deliberately *shorter than a single alpha cycle* — and
  contrast was titrated per person to ~70% accuracy, so the design is exquisitely sensitive to
  *when* in the alpha cycle the stimulus lands.
- They related **pre-stimulus IAF** (alpha frequency in the window before the flash) to behaviour
  using four converging methods: tercile **binning**, **Bayesian** statistics (Bayes factors),
  **signal detection theory** (`d'` vs `c`), and a **single-trial DDM**.

### What they found
- **Faster alpha → more accurate, more sensitive perception.** Trials in the top IAF tercile had
  higher accuracy (0.76 vs 0.72) and higher sensitivity (`d'` 1.65 vs 1.46) than the bottom tercile.
- **Bias was essentially untouched.** The SDT criterion `c` barely differed between IAF terciles —
  faster alpha buys you acuity, not a change in your yes/no tendency.
- **In the DDM, IAF loaded onto drift rate.** With IAF as a single-trial covariate, the posterior
  mean **IAF → `v`** coefficient was **0.049, HDI [0.026, 0.073], `q` < 0.001** (credibly positive),
  while the **IAF → `z`** coefficient was a null **0.001, HDI [−0.003, 0.005], `q` = 0.392**. Faster
  alpha speeds up evidence accumulation; it does not shift the starting point.
- **Mechanism — the "alpha clock".** Faster alpha sweeps through *more phase angles* within the
  brief stimulus window, raising the chance the stimulus coincides with an *optimal* phase for
  perception. A second DDM adding alpha phase and the IAF×phase interaction showed phase also drives
  `v` (coef 0.064, `q` = 0.002) and that phase matters *more when alpha is slow* (negative
  interaction, coef −0.041, `q` = 0.043) — exactly what you'd expect if fast alpha already
  guarantees good phase coverage.

### Key methods (EEG + DDM)
- **EEG:** occipito-parietal electrodes (Oz, POz, O2, PO4, PO8); per-subject pick of max-alpha
  channel. **Instantaneous frequency** = temporal derivative of the Hilbert phase angle of
  band-limited alpha, median-filtered to denoise. (Note the contrast with our future study, which
  defines IAF differently — see the prereg below.)
- **DDM:** the **HDDM** Python package (Bayesian, MCMC). IAF was **z-scored**; RTs trimmed to
  100 ms–4 s; 5000 posterior samples with 500 burn-in. Parameters `v, z, a, t` were each given a
  regression on IAF. Results reported as `q`-values (posterior mass crossing zero).

### Why it matters for our pipeline
- This is **the conceptual basis for our `v ~ alpha` covariate.** When `e_HSSM_module.py` regresses
  trial-by-trial EEG alpha onto the drift rate, it is operationalising exactly Romei & Tarasi's
  "alpha clock → evidence accumulation" claim. (See learning note `../learning/11-eeg-predicts-drift-rate.md`.)
- **It's the figure we recreated.** Their **Fig 4C** — a row of histograms, one per DDM regression
  coefficient, each with a **dashed vertical line at 0** and annotated with how much posterior mass
  crosses it — is the template for our `hssm_posterior_coefficients` plot. The dashed-zero line is
  the load-bearing element: it turns "is this effect real?" into "how much of the posterior is on
  the wrong side of zero?" See [./05-plots.md](./05-plots.md) for our version.
  - *Demo caveat:* on our 3-subject data the alpha→drift coefficient straddles zero (`q ≈ 0.23`) —
    the right *sign*, but nowhere near credible. That's the expected look of an underpowered run, not
    a refutation of Romei.

---

## 2. Franzen et al. (2025) — *Prior Information Shapes Perceptual Evidence Accumulation Dynamics Differentially in Psychosis*
*bioRxiv preprint, 2025-07-03.*

### In one sentence
When people use a **prior expectation** to decide, healthy individuals fold it into the **evidence
accumulation itself** (the *gain* route — drift), whereas people with psychosis lean on a
**pre-evidence starting bias** (the *origin* route — start point) — a clean `v`-vs-`z` dissociation.

### What they did
- Compared **healthy controls vs patients with psychosis** on **detection tasks in two modalities**:
  hearing a human *voice* in noise, and seeing a human *face* in a noisy image, each at two noise
  levels.
- Manipulated **prior probability at the block level** with truthful cues: **P−** (targets rare),
  **P=** (uninformative, 50/50), **P+** (targets common).
- Asked the core theoretical question: does a prior act **before** evidence (shift the **start point
  `z`** — the *origin model*) or **during** accumulation (change the **drift criterion / drift rate**
  — the *gain model*), or **both** (*multi-stage model*)?
- Fit **hierarchical drift-diffusion models** to RTs and choices, alongside SDT (`d'`, `c`) and
  choice-probability analyses as cross-checks.

### What they found
- **Healthy people use gain.** Priors modulated their **drift criterion and drift rate** — the
  expectation reshaped how evidence was *interpreted as it came in*. Classic gain model.
- **Psychosis patients use origin.** They showed **diminished sensory gain** and instead leaned on a
  **pre-evidence start-point bias** `z`. Their drift bias varied much less across cue conditions.
- **A conservative baseline for everyone.** All participants showed a **negative drift bias** — they
  treated ambiguous evidence as favouring the "non-target" choice — but only healthy controls
  *amplified* that bias with informative priors.
- **Patients are more cautious.** Boundary separation `a` was higher in patients (1.80 vs 1.60, with
  non-overlapping HDIs, P = 0.997) — they demand more evidence before committing.
- **Clinical link.** Greater positive-symptom severity predicted a *reduction* in the traditional
  SDT criterion bias, hinting that "sensory gain deficit" could be a computational marker of
  psychosis progression.

### Key methods (EEG + DDM)
- This study is **behavioural + computational** (no EEG analysis in the parts relevant to us) — its
  value is the **modelling logic**, not a signal-processing recipe.
- **HDDM** toolbox (Bayesian, MCMC), fit with **stimulus coding** so the model can express a
  *start-point* bias separately from a *drift* bias. They explicitly separated `z` (origin), `dc`
  (gain during accumulation), `a`, and `t`, and read off effects from posterior overlap.

### Why it matters for our pipeline
- **This is the backbone of our start-point-vs-drift-bias separation.** Jonathan's whole point — that
  a prior can bias a decision in *two mechanistically different places* — comes straight from this
  origin-vs-gain framing. It's why `e_HSSM_module.py` can fit a **`z` (start-point bias) model**
  distinct from the drift model. (See learning note `../learning/12-startpoint-vs-drift-bias.md` and
  [./01-meeting-jonathan.md](./01-meeting-jonathan.md) for Jonathan's anchor: *`z` ← low-level prior,
  `v` ← both priors*.)
- **It's the second figure we recreated.** Their **Fig 5** stacks **posterior densities (KDEs) of a
  DDM parameter across conditions** — `z` and drift bias drawn as overlapping filled curves, one per
  cue level, for each group. That "ridgeline / joyplot" is the template for our
  `hssm_posterior_ridgeline` plot (`hssm_ridgeline_z_by_exp.png`, `hssm_ridgeline_v_by_exp.png`). It
  answers "does this parameter *shift* across conditions?" by showing whole distributions side by
  side instead of single dots. See [./05-plots.md](./05-plots.md).
  - *Demo caveat:* with 3 subjects our ridgelines mostly show *overlapping* curves — again, the
    honest look of an underpowered pipeline test, not a finding.

---

## 3. Preregistration — *"Drifting Through Natural Uncertainty"* (final version)
*The lab's next study. No data collected yet.*

### In one sentence
A planned EEG study that uses **AI-generated bear/dog composite images** at three "blend" levels
(sensory uncertainty) preceded by a bear-or-dog **cue** (perceptual prior), to test how priors and
uncertainty jointly shape **HSSM decision parameters**, the **1/f aperiodic EEG spectrum**, and the
**centro-parietal positivity (CPP)** — i.e. the exact pipeline we're generalising toward.

### What they will do
- **Stimuli:** a pretrained AI model perturbs animal images so each contains both bear and dog
  features at **three mixing ratios → low / medium / high sensory uncertainty**. The participant
  reports the *dominant* animal plus a confidence rating.
- **Prior:** each target is preceded by a *clear* bear or dog **cue**, valid on **75% of trials** —
  inducing a template-based expectation. Valid = cue matches the base image, giving **cue
  congruency** (congruent vs incongruent).
- **Design:** **2 × 3 within-subjects** (cue congruency × sensory uncertainty), **453 trials** (151
  per uncertainty level → 339 congruent, 114 incongruent), 3 blocks. **Target n = 45** complete
  datasets.

### Confirmatory hypotheses (the pre-committed ones)
- **HSSM / behaviour:** **bias-related parameters — drift rate `v` and start point `z` — are higher
  in congruent than incongruent trials** (a congruent cue should both pre-bias the start point and
  speed the drift). Drift rate should also rise with **subjective confidence**.
- **Choice frequency:** distractor-animal classifications should *increase* in incongruent trials as
  uncertainty rises (a congruency × uncertainty effect on raw choices). *(This analysis was newly
  added in the final version — see "what changed" below.)*
- **Aperiodic EEG (1/f):** the **SpecParam** aperiodic exponent/offset will differ between congruent
  and incongruent trials (tested with cluster-based permutation tests across 64 electrodes).
- **CPP:** single-trial **CPP amplitude scales with sensory uncertainty**, and the **CPP build-up
  slope links to the drift rate** (steeper CPP → higher `v`), tested inside HSSM-embedded LMMs.

### Key methods (EEG + DDM)
- **DDM → HSSM.** Unlike the two papers (which used the older HDDM), this study uses the **HSSM**
  package — the same one in our `e_HSSM_module.py`. It estimates Bayesian posteriors of `v`, `z`, `a`
  per congruency × uncertainty cell, with **trait covariates** (intolerance of uncertainty IU-18,
  delusion proneness PDI, hallucination proneness CAPS) and a **per-subject random intercept**.
- **CPP** (centro-parietal positivity): a response-locked ERP over a centro-parietal cluster, taken
  as a *neural read-out of evidence accumulation* — its build-up slope is the EEG analogue of drift.
- **SpecParam (FOOOF):** separates the **1/f-like aperiodic** background from oscillatory peaks in
  the 0–500 ms post-stimulus PSD. Crucially, **IAF here is defined as the SpecParam Gaussian peak
  centre frequency in 7–14 Hz** — a *spectral-peak* definition, **not** Romei & Tarasi's Hilbert
  instantaneous-frequency definition. Worth flagging when comparing the two.
- **Other:** Morlet time-frequency (3–60 Hz), **sample entropy** (signal irregularity), and the
  questionnaires above.

### Exploratory analyses
The **congruency × uncertainty interaction on drift rate** (graded prediction error → graded drift
modulation), CPP × uncertainty on drift, IAF × drift (the explicit Romei & Tarasi link), trait
associations with drift, confidence effects on `z`/`a`/CPP, 3D time-frequency cluster tests, and the
aperiodic × uncertainty interaction.

### What changed from the earlier draft
Skimming the earlier draft (`Preregistration_NaturalUncertainty.pdf`) against the final shows the
study was *tightened*, not redesigned:
- **The congruency × uncertainty interaction on drift was demoted** from a confirmatory prediction
  to an **exploratory** one — the lab judged the prior evidence too weak to pre-commit to a direction.
- **A choice-frequency analysis was added** as a confirmatory test (distractor classifications rising
  with incongruency × uncertainty).
- The confirmatory HSSM hypothesis was broadened from "drift" to **"bias-related parameters (drift
  *and* start point)"**, echoing Franzen's origin+gain split.
- Housekeeping: Jonathan's inline review comments and the explicit LMM formulas were removed, and a
  few specifics (the exact CPP electrode list, the SampEn software package) were dropped from the
  committed text.

### Why it matters for our pipeline
This preregistration **is the spec** the pipeline is being generalised toward. The HSSM hardening,
the per-parameter formulas, the alpha→drift covariate, the SpecParam/FOOOF aperiodic step, and the
recreated posterior plots all exist so that when the bear/dog data arrives, the modelling and figures
are ready. The 3-subject RDK demo is the dress rehearsal.

---

## How the three fit together

| Source | Core idea it contributes | The figure / mechanism we took from it |
|--------|--------------------------|----------------------------------------|
| **Romei & Tarasi (2026)** | **Alpha → drift coupling.** Faster alpha = faster, cleaner evidence accumulation (`v`), not bias. Justifies our `v ~ alpha` covariate. | **Fig 4C** coefficient histograms (dashed line at 0) → our `hssm_posterior_coefficients`. |
| **Franzen et al. (2025)** | **Prior = origin vs gain.** A prior can bias the **start point `z`** (before evidence) or the **drift `v`/`dc`** (during) — a mechanistic dissociation. Justifies fitting separate `z` and `v` models. | **Fig 5** stacked posterior densities → our `hssm_posterior_ridgeline`. |
| **Natural Uncertainty prereg** | **The study these feed.** A 2×3 cue-congruency × uncertainty design that asks *both* questions at once — does a congruent cue move `z`, `v`, or both? — with CPP and 1/f EEG read-outs. | The whole design; defines IAF via SpecParam peak (7–14 Hz). |

The throughline: **Franzen tells us *where* a prior can act in the decision (origin `z` vs gain
`v`); Romei tells us that a specific EEG signal (alpha speed) plugs into the *gain* side (`v`); and
the preregistration combines both into one experiment** — manipulating the prior (cue congruency) and
the evidence quality (uncertainty) together while recording the EEG signals (CPP, aperiodic, alpha)
that the first two papers say should matter. Our job is the plumbing that makes all of that
reproducible — which is what the 3-subject demo has been quietly verifying.

**See also:** [./01-meeting-jonathan.md](./01-meeting-jonathan.md) (Jonathan's framing of the
origin-vs-gain question and the `z` ← low-level prior anchor) and [./05-plots.md](./05-plots.md) (the
recreated Romei-style and Franzen-style figures, with images embedded).
