# 01 — What Jonathan wanted: the 6/15 meeting + his two follow-up emails

> **Who / what / when.** A working session on **15 June 2026** between **Aniket** (CS
> student / research assistant) and **Jonathan Buchholz** (PhD candidate and Aniket's
> de-facto supervisor in the Hesselmann lab, Berlin). The meeting room was named after
> **Prof. Guido Hesselmann** (the PI), but **Guido did not attend** — it was just Aniket
> and Jonathan, in English. Two short follow-up **emails from Jonathan on 17 June** are
> summarised at the end.
>
> **Stage of the project.** We had just reached the `e_HSSM_module.py` milestone:
> feeding **trial-by-trial individual alpha frequency (IAF)** into a drift-diffusion model
> as a covariate on the drift rate. Everything ran on the **3-subject demo dataset** (the
> Hierarchical-Priors RDK task).
>
> **The single most important caveat.** All numbers here come from **3 subjects**. They
> are *plumbing and validation* — proof the machinery runs — **not scientific findings**.
> Jonathan's own rule, said out loud in the meeting: *"don't interpret data below 10
> observations."* You can find a "significant" correlation in 3 people that is pure
> artifact. Treat every result below as "the code works," never as "the brain does X."

This briefing is part of a set. See also:
- `./02-paper-summaries.md` — summaries of the papers behind this work (Romei & Tarasi, Franzen) and the Natural Uncertainty preregistration.
- `./03-model-changes.md` — the concrete model edits (fixed `t`, the `a` formula, etc.).
- `./04-hssm-questions.md` — the open "how does HSSM behave?" questions and their answers.
- `./05-plots.md` — the new posterior-plotting functions Jonathan asked for.

---

## A 60-second primer on the model (so the rest reads cleanly)

A **drift-diffusion model (DDM)** explains a single decision (and its reaction time) as
**noisy evidence piling up over time until it hits one of two boundaries** (e.g. "left
motion" vs "right motion"). It has four core parameters:

| Symbol | Name | Plain meaning |
|--------|------|---------------|
| `v` | **drift rate** | *How fast and in which direction* evidence accumulates. Higher `v` = stronger/clearer evidence = faster, more accurate. |
| `a` | **boundary separation** | *How much* evidence you demand before committing. Wide `a` = cautious/slow/accurate; narrow `a` = hasty. |
| `z` | **start point** (bias) | *Where accumulation begins* between the two boundaries. Off-centre `z` = a head start toward one answer *before any evidence* — a prior expectation. |
| `t` | **non-decision time** | Time spent **not** deciding: sensory encoding + motor response. RT = `t` + decision time. |

**HSSM** (Hierarchical Sequential Sampling Models) is the Python library that fits this.
"**Hierarchical**" = it fits **all subjects at once** in one model, simultaneously
estimating a group-level effect *and* per-subject deviations, instead of fitting each
person separately. Because it is **Bayesian**, it doesn't return a single number per
parameter — it returns a **posterior**: a whole probability distribution of plausible
values given the data. The **HDI** (highest-density interval) is the Bayesian credible
range; **if the HDI for an effect includes 0, the effect is not credible** (we can't rule
out "no effect").

**IAF (individual alpha frequency)** is the peak frequency of a person's alpha rhythm
(~7–14 Hz EEG oscillation), which varies trial-to-trial. The scientific bet we are wiring
up: **alpha frequency modulates the drift rate** — i.e. a brain-state measure predicts how
efficiently evidence is accumulated. See `../learning/11-eeg-predicts-drift-rate.md`.

---

## What Jonathan wanted (the asks)

Jonathan's overarching message was: *"this is the right way to do it; now make it correct
and inspectable, and find out how the library actually behaves."* Concretely he wanted:

1. **Fix non-decision time, don't fit it.** The task design forces a window where no
   response is possible, so that latency is *not* decision time and shouldn't be estimated.
2. **Add a boundary-separation (`a`) formula** — at minimum a per-participant random
   intercept — and check whether HSSM already does this by default.
3. **Test on the low-level prior specifically**, because it changes trial-by-trial (like
   the cues in his real upcoming experiment), rather than lumping all priors together.
4. **Understand the library, not guess it:** how does HSSM treat missing (NaN) covariate
   values? Does it return *subject-level* posteriors or only group-level? Does it output a
   single regression weight (β) or a separate posterior per condition?
5. **Make the work visible:** save the exact dataframe fed to HSSM as a CSV, and improve
   the posterior plots (he reinforced this by email — see below).
6. **Be careful with the alpha extraction:** keep FOOOF with a center-of-gravity (COG)
   fallback and a "peak found" boolean; sanity-check the alpha values (range, variance,
   multiple peaks) against the expected ~7–14 Hz / mean ~10 Hz.

---

## Decisions from the meeting

These were settled in the call (mostly Jonathan deciding, Aniket confirming):

1. **Fix `t` to a constant (~0.2 s).** The paradigm presents the stimulus for ~500 ms
   during which the participant *cannot answer* (they respond on the next screen). That
   ~500 ms is structural lockout, not deliberation. So `t` is pinned, not estimated. (See
   `../learning/13-hssm-model-hardening.md` for how this was implemented and why a fixed
   scalar is a *constant*, not a tight prior.)
2. **Add an `a` formula** with at least a participant random intercept; also verify the
   no-formula default.
3. **Use the low-level prior for functionality testing** — it varies trial-by-trial and
   therefore mirrors the trial-wise cues in the real (Natural Uncertainty) experiment.
4. **Keep FOOOF for IAF extraction**, with the **COG fallback** (a power-weighted mean
   frequency in the alpha band, used when FOOOF finds no credible peak) and a
   **`peak_found` boolean** flag. Jonathan: keep COG "for now, maybe we'll use it later."
5. **Fit ONE hierarchical model across all subjects**, not one model per subject — that's
   the whole point of *hierarchical* DDM (better reliability via partial pooling).
6. **For missing IAF (~8% of trials): drop the whole trial. Do NOT interpolate.** Jonathan
   was explicit: *"interpolating, I wouldn't interpolate."* An 8% trial loss is normal for
   human EEG. (Caveat noted for later: with a bigger formula carrying several trial-wise
   covariates, dropping any trial missing *any* covariate could cost a lot of data — worth
   watching.)
7. **Treat the three prior blocks (baseline / low-level / high-level) as separate
   experiments.** The priors are *not overlaid* — no single trial has both a low- and a
   high-level prior — so you **fit a separate HSSM per block** and then **compare the
   parameter posteriors across blocks within subjects.**

---

## Action items (checklist)

All assigned to Aniket; all originated from Jonathan.

- [ ] **Fix `t` as a constant** (try 0.2 s), then **re-run and report.**
- [ ] **Add the `a` formula** (participant random intercept) and **check the no-formula
      default** (does HSSM add a random intercept on its own?).
- [ ] **Find out how HSSM handles missing / NaN values** trial-by-trial — does it drop the
      whole trial or just skip that parameter's contribution?
- [ ] **Determine whether HSSM returns subject-level posteriors**, not only group-level.
- [ ] **Clarify exactly what HSSM outputs** — a regression β (association weight) vs a
      separate posterior per condition.
- [ ] **Save `df_hssm`** (the model-input dataframe) to CSV to show Jonathan next time.
- [ ] **Confirm WHICH prior produced the credible start-point (`z`) effect** — Aniket
      guessed *high-level* but must verify — then **re-run the start-point analysis split by
      the low-level prior.**
- [ ] **Handle multiple alpha peaks per trial** (currently averaged) and **inspect their
      range/variance** — should sit ~7–14 Hz with mean ~10 Hz.
- [ ] **Research better IAF-extraction methods** (lower priority; FOOOF is fine for now).
- [ ] **On real data: restrict the FOOOF/alpha window to the prediction window**
      (cue → stimulus), not the whole epoch.

**Logistics action items:**
- [ ] **Join the CFC/HGF help-group meeting** — moved to **1 July 2026, 4 pm CET** (Aniket's
      10 am). It's a **closed PhD/postdoc meeting**: *ask Jonathan before each join.*
- [ ] **Read the Nature Communications inspiration paper** — but *after exams.*

Several of these were answered in follow-up work: see `./04-hssm-questions.md` and
`../learning/13-hssm-model-hardening.md` (NaN handling → HSSM errors on NaN, so the
pre-filter is required; per-subject posteriors → yes) and `../learning/14-hssm-posterior-plots.md`
(what the output looks like).

---

## The science discussed

Remember: **n = 3.** These are directions of trends in plumbing tests, nothing more.

**IAF → drift rate (`v`).** The effect was **consistently positive** — higher alpha
frequency tended to go with higher drift rate, across most trials — **but the HDI included
0**, so it is **not credible** at this sample size. (In the later plots this shows up as a
"right direction, not credible" coefficient; see `../learning/14-hssm-posterior-plots.md`.)

**Coherence → drift rate.** Behaved exactly as expected: **higher motion coherence →
higher drift rate.** Coherence is the fraction of dots moving coherently in the RDK task —
literally the strength of the sensory evidence — so this is the basic sanity check that the
model is wired correctly.

**A credible start-point (`z`) shift** appeared with the **congruent prior** condition.
Aniket thought it came from the **high-level** prior but flagged it as **unconfirmed** —
hence the action item to verify and then re-split by the low-level prior.

### ⭐ The key anchor — Jonathan's prior finding (memorise this)

> In Jonathan's own (larger, real) analyses the pattern is **specific**:
>
> - **Start point `z` is modulated by the LOW-level prior** (he does *not* find a `z`
>   modulation from the high-level prior).
> - **Drift rate `v` is modulated by BOTH the low- and high-level priors.**

This matters because Aniket's demo result hinted at a `z` shift from the *high-level*
prior — the **opposite** of Jonathan's anchor. If that held up on real data it would be
genuinely interesting; with n = 3 it's far more likely noise. This is exactly why Jonathan
wants the analysis **re-run split by the low-level prior**: it's the prior that actually
changes trial-by-trial and the one his theory ties to the start point.

**Why the `z` vs `v` distinction is conceptually deep:** a prior can bias a decision in two
different ways. It can move the **start point** (`z`) — a head start toward the expected
answer *before any evidence arrives* — or it can change the **drift rate** (`v`) — biasing
*how the incoming evidence itself is interpreted*. Same behavioural bias, different
mechanism. Disentangling them is much of the scientific payoff. See
`../learning/12-startpoint-vs-drift-bias.md`.

### Formula-design decisions

- **Coherence enters as an interaction with the experiment block:** `coh * exp`. (We expect
  the coherence effect to differ across blocks.)
- **IAF enters additively on drift, with NO interaction.** Jonathan endorsed this: there's
  no reason individual alpha frequency should vary *with* coherence, so an additive term is
  the honest model. The only effect this formula investigates for alpha is the
  alpha → drift-rate association.
- **Centering of continuous covariates:** subject-mean / grand-mean centering (each
  continuous covariate is centered per subject before fitting — a normalization so the
  intercept and slopes are interpretable and subjects are comparable).

---

## The two emails (Jonathan, 17 June 2026)

**Email 1 — 6:28 AM (a missing dependency).** Jonathan ran the code and hit a missing
`absl-py` dependency. His instruction, quoted:

> *"There seems to be a dependency on the absl-py package. When you encounter new
> dependencies, just add them to the .yml file. Might also be OS thing. Some packages are
> pre-installed in windows / Mac."*

**Takeaway:** keep `environment_setup.yml` complete and reproducible — don't rely on
packages that happen to be pre-installed on one OS. (CS concept: pin your dependencies so
the environment is the same everywhere.)

**Email 2 — 6:53 AM (it works; now make the plots better).** Quoted in full:

> *"I pushed the updated .yml file to main. Other than that everything worked perfectly
> smooth! You could also update the hssm_trace plot, so that the variables are readable. If
> you find other cool ways of plotting the posterior distributions between conditions,
> including more plots is always helpful. Best to add the functions to the plotting_module,
> so that the other modules stay clean. I also attached code from a guy I met at the
> conference. The plot is not really pretty, neither is the code. But I like the idea. It
> is similar to the plot in Romei & Tarasi (2026). So maybe you could recreate that plot
> with the code I send you. Very cool work. Btw I will chat with Guido today about
> publications."*

**Concrete asks from Email 2:**
- Jonathan **fixed the dependency himself** and **pushed the updated `.yml` to `main`.**
- **Make the `hssm_trace` plot readable** (the variable labels overlapped).
- **Add more ways to plot posterior distributions between conditions** — more plots help.
- **Put all plotting functions in `plotting_module.py`** so the analysis modules stay clean.
- **Recreate the Romei & Tarasi (2026)-style plot** from the attached conference code
  (`DDM_Plots_functions.py`) — he likes the *idea*, not the (admittedly rough) execution.

**Attachments:** Franzen et al. (2025), Romei & Tarasi (2026), and `DDM_Plots_functions.py`.
These drove the plotting work — see `./05-plots.md` and
`../learning/14-hssm-posterior-plots.md` (the new functions: a readable `hssm_trace`, a
Romei & Tarasi Fig 4C coefficient-histogram, a Franzen ridgeline, and a corrected DDM
schematic).

---

## Publication & logistics

**A full dataset is coming.** Jonathan will upload **~40 more subjects** (so ~43 total) in
the **same BIDS format with behavioural data embedded** — a drop-in for the existing
pipeline, just more subjects. He'd coordinated with Guido on a host site and planned to
upload "soon." This is what unlocks actual interpretation; until then everything is n = 3.

**Authorship.** Jonathan was warm but realistic:
- **Quick publications are "not our style"** — the lab aims for thorough, high-quality work
  in good journals, which takes time (his current paper took ~2.5 years).
- **A publication within a 3-month internship is unlikely / not feasible.**
- **But he's open to shared-authorship work**, most plausibly an **HGF (hierarchical
  Gaussian filter) reanalysis** of the hierarchical-priors data — something he's wanted to
  do (he did a DDM on it before but never an HGF). This would come **after** the HSSM work
  is solid and merged. A reanalysis is faster than a fresh study (data already collected).
- He said he'd **chat with Guido about publications** (and repeated this in Email 2).

**The HGF connection.** This is why joining the **CFC/HGF help group** (with the ETH Zürich
people) matters — it feeds the possible HGF reanalysis. Meeting is **1 July 2026, 4 pm CET**;
it's closed (PhDs/postdocs), so check with Jonathan before each join.

**Immediate sequencing.** Priority is to get the **HSSM analysis ready and merged into
`main`**; once that's comfortable, the HGF work can begin in parallel, with a publication in
mind.

---

## Glossary

| Term | Definition |
|------|-----------|
| **DDM (drift-diffusion model)** | A model of two-choice decisions: noisy evidence accumulates over time until it crosses one of two boundaries; the crossing point and time give the choice and reaction time. |
| **HSSM** | Hierarchical Sequential Sampling Models — the Python library used here to fit Bayesian, hierarchical DDMs (fits all subjects jointly). |
| **HDDM** | The earlier-generation hierarchical DDM library; HSSM is its successor. |
| **Drift rate (`v`)** | Speed/direction of evidence accumulation; reflects evidence quality. Higher = faster, more accurate. |
| **Boundary separation (`a`)** | How much evidence is required before committing; the speed-vs-caution trade-off. |
| **Start point (`z`)** | Where accumulation begins between the boundaries; off-centre = a prior bias / head start before evidence. In HSSM it's *relative* (0–1), so the absolute height is `z·a`. |
| **Non-decision time (`t`)** | RT not spent deciding (sensory encoding + motor response). Here fixed at ~0.2 s rather than estimated. |
| **Hierarchical model** | One model fit to all subjects at once, estimating group-level effects and per-subject deviations together; "partial pooling" shrinks noisy per-subject estimates toward the group. |
| **Posterior** | In Bayesian inference, the full probability distribution of a parameter given the data (not a single point estimate). |
| **HDI (highest-density interval)** | The Bayesian credible interval; the narrowest range holding a given probability mass (e.g. 94%). If it includes 0, the effect isn't credible. |
| **Random intercept** | A per-group (here per-participant) baseline offset in a mixed model — lets each subject have their own baseline level of a parameter. |
| **Covariate** | A trial-level predictor entered into the formula (e.g. IAF, coherence). |
| **IAF (individual alpha frequency)** | The peak frequency of a person's alpha EEG rhythm (~7–14 Hz, mean ~10); varies trial-to-trial; our covariate on drift rate. |
| **FOOOF / specparam** | An algorithm that separates a power spectrum into **aperiodic** (1/f background) and **periodic** (oscillatory peaks) parts, then fits Gaussians to the peaks — used to extract the alpha peak per trial. |
| **COG (center of gravity)** | A power-weighted mean frequency within the alpha band (after removing the 1/f background); a fallback IAF estimate when FOOOF finds no credible peak. |
| **Coherence (`coh`)** | In the random-dot-kinematogram (RDK) task, the fraction of dots moving coherently — the strength of the sensory evidence. |
| **Prior block (baseline / low-level / high-level)** | The three experimental conditions; treated as separate experiments since their priors are never overlaid on the same trial. |
| **Grand-mean / subject-mean centering** | Subtracting a covariate's mean (per subject) before fitting, so coefficients are interpretable and subjects comparable. |
| **Regression weight (β)** | A single number summarising how strongly two variables are associated in a (mixed) linear model — one of the two output forms HSSM can give. |
| **HGF (hierarchical Gaussian filter)** | A Bayesian model of how an agent updates beliefs under uncertainty; the candidate method for the possible reanalysis paper. |
| **CPP (central-parietal positivity)** | An EEG signal often treated as a neural correlate of evidence accumulation (linked to drift rate in the inspiration paper). |
