# 04 — The HSSM "how does it actually behave?" questions

*Companion to [01-meeting-jonathan.md](./01-meeting-jonathan.md) (where these questions came up) and
[03-model-changes.md](./03-model-changes.md) (the code changes that answer some of them).*

> **Caveat, up front.** Every number quoted here is from the **3-subject demo** dataset
> (the Hierarchical-Priors RDK task). These are **plumbing / validation** results — they tell us the
> machine runs and is wired correctly, **not** what is true about brains. Don't interpret effects
> below ~10 observations. The real test is the future "Natural Uncertainty" dataset.

---

## Why this is its own document

Six of the meeting items were not "what model should we fit?" questions — they were
**"how does the library behave in this situation?"** questions:

- What does HSSM do with a missing (NaN) covariate on a trial?
- Can it give me per-subject estimates, or only the group?
- If I "fix" non-decision time, is that the same as a very tight prior?
- If I give boundary `a` no formula, what does it silently default to?

You cannot answer these reliably from memory, blog posts, or even the official docs — docs lag the
code and often describe intent rather than what the installed version literally does.

### The CS lesson: the installed source is ground truth

HSSM is a thin orchestration layer that delegates the actual model-building to **bambi** (a
regression-formula front-end for PyMC) which in turn delegates formula parsing to **formulae**, and
the sampling to **PyMC** / **NumPyro**. A behaviour you observe is the *composition* of all those
layers. The only authoritative description of that composition is the code that is actually imported
at runtime. So the right move — and what we did — is to **read the package source in the active
conda env**:

```
C:\Users\Aniket\miniconda3\envs\mne-env\lib\site-packages\
    hssm\        # the DDM wrapper
    bambi\       # formula -> PyMC model
    formulae\    # formula string -> design matrices
```

Each answer below ends with the exact file (and where useful, the line) that proves it. That
traceability is the point: when Guido or Jonathan asks "are you sure?", the answer is a file path,
not a recollection.

A quick glossary, since the answers lean on it:

| Term | Plain meaning |
|---|---|
| **Drift rate `v`** | how fast (and which way) evidence piles up; the "signal strength" of the decision |
| **Boundary separation `a`** | how much evidence is needed before committing — the speed/caution trade-off |
| **Start point `z`** | where accumulation begins between the two boundaries (0–1); a *pre-evidence* bias |
| **Non-decision time `t`** | the part of the reaction time that is *not* deciding (sensory encoding + motor) |
| **Posterior** | the full probability distribution of a parameter *after* seeing the data (Bayesian output) |
| **HDI** | highest-density interval — the Bayesian "where the parameter credibly lives" interval (we report 94%) |
| **Random intercept `(1\|participant)`** | each subject gets their own offset from the group mean for that parameter |
| **Covariate** | a per-trial predictor fed into a parameter's formula (here: condition, coherence, trial alpha) |

---

## Q1 — How does HSSM handle missing / NaN values trial-by-trial?

### How it arose
The drift model uses trial-wise covariates: condition, coherence, and the per-trial individual alpha
frequency (IAF). On these short (~1.5 s) epochs, FOOOF finds **no alpha peak on roughly 7–8% of
trials**, so `alpha_cf_fooof` is `NaN` there. Jonathan's exact worry: when a trial has a missing
covariate, does HSSM **drop only that covariate's contribution** for that trial, or does it **drop
the whole trial** — and either way, does it do it *silently*?

### The answer
**HSSM does neither silently — its native default is to ERROR.** There is no quiet row-dropping and
no quiet covariate-zeroing. Two independent guards fire:

1. **A NaN in a *covariate* is delegated to bambi**, whose `na_action` defaults to `"error"`. In the
   installed source:
   ```python
   # bambi/models.py
   na_action = "drop" if dropna else "error"     # dropna defaults to False
   ```
   With the default `dropna=False`, formulae then refuses to build the design matrix and raises a
   "data contains N incomplete rows" error. A single NaN covariate would **abort the entire fit**.

2. **A NaN in the *response* (`rt`)** is caught even earlier, by HSSM's own validator, before bambi
   is ever reached:
   ```python
   # hssm/data_validator.py
   if np.any(rt_filtered.isna(), axis=None):
       raise ValueError("You have NaN response times in your dataset, ...")
   ```

So the library will *not* limp along on partial data — it stops. That means **the pipeline must
pre-filter**, and it does. `prep_hssm_data` in `e_HSSM_module.py` drops trials missing any alpha
covariate **that the formula actually uses**, computed on the **union of all four formulas**
(`v`, `z`, `t`, `a`) *before* HSSM sees the data:

```python
# e_HSSM_module.py — fit_hssm_hierarchical
all_formulas = ' '.join([formula_v, formula_z, formula_t, formula_a])
df_hssm = prep_hssm_data(df, cond_col, conditions, formula=all_formulas)

# e_HSSM_module.py — prep_hssm_data
need = [c for c in df.columns if c.startswith('alpha_cf_') and c in formula]
if need:
    df = df.dropna(subset=need).reset_index(drop=True)
```

Two deliberate choices here:

- **We do NOT rely on bambi's `dropna=True`.** bambi builds a **separate design matrix per
  parameter** (`v` vs `z` vs `t` vs `a`). If we let each one drop NaN rows independently, the
  matrices could end up with **different row counts** and misalign the trials — a silent corruption
  far worse than an error. Dropping once, up front, on the union keeps every parameter's matrix on
  the same rows.
- **We do NOT interpolate** the missing IAF (a meeting decision). Inventing an alpha value would
  fabricate the very neural signal we're testing.

### Why it matters / what we did
This converts a potential *silent* bug (mismatched matrices, or a fabricated covariate) into either
a clean error or a logged, explicit drop. The number of dropped trials is written to the log, and
the exact matrix the sampler sees is saved to `hssm_input_data.csv` for inspection. Because the
centering (`_wc`, `_gc`) is computed *before* the drop, those means use all available trials and are
unaffected by it.

**Source:** `bambi/models.py` (`na_action` default); `formulae` design-matrix "incomplete rows"
logic; `hssm/data_validator.py` (NaN-`rt` check); `e_HSSM_module.py` (`prep_hssm_data`, formula
union).

---

## Q2 — Can HSSM return SUBJECT-LEVEL posteriors, not just group-level?

### How it arose
Aniket wanted **per-participant** estimates, not just one group number. The reference point was a
bachelor student's old **HDDM** (the predecessor library) code, which returned both group- and
subject-level values. Does the newer HSSM still give individual estimates?

### The answer
**Yes — and the pipeline already produces them.** The hierarchical term `(1|participant)` is a
**random intercept**: it tells the model to give each subject their own offset from the group mean.
After the fit, the ArviZ `InferenceData` contains, for each parameter with that term, three kinds of
variable. Using `v` (drift) as the example, straight from the real `hssm_posterior_summary.csv`:

| Variable in the trace | What it is |
|---|---|
| `v_Intercept` | the **population** drift (group mean) |
| `v_1\|participant[sub-001 … sub-003]` | each subject's **scaled offset** (deviation from the group mean) |
| `v_1\|participant_offset[sub-001 …]` | the **raw, standardized** version of that offset (pre-scaling) |
| `v_1\|participant_sigma` | the **between-subject SD** — how much subjects vary |

A single subject's value is reconstructed as **population intercept + that subject's offset**:

```
v(sub-002)  ≈  v_Intercept + v_1|participant[sub-002]
            =  -0.181 + (-0.007)  =  -0.188
```

Two practical rules:

- **Use `v_1|participant`, not `v_1|participant_offset`.** The plain name is the offset already
  scaled by `sigma` (i.e. `offset × sigma`), which is the deviation in the parameter's own units.
  The `_offset` variable is the raw standardized draw bambi samples internally — useful for
  diagnostics, not for reading a subject's value.
- **Apply the inverse link for bounded parameters.** `z` is fit on a logit scale and `a` on a log
  scale, so to read them in natural units you invert the link (`expit` for `z`, `exp` for `a`).
  Drift `v` uses the **identity** link in our model, so **no transform is needed for `v`**.

**Two caveats, both important at n=3:**

1. `(1|participant)` is a random **intercept only** — it shifts each subject's baseline up or down,
   but the **slopes (condition/alpha effects) are shared** across subjects. If you want
   *per-subject* condition effects, you need a random **slope** too:
   `v ~ 1 + exp + (1 + exp | participant)`. We have not done that (3 subjects can't support it).
2. With only 3 subjects, the per-subject estimates are **heavily shrunk** toward the group mean
   (partial pooling) and have **wide HDIs**. That shrinkage is a feature — it stops one noisy
   subject from dominating — but it means the individual numbers are not yet interpretable.

### Why it matters / what we did
The thing Aniket asked for already exists in the output; no code change was needed, only knowing
*which rows* to read and *which transform* to apply. The subject rows
(`v|`, `a|`, `z|` per `sub-00x`) are already in `hssm_posterior_summary.csv`.

**Source:** `hssm/utils.py` (`_get_alias_dict`, which builds the readable `*_1|participant[...]`
names); bambi's group-specific-term coordinates; and the real `hssm_posterior_summary.csv`, which
already lists these rows.

---

## Q3 — How do you FIX non-decision time `t` to a constant — and how is that different from a tight prior?

### How it arose
A meeting decision: **fix `t`**. The task imposes a ~500 ms **response-lockout** window in which the
participant *cannot* respond, so that latency is structural, not decision-related. Letting the model
freely estimate `t` lets it absorb non-decision variance and bias the parameters we care about.

### The answer
The two options are **fundamentally different mechanisms**, not two strengths of the same thing.

**Fixing `t` (what we did):** pass a *scalar float* as the parameter's `prior`. HSSM detects this and
flags the parameter `is_fixed`:

```python
# hssm/param/param.py
@property
def is_fixed(self) -> bool:
    """Whether the parameter is fixed as a scalar or a vector."""
    return isinstance(self.prior, (int, float, np.ndarray))
```

bambi then builds a **`ConstantComponent`** whose value is the literal number. **No PyMC random
variable is created, and nothing is sampled.** The consequence you can see with your own eyes:
**`t` is completely ABSENT from the posterior table** — there's no RV to summarize. (Check
`hssm_posterior_summary.csv`: it has `v_*`, `a_*`, `z_*` rows and **no `t` row**.) This is the
modelling equivalent of **constant-folding**: you remove a degree of freedom from the inference
graph entirely.

**A tight prior (the thing it is NOT):** pass a *distribution* — a dict or `bmb.Prior`, e.g.
`Normal(mu=0.2, sigma=0.001)`. That **does** create a sampled random variable. `t` would still be
*estimated*, just heavily constrained, and it would **still appear in the trace** with a posterior.

In short:

| | Mechanism | RV created? | In the posterior? |
|---|---|---|---|
| **Fixed** (`prior = 0.2`) | bambi `ConstantComponent` | **No** | **No** — disappears |
| **Tight prior** (`Normal(0.2, 0.001)`) | sampled, constrained | **Yes** | **Yes** — narrow distribution |

We used `"fix_t": 0.2` in `inputs.json`. In code:

```python
# e_HSSM_module.py
if fix_t is not None:
    include.append({"name": "t", "prior": float(fix_t)})   # scalar -> fixed constant
```

### Why it matters / what we did
Pinning `t` at 0.2 s removes a free parameter that would otherwise quietly soak up the
response-lockout artifact, protecting `v`, `a`, and `z`. And we **verified** it worked the only
unambiguous way: `t` is gone from the posterior trace, proving it is a true constant and not a
sampled-but-narrow variable. (A tight prior would have *looked* fixed in the point estimate but
still carried — and reported — uncertainty.)

**Source:** `hssm/param/param.py` (`is_fixed`); bambi's `backend`/`model_components` (a float prior
becomes a constant output with no RV); `e_HSSM_module.py` (`fix_t` branch).

---

## Q4 — When NO formula is given for boundary `a`, what does HSSM default to?

### How it arose
A meeting item with two parts: "**add an `a`-formula**" and "**check whether the no-formula case
already defaults to a random effect**" — i.e. is HSSM *secretly* giving us per-subject boundaries
even when we don't ask?

### The answer
**No hidden random effect.** With no formula, `a` is a **single group-level scalar** — one value
shared across all subjects and all trials. Specifically:

- **Prior:** `HalfNormal(sigma=2.0)`. This is not hard-coded for `a` in the model config (only `t`
  has an explicit default prior there); `a` gets its prior from `_make_default_prior`, driven by its
  bounds:
  ```python
  # hssm/likelihoods/analytical.py
  ddm_bounds = {"v": (-inf, inf), "a": (0.0, inf), "z": (0.0, 1.0), "t": (0.0, inf)}

  # hssm/param/utils.py — _make_default_prior
  elif not np.isinf(lower) and np.isinf(upper):
      if lower == 0:
          prior = bmb.Prior("HalfNormal", sigma=2.0)   # <- this branch for a
  ```
  Because `a`'s bounds are `(0, inf)` with `lower == 0`, the default is exactly `HalfNormal(2.0)`.
- **Bounds:** `(0, inf)` — `a` must be positive.
- **No link, no random effects** — just one scalar.

Two things this is **NOT**:

- It is **not** the HDDM-style `Gamma` prior. That "safe" prior only applies to **regression /
  formula** parameters; a plain group scalar uses the bounds-derived default above.
- It is **not** per-subject. To get a per-subject boundary you must add the formula explicitly,
  which we did:
  ```json
  "formula_a": "a ~ 1 + (1|participant)"
  ```
  After that change, the posterior gains `a_Intercept` plus `a_1|participant[sub-001…003]` offsets
  and an `a_1|participant_sigma` (all visible in `hssm_posterior_summary.csv`).

### Why it matters / what we did
This settled the ambiguity: the no-formula default was a single shared boundary, so subjects who are
simply more cautious had no way to express that. Adding `a ~ 1 + (1|participant)` lets each subject
have their own boundary, and hierarchical partial pooling shrinks the noisy per-subject estimates
toward the group mean — the right behaviour at n=3.

**Source:** `hssm/modelconfig/ddm_config.py` (default config); `hssm/param/utils.py`
(`_make_default_prior` → `HalfNormal(2.0)`); `hssm/likelihoods/analytical.py` (`ddm_bounds`,
`a = (0, inf)`); "safe"/regression-only prior logic in `hssm/param/params.py`.

---

## Q5 — Is our IAF (individual alpha frequency) extraction adequate, and how should multiple alpha peaks be handled?

### How it arose
At the meeting: should we keep FOOOF for per-trial alpha? How do we handle a trial that shows
**more than one** alpha peak? And is the resulting frequency in a sane range?

A note on terms: **FOOOF** (now packaged as **SpecParam**, `fooof == 1.1.1`) decomposes a power
spectrum into a **1/f aperiodic background** plus a set of **Gaussian "bumps"** (oscillatory peaks).
**IAF / Peak Alpha Frequency** is the centre frequency of the alpha-band bump — a per-person (here,
per-trial) marker linked to attention / cortical inhibition.

### The answer
**Keep it.** `extract_trial_alpha` in `c_EEGAnalysis_module.py` fits a FOOOF/SpecParam model **per
trial** and takes the **centre frequency of the strongest in-band Gaussian peak** as the IAF. The
alpha band is **7–14 Hz** (`alpha_freq_range`). This *is* the preregistration's definition of IAF.

- **Multiple peaks are resolved by MAX POWER** — the standard Peak Alpha Frequency convention:
  ```python
  # c_EEGAnalysis_module.py — extract_trial_alpha
  in_alpha = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
  if len(in_alpha):
      cf_f, pw_f = in_alpha[np.argmax(in_alpha[:, 1])][:2]   # strongest peak
  ```
  This is defensible. Do **not** average the centre frequencies of multiple peaks (that invents a
  frequency nothing oscillated at), and do **not** snap to 10 Hz (that erases the individual
  differences we're trying to measure).

- **There is always a fallback.** Alongside the parametric peak (`alpha_cf_fooof`, which is `NaN`
  when no peak is found), the function also computes `alpha_cf_cog` — a **centre of gravity**: the
  power-weighted mean frequency in the band *after* subtracting the fitted 1/f background. It is
  **always defined**. A boolean `alpha_peak_found` flags which trials had a real parametric peak.

- **Range is correct by construction** (7–14 Hz), and the pilot mean is ~10.5 Hz — physiologically
  sensible.

**Minor, optional follow-ups (flagged, not yet applied):**

- The default drift formula in `inputs.json` currently uses the **COG** estimate
  (`alpha_cf_cog_wc`), whereas the prereg specifies the **peak** (`alpha_cf_fooof_wc`). COG is the
  more robust choice (no NaNs, so no trials dropped), but it is a **modelling decision** to flag to
  Jonathan, since it departs from the registered IAF definition. (This is also why Q1's NaN handling
  matters: switching to the peak covariate is what triggers the ~7–8% drop.)
- A latent crash-guard around the per-trial FOOOF fit, and a small band-edge consistency fix.

### Why it matters / what we did
The extraction matches the prereg, uses the standard max-power rule for multiple peaks, and ships a
robust always-defined fallback plus a coverage flag — so it's adequate as-is. The one thing the PI
should consciously decide is **peak vs COG** as the drift covariate.

**Source:** `c_EEGAnalysis_module.py` (`extract_trial_alpha`); `docs/learning/07-trial-alpha-extraction.md`;
the Natural Uncertainty preregistration.

---

## Q6 — Which prior drove the start-point (`z`) effect?

### How it arose
At n=3 a credible-looking **start-point shift** appeared in an earlier fit (see
`docs/learning/12`). The task has **two kinds of prior** — a **low-level** and a **high-level**
prior — and Jonathan's standing hypothesis is that the **start point is driven by the low-level
prior** (`z ← low-level`), while drift is driven by both. Aniket needed to confirm *which* prior the
demo's `z` effect was actually attributable to.

### The answer
**The original `z` formula could not answer the question — by construction.** It keyed on
`prior_dic`, a binary "prior **present** vs **absent**" variable. That **collapsed high- vs
low-level prior into one bucket**, so the model structurally *could not* attribute `z` to a specific
*level*. Any `z` effect it found was "a prior moved the start point", with no way to say which.

**The fix (Option A):** re-key `z` on the same condition factor `exp` that `v` uses, so `z` gets a
coefficient per level:

```json
"formula_z": "z ~ 1 + exp + (1|participant)"
```

This produces `z_exp[lowlevel]` and `z_exp[highlevel]` — separate start-point shifts for each prior
level. The demo result (n=3, **not credible**, direction only):

| Term | Estimate | Reading |
|---|---|---|
| `z_exp[lowlevel]` | **−0.046** | larger start-point shift |
| `z_exp[highlevel]` | **−0.014** | ~3× smaller |
| `v_exp[lowlevel]` | **+0.329** | drift modulated… |
| `v_exp[highlevel]` | **+0.192** | …by **both** levels |

So the low-level prior moves the start point ~3× more than the high-level prior — the **direction**
of Jonathan's prediction — while drift `v` responds to **both** levels. The demo therefore
*directionally* reproduces the **"`z` ← low-level, `v` ← both"** pattern. (All HDIs straddle 0 at
n=3; this is plumbing confirming the model *can* express the dissociation, not evidence that it's
real.)

**Option B (for the real data):** the meeting's alternative is to split the analysis into
**separate experiments per prior block** rather than putting both levels in one formula — cleaner
when there's enough data per block. That's a choice for the real dataset, written up in
[03-model-changes.md](./03-model-changes.md).

### Why it matters / what we did
This was the one HSSM question that required a **real model change**, not just a source lookup: the
old formula was structurally incapable of separating the two priors, so it could have produced a
misleading "the prior shifts `z`" story without ever saying which prior. Re-keying `z` on `exp` makes
the model able to attribute the effect to a level — necessary before the result means anything on the
full dataset.

**Source:** `e_HSSM_module.py` (`formula_z`); `inputs.json` (`Analysis.hssm.formula_z`); the real
`hssm_posterior_summary.csv` (the `z_exp[*]` / `v_exp[*]` rows quoted above);
`docs/learning/12-startpoint-vs-drift-bias.md`.

---

## Status at a glance

| # | Question | Answer in a phrase | Status |
|---|---|---|---|
| Q1 | NaN covariate handling | HSSM/bambi **error by default** (no silent drop); pipeline pre-filters on the formula union | **Done** |
| Q2 | Subject-level posteriors? | **Yes** — read `*_1\|participant[sub-xx]` (+ intercept), apply inverse link; already in the summary | **Done** |
| Q3 | Fix `t` vs tight prior | Scalar prior → `ConstantComponent`, **no RV, absent from trace**; tight prior still samples | **Done** |
| Q4 | No-formula `a` default | Single shared scalar, `HalfNormal(2.0)`, bounds (0, ∞), **no random effect**; we added the formula | **Done** |
| Q5 | IAF extraction adequate? | **Keep it** — strongest in-band FOOOF peak (max power), COG fallback | **Done** — *decision needed: peak vs COG covariate* |
| Q6 | Which prior drives `z`? | Old `prior_dic` collapsed levels; re-keyed `z ~ exp` → low-level shift ~3× high-level (n=3, directional) | **Done** — *Option B (per-block) open for real data* |
