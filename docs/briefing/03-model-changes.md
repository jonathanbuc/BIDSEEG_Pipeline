# 03 — HSSM model changes this session

This note documents the changes made to the hierarchical drift-diffusion model
(HSSM) this session: **what** changed, the **exact code and where to find it**, and
a plain-English **explanation** of each change. It is written so you can both
understand the work and brief Jonathan and Guido from it.

> **Read this first — the numbers are plumbing, not science.** Everything below was
> run on the **3-subject DEMO dataset** (the Hierarchical-Priors RDK task). With
> ~3 observations at the group level, no posterior estimate is statistically
> credible. The point of this work was to validate the *pipeline* — that the model
> is specified correctly, samples cleanly, and produces the right outputs — **not**
> to draw scientific conclusions. Do not interpret group effects below ~10
> observations.

---

## Background: what an HSSM / DDM is in three lines

A **drift-diffusion model (DDM)** explains a single choice + reaction time as noisy
evidence accumulating over time until it hits one of two boundaries. It has four
core parameters:

| Param | Name | What it means (plain terms) |
|-------|------|------------------------------|
| `v` | **drift rate** | How fast (and toward which boundary) evidence accumulates — the *quality/direction* of the evidence. |
| `a` | **boundary separation** | How much evidence is needed before committing — the *speed vs. caution* trade-off. |
| `z` | **start point** | Where accumulation begins between the two boundaries — a *pre-evidence bias* toward one response. |
| `t` | **non-decision time** | Time spent on sensory encoding + motor execution, *not* on deciding. |

**HSSM** (Hierarchical Sequential Sampling Models) fits this as a **hierarchical
Bayesian** model: instead of one number per parameter, it estimates a full
**posterior** (a probability distribution over plausible values given the data), and
the *hierarchical* part lets each subject have their own offset while sharing
information across subjects (see "random intercept" below). We sample the posterior
with **NUTS** (the No-U-Turn Sampler, a Markov-chain Monte Carlo method) via the
`numpyro` backend.

All four changes live in two files:
- `e_HSSM_module.py` — the model code.
- `inputs.json` → `Analysis.hssm` — the config block (lines 194–209).

The config block as it stands now:

```jsonc
// inputs.json, lines 194-209
"hssm": {
    "model_type": "ddm",
    "sampler": "nuts_numpyro",
    "draws": 1000,
    "tune": 1000,
    "chains": 2,
    "cores": 2,
    "target_accept": 0.9,
    "prior_settings": "safe",
    "link_settings": "log_logit",
    "formula_v": "v ~ 1 + exp * coh_level + alpha_cf_cog_wc + (1|participant)",
    "formula_z": "z ~ 1 + exp + (1|participant)",
    "formula_t": "",
    "fix_t": 0.2,
    "formula_a": "a ~ 1 + (1|participant)"
}
```

---

## CHANGE 1 — Fix non-decision time `t` to a constant (0.2 s)

### What changed
`t` used to be a **free parameter** the sampler estimated. It is now **pinned to a
literal constant, 0.2 s**, so the sampler does not touch it at all. This is driven
by a new config key `fix_t` and a small rewrite of how the model's parameter list is
assembled.

### Exact code & where

**Config** — `inputs.json`, line 207 (alongside `formula_t` on 206 and `formula_a` on 208):

```jsonc
// inputs.json
"formula_t": "",        // line 206
"fix_t": 0.2,           // line 207  <-- new key
"formula_a": "a ~ 1 + (1|participant)"   // line 208
```

**Load-time read** — `e_HSSM_module.py`, line 58 (`None` means "estimate `t`
normally"):

```python
# e_HSSM_module.py, line 58
fix_t          = hssm_cfg.get('fix_t', None)
```

**Function signature** — `e_HSSM_module.py`, line 132 gained `fix_t=None`:

```python
# e_HSSM_module.py, line 132
def fit_hssm_hierarchical(df, condition_dict, formula_v, model_type,
                           prior_settings, link_settings,
                           draws, tune, chains, cores, target_accept, sampler,
                           formula_z='', formula_t='', formula_a='', fix_t=None):
```

**Call site** — `e_HSSM_module.py`, line 267 passes it through:

```python
# e_HSSM_module.py, lines 264-268
model = fit_hssm_hierarchical(
    df_group, condition_dict, formula_v, model_type,
    prior_settings, link_settings, draws, tune, chains, cores, target_accept, sampler,
    formula_z=formula_z, formula_t=formula_t, formula_a=formula_a, fix_t=fix_t
)
```

**The rewritten `include` build** — `e_HSSM_module.py`, lines 156–166. This replaced
an older loop over `z`/`t`/`a` that only handled non-empty *formula* strings and
carried a dead comment (`# changes formula_t to 0.2 s`) that did nothing:

```python
# e_HSSM_module.py, lines 156-166
include = [{"name": "v", "formula": formula_v, "link": "identity"}]
# Fix t to a constant when requested: HSSM treats a scalar `prior` as a fixed value
# (a bambi ConstantComponent -- no sampled RV), unlike a `formula`. A fixed t will
# therefore not appear in the posterior trace at all.
if fix_t is not None:
    include.append({"name": "t", "prior": float(fix_t)})
elif formula_t:
    include.append({"name": "t", "formula": formula_t})
for name, f in (("z", formula_z), ("a", formula_a)):
    if f:
        include.append({"name": name, "formula": f})
```

**Generic log message** — `e_HSSM_module.py`, lines 169–171. A spec for a fixed `t`
has a `"prior"` key but **no** `"formula"` key, so the old `spec["formula"]` access
would crash. The line now falls back gracefully:

```python
# e_HSSM_module.py, lines 169-171
formula_msg = " | ".join(
    spec.get("formula", f'{spec["name"]}={spec.get("prior")}') for spec in include
)
```

### How the code works
The whole model is described by the `include` list — one dict per parameter. `v`
always gets a regression `formula`. The new branch decides `t`'s fate:

- `fix_t is not None` → append `{"name": "t", "prior": float(fix_t)}`. The key insight:
  **HSSM (via the `bambi` library) interprets a scalar passed as `"prior"` as a true
  fixed constant.** Internally it builds a `ConstantComponent` — **no random variable
  is created, and nothing is sampled** for `t`. This is the modelling equivalent of
  *constant-folding*: the degree of freedom is removed from the inference graph
  entirely.
- This is **different from giving `t` a very tight prior** (a narrow *distribution*).
  A distribution — however narrow — is still a random variable that gets sampled.
  A scalar is not.

`z` and `a` are still handled by the loop on lines 164–166: each gets a `formula`
entry only if its config string is non-empty.

### Why it helps (the neuro reason)
A DDM splits a reaction time into non-decision time `t` (encoding + motor) and the
decision proper (evidence accumulating at rate `v` from start `z` to boundary `a`).
The RDK task **forces a ~500 ms window in which the participant physically cannot
respond.** That latency is structural, not part of the decision. If you let the model
freely estimate `t`, it will happily soak up that lockout variance — and because the
parameters trade off against each other, a mis-estimated `t` **biases the `v`, `a`,
and `z` we actually care about.** Pinning `t` at 0.2 s anchors it to the known floor
and protects the decision parameters.

### How it was verified
After the change, **`t` does not appear anywhere in the posterior summary**
(`hssm_posterior_summary.csv`). There is literally no variable to summarize — exactly
what you expect from a true constant (as opposed to a tight prior, which *would* still
show up). See `./04-hssm-questions.md` for the package-source detail.

---

## CHANGE 2 — Make boundary separation `a` hierarchical

### What changed
`a` was a single shared scalar forced on every subject. It is now **hierarchical**:
each participant gets their own boundary offset, partially pooled toward the group.

### Exact code & where

**Config only** — `inputs.json`, line 208. `formula_a` went from `""` to a formula:

```jsonc
// inputs.json, line 208
"formula_a": "a ~ 1 + (1|participant)"
```

**No `e_HSSM_module.py` code change was needed.** The existing loop already turns any
non-empty `formula_a` into an `include` entry:

```python
# e_HSSM_module.py, lines 164-166
for name, f in (("z", formula_z), ("a", formula_a)):
    if f:
        include.append({"name": name, "formula": f})
```

And `link_settings="log_logit"` (`inputs.json`, line 203) auto-applies the correct
**log link** to `a` so the positive boundary stays positive.

### How the code works
Read the formula `a ~ 1 + (1|participant)` in two parts:

- `1` — a global **intercept**: one group-level mean boundary.
- `(1|participant)` — a **random intercept**: a per-subject deviation from that
  group mean, where the deviations are themselves drawn from a shared distribution
  whose spread the model also estimates.

This is **partial pooling**. A *fully pooled* model would force one `a` on everyone
(ignores individual differences). A *no-pooling* model would fit each subject
independently (noisy with few trials). Partial pooling is the middle ground: each
subject gets their own estimate, but noisy per-subject estimates are **shrunk toward
the group mean** — exactly the behaviour you want at n=3.

The `log_logit` link setting means `a` is estimated on a log scale (guaranteeing
positivity) and back-transformed for interpretation; `z` would similarly get a logit
link to stay in [0, 1]. You do not have to specify the link per parameter — HSSM
applies the appropriate one automatically.

### Why it helps (the neuro reason)
Boundary separation `a` is the **speed/caution trade-off**: how much evidence a person
demands before committing to a response. People genuinely differ in caution, so
forcing one boundary on everyone is misspecified. A random intercept acknowledges
that, while partial pooling keeps the small-sample estimates stable.

### How it was verified
The posterior summary now contains `a_Intercept`, three per-subject terms
`a_1|participant[sub-001 .. sub-003]`, and the group spread `a_1|participant_sigma`
— where before there was a single scalar `a`.

---

## CHANGE 3 — Save the model-input dataframe

### What changed
The exact recoded / centered / NaN-dropped matrix that the sampler actually sees is
now written to disk as `hssm_input_data.csv`.

### Exact code & where
`e_HSSM_module.py`, lines 148–152, right after `df_hssm = prep_hssm_data(...)`:

```python
# e_HSSM_module.py, lines 146-152
df_hssm = prep_hssm_data(df, cond_col, conditions, formula=all_formulas)

# Persist the exact matrix the sampler sees (recoded, centered, NaN-dropped) so the
# model input is inspectable without re-deriving it.
input_path = os.path.join(result_dir, 'hssm_input_data.csv')
df_hssm.to_csv(input_path, index=False)
utils.log_msg(f'        HSSM input data: {input_path} ({len(df_hssm)} rows)')
```

### How the code works
`prep_hssm_data` (lines 67–126) does a lot of silent transformation before the model
sees the data: it drops RT-flagged outliers, recodes `response_prior` to the
±1 boundary encoding HSSM expects, maps `coh_level` low/medium → 0/1, builds
grand-mean- and subject-mean-centered versions of coherence and individual alpha
frequency, filters to the conditions of interest, and drops trials missing any alpha
covariate the formula uses. The new three lines simply dump the result to CSV before
sampling — no transformation, just a snapshot.

### Why it helps
It makes the model input **inspectable without re-deriving it.** You (or Jonathan) can
open `hssm_input_data.csv` and see exactly which trials survived, how covariates were
coded, and what was centered — instead of mentally re-running `prep_hssm_data`. This is
basic experimental provenance: the thing you analysed is on disk next to the results.

### How it was verified
The run log prints
`HSSM input data: ...hssm_input_data.csv (1093 rows)` — confirming the file is written
and how many trials the model saw on the demo dataset.

---

## FOLLOW-UP (Option A) — making the start point answer "which prior *level*"

This addresses open question #6 ("confirm which prior drives `z`"). Jonathan's anchor
hypothesis is **`z` ← low-level prior, `v` ← both levels**: the low-level prior should
shift the *start point* (a pre-evidence bias), while both prior levels should modulate
the *drift rate*.

### What changed
The start-point formula was re-keyed from the binary prior indicator to the
three-level experimental factor.

`inputs.json`, line 206:

```jsonc
// before:  "formula_z": "z ~ 1 + prior_dic + (1|participant)"
// after  (line 206):
"formula_z": "z ~ 1 + exp + (1|participant)"
```

And the `z` ridgeline plot was updated to run over the `exp` factor (`cond_col`)
instead of `prior_dic` — `e_HSSM_module.py`, lines 294–298:

```python
# e_HSSM_module.py, lines 294-298
if 'z_Intercept' in idata.posterior.data_vars:
    # z now shares the exp factor (formula_z = z ~ 1 + exp), so the start-point
    # ridgeline runs over the same conditions as v; logit link back to [0,1].
    plotting.hssm_posterior_ridgeline(idata, 'z', cond_col,
                                      condition_dict[cond_col], result_dir, link='logit')
```

### Why it matters
`prior_dic` encodes prior **present vs. absent** — it collapses high-level and
low-level priors into one bucket, so a model keyed on it **cannot say which *level***
moved the start point. The `exp` factor has three levels (`base` / `lowlevel` /
`highlevel`), so keying `z` on `exp` produces separate coefficients
`z_exp[lowlevel]` and `z_exp[highlevel]` — the model can now distinguish them. This
mirrors how `v` is already specified (its `formula_v` uses `exp`), so `z` and `v` run
over the same conditions and the ridgeline plots line up.

### Result (n=3 demo — directional only, NOT credible)
On the demo data:

| Coefficient | Value | Reading |
|-------------|-------|---------|
| `z_exp[lowlevel]`  | **-0.046** | low-level prior start-point shift |
| `z_exp[highlevel]` | **-0.014** | high-level prior start-point shift |

The low-level prior shows a **~3× larger start-point shift** than the high-level
prior — the *direction* of Jonathan's "`z` ← low-level" prediction. Meanwhile `v` is
modulated by **both** levels (`v_exp[lowlevel]` is credibly positive). So the demo
reproduces the **"`z` ← low-level, `v` ← both"** pattern *directionally*.

**Caveat, restated:** at n=3 these are validation numbers. The pattern shows the model
*can express* the hypothesis and the plumbing works end-to-end; it is not evidence for
the hypothesis. See `./04-hssm-questions.md`.

---

## How to run it

```bash
conda activate mne-env
python e_HSSM_module.py inputs.json
```

The whole module is gated by `perform.compute_hssm` in `inputs.json` (currently
`true`). The fit is a **hierarchical Bayesian DDM** sampled with **NUTS via the
`numpyro` backend** (`sampler: "nuts_numpyro"`). On the 3-subject demo it takes
**~3 minutes** and converges cleanly (**`r_hat ≈ 1.00`** — `r_hat`, the Gelman-Rubin
statistic, compares variance within vs. across chains; values near 1.00 indicate the
chains agree and the sampler has converged).

### Where the outputs land (`results/groupBehavioral/`)
- `hssm_input_data.csv` — the exact model-input matrix (Change 3).
- `hssm_posterior_summary.csv` — per-parameter posterior summary (note: **no `t` row** — Change 1).
- `hssm_idata.nc` — the full `InferenceData`, so plots can be regenerated without refitting.
- Posterior plots — trace, Romei & Tarasi Fig 4C coefficient histograms, Franzen Fig 5
  ridgelines for `v` and `z`, and the DDM-anatomy schematic (which receives the fixed
  `t` via `t_value=fix_t`).

---

## One-line summary for the PI

We hardened the HSSM model: **fixed `t` at 0.2 s** (it now drops out of the posterior,
proving it is a true constant), made **boundary `a` hierarchical** (per-subject random
intercepts with partial pooling), **saved the model-input matrix** for provenance, and
**re-keyed `z` onto the three-level `exp` factor** so it can say *which* prior level
moves the start point. On the 3-subject demo the model samples cleanly (`r_hat ≈ 1.00`)
and reproduces the "`z` ← low-level, `v` ← both" pattern *directionally* — pipeline
validation, not a result.
