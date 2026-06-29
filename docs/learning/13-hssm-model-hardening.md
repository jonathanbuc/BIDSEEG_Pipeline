# 13 — Hardening the HSSM model: a fixed non-decision time, a hierarchical boundary, and reading the library's source

## What I changed
- **Non-decision time `t` is now fixed** to a constant `0.2 s` instead of being estimated.
  Config: `inputs.json → Analysis.hssm.fix_t = 0.2`. Code: `e_HSSM_module.py` now passes
  `{"name": "t", "prior": 0.2}` into HSSM's `include` list. (I also deleted a stale
  comment, `# changes formula_t to 0.2 s`, that claimed this was already done — it wasn't.)
- **Boundary separation `a` now varies by participant**: `formula_a = "a ~ 1 + (1|participant)"`
  (it used to be a single shared scalar).
- **The exact model-input matrix is saved** to `hssm_input_data.csv`.
- Two side questions got answered by *reading the installed package source* rather than
  guessing: how HSSM handles missing trials, and whether it returns per-subject estimates.

## The CS concept — a fixed parameter is constant-folding, not a tight prior
HSSM (via bambi) treats a **scalar** `prior` value as a literal constant: it becomes a
bambi `ConstantComponent`, no random variable is created, and nothing is sampled. That is
fundamentally different from giving `t` a *very tight prior*, which still creates a sampled
variable. The proof is visible in the output — after the change, `t` simply **disappears
from the posterior table**, because there is no RV to summarize. This is the modelling
equivalent of constant-folding: you remove a degree of freedom from the inference graph
entirely, rather than nudging it.

The second CS lesson: for "how does this library actually behave?" questions, the
authoritative source is the installed package code, not documentation or memory. Reading
`hssm`/`bambi`/`formulae` directly showed that a `NaN` in a covariate makes HSSM **error**
(bambi's default `na_action="error"`), not silently drop the row. So the pipeline's
existing pre-filter — drop trials missing an alpha covariate, computed on the *union* of
every formula's covariates — isn't just acceptable, it's required for the model to run.

## The psych/neuro concept — non-decision time vs decision time, and caution
A drift-diffusion model splits a reaction time into **non-decision time `t`** (sensory
encoding + motor execution) and the **decision** itself (evidence accumulating at rate `v`
from a start point `z` to a boundary `a`). This task enforces a ~500 ms window in which the
participant *cannot* respond. That latency is structural, not decision-related — so letting
the model freely estimate `t` lets it absorb variance that isn't decision time, biasing the
parameters we care about. Fixing `t` at 0.2 s pins it to the known floor and protects
`v`/`a`/`z`.

Boundary separation `a` is how much evidence is required before committing — the
speed/caution trade-off. Letting it vary by participant (a random intercept) acknowledges
that some people are simply more cautious, and hierarchical partial pooling shrinks noisy
per-subject estimates toward the group mean — important at n=3.

## Why it helped
Before: `t` was a free parameter quietly soaking up the response-lockout artifact, `a` was
one value forced on every subject, and the model input was invisible. After: `t` is pinned
at 0.2 s (and *gone* from the posterior, which confirms it's a true constant and not a
sampled RV), `a` is hierarchical with per-subject offsets (`a_1|participant[sub-001..003]`),
and the model-input matrix is saved for inspection. The re-fit converged cleanly
(`r_hat ≈ 1.00`). The changes match the 6/15 meeting decisions and make the model both more
correct and more inspectable.
