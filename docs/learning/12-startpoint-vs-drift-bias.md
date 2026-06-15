# 12 — Start-point bias vs drift bias (per-parameter formulas)

*Teaching note for the `z`-formula change in `e_HSSM_module.py` — letting each DDM
parameter take its own regression, and using it to ask **how** a prior biases a
decision: by moving the starting line, or by tilting the evidence? Builds on
[[09-alpha-in-drift]] and [[11-eeg-predicts-drift-rate]].*

---

## What I changed

| File | Change |
|------|--------|
| `e_HSSM_module.py` | `fit_hssm_hierarchical` now takes optional `formula_z/t/a` and builds the `include` list from whichever are non-empty |
| `inputs.json` | added `formula_z = "z ~ 1 + prior_dic + (1\|participant)"` (t/a left empty) |

Before, only the drift rate `v` had a formula; `z`, `t`, `a` were fixed group
parameters. Now any of them can be modelled — the PI's request to "model starting
biases with a formula."

---

## The CS concept — config-driven assembly instead of a hard-coded list

The old code hard-coded `include=[{"name":"v", ...}]`. The new code *builds* that
list from configuration:

```python
include = [{"name": "v", "formula": formula_v, "link": "identity"}]
for name, f in (("z", formula_z), ("t", formula_t), ("a", formula_a)):
    if f:                      # empty string => leave that parameter simple
        include.append({"name": name, "formula": f})
```

This is the same idea as the pipeline's other dispatch points (the `match` savers,
the `perform` toggles): **behaviour is data, not code.** Adding a non-decision-time
model later is a one-line edit to `inputs.json`, not a code change. The empty-string
default keeps every parameter that *isn't* configured as a plain group estimate, so
the change is backward-compatible (a v-only run behaves exactly as before).

One subtlety: `prep_hssm_data` has to see **all** the formulas joined together, so it
knows which covariates to keep (it only drops trials missing an alpha term that some
formula actually uses). I pass `' '.join([formula_v, formula_z, ...])` for that.

---

## The psych/neuro concept — two ways a prior can bias a decision

The DDM is a race: evidence accumulates from a **starting point** until it hits a
boundary. An expectation ("the dots will probably go left") can tip that race two
different ways:

```
z — START-POINT bias:  begin the race already closer to "left"
                        (a head start, BEFORE any evidence)

v — DRIFT bias:         read the incoming evidence as more "left-ish"
                        than it is (a tilt DURING accumulation)
```

They look identical in the choices, but leave different signatures in the
reaction-time distribution — so the DDM can separate them. That's the whole point of
modelling `z` *and* `v`: not just "did the prior bias the decision?" (we knew that)
but **through which mechanism?**

- `z ~ 1 + prior_dic` puts the prior on the starting point.
- the intercept of `v` (the drift criterion) carries any drift bias.

Because our boundary is coded prior-congruent (+1) vs incongruent (−1), a positive
`z_prior_dic` literally means "when a prior is present, the start point moves toward
the prior-expected response."

---

## Why it helped / what improved — a clean dissociation

Full `v + z` fit on the 3-subject pilot (1000 draws):

| Term | Meaning | Estimate (94% HDI) | Credible? |
|------|---------|--------------------|-----------|
| `z_prior_dic[prior]` | prior → start-point shift | **+0.348 [0.20, 0.51]** | **yes** |
| `v_Intercept` | drift bias (drift criterion) | −0.061 [−0.32, 0.27] | no |
| `v_alpha_cf_cog_wc` | alpha → drift | +0.068 [−0.04, 0.17] | no |

The reading: **in this pilot the prior biases decisions by shifting the starting
point, not by tilting the drift.** That's a real, credible mechanistic result —
credible even at n = 3, because a start-point bias from an explicit cue is a strong
effect (unlike the subtle trial-level alpha→drift effect, which needs the full
sample). The alpha→drift coefficient is unchanged from the v-only fits (+0.068 ≈
+0.061), so adding the `z` model didn't disturb the drift story.

This is exactly the kind of question the DDM exists to answer, and it's the first
step of the PI's bigger plan (next: `t` and `a` formulas, then neural and
transdiagnostic covariates on each parameter).

---

## Key things to know for the PI quiz

- **What's new:** each DDM parameter (v/z/t/a) can take its own Bambi-style formula;
  configured in `inputs.json`, assembled into HSSM's `include` list at fit time.
- **z vs v bias:** `z` = head start *before* evidence; `v`-intercept = tilt *during*
  accumulation. The RT-distribution shape distinguishes them.
- **The result:** prior gives a **credible start-point bias** (+0.35) but **no
  credible drift bias** — the prior moves *where* accumulation starts.
- **Why credible at n=3 when alpha wasn't:** an explicit-cue start-point bias is a
  large effect; trial-level alpha→drift is small and needs more subjects.
- **Backward compatible:** empty formula = that parameter stays a simple group
  estimate, so v-only runs are unchanged.
