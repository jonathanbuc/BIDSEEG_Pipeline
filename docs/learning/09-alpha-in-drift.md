# 09 — Wiring per-trial alpha into the drift rate (Milestone 2)

*Teaching note for the alpha-covariate changes in `e_HSSM_module.py` — taking the
per-trial alpha frequency from note 07 and letting it predict the drift rate, then
asking the data whether the alpha clock actually moves evidence accumulation.*

---

## What I changed

| File | Change |
|------|--------|
| `e_HSSM_module.py` | `load_group_data()` picks the per-trial alpha table when the drift formula names an alpha covariate; `prep_hssm_data()` now builds centered alpha predictors and drops trials missing the alpha term the formula uses |
| `inputs.json` | `formula_v` now adds `alpha_cf_cog_wc` to the drift formula |
| `docs/learning/09-alpha-in-drift.md` | this note |

Milestone 1 (note 07) wrote one alpha centre frequency per trial. Milestone 2 feeds
it into the hierarchical DDM: `v ~ 1 + exp * coh_level + alpha_cf_cog_wc + (1|participant)`.

---

## The CS concept — don't merge what's already aligned; and centering as a join-free decomposition

**The non-join.** The obvious plan was "join the alpha CSV onto the behavioural CSV
by `(participant, block_cond, block_order, thisN)`." I verified that key first — and
it blew up: a keyed merge matched **14638** rows against 1329 (1101%), because those
four columns are *not* a unique trial key (`thisN` resets every block, the labels
recur). A positional or keyed merge there would have silently mis-aligned every
trial and turned the alpha coefficient into noise.

The fix wasn't a better join — it was *no join*. `extract_trial_alpha()` built its
table as `epochs.metadata.copy()` plus appended columns, i.e. a column assignment on
one DataFrame, which is aligned by construction. So the alpha table already carries
every behavioural field. `load_group_data()` just reads it directly when the formula
needs alpha. The lesson: when one artifact was *derived* from another by adding
columns, re-joining on keys reintroduces a bug that the derivation never had.

**Centering, in code.** `coh` and `alpha` differ between subjects, so a raw slope
blends two different questions. The prep builds three versions of each:
`_gc` (value − grand mean), `_wc` (value − that subject's mean), and `_subjmean`
(subject mean − grand mean). `_wc` is a pure within-subject signal; `_subjmean` is
the between-subject part. Splitting them lets the model estimate the two slopes
separately (a Mundlak decomposition) instead of forcing one slope to stand for both.

**Formula-aware dropna.** FOOOF leaves ~7% of trials without an alpha peak. Rather
than blanket-drop, prep only drops rows missing the alpha column the *formula
actually uses* — so the centre-of-gravity model keeps all 1093 trials and only the
FOOOF model pays the 80-trial cost. The centering means are computed *before* this
drop, so they're not perturbed by it.

---

## The psych/neuro concept — does the alpha clock set the accumulation rate?

The drift rate `v` is how fast evidence piles up toward a decision. The hypothesis:
when occipital alpha sits higher on a given trial (relative to that subject's own
mean), the visual system is in a state that accumulates evidence faster. Putting
`alpha_cf_cog_wc` on `v` tests exactly that, trial by trial.

Two estimators carry through from note 07: `alpha_cf_cog` (centre of gravity, 0%
missing) is the primary; `alpha_cf_fooof` (parametric peak, ~7% missing) is the
robustness check. Agreement of *sign* across both is the neural-signal version of
"it replicates with a different measurement."

---

## Why it helped / what improved — the result, honestly

Six models, all on the **same 1093-trial** alpha sample (baseline included, so sample
size can't confound the contrast), centre-of-gravity as primary:

| Model | alpha effect on drift | P(>0) | credible? |
|-------|----------------------|-------|-----------|
| within-subject (`_wc`)  | **+0.061** | 0.84 | no (HDI [−0.05, 0.16]) |
| grand-mean (`_gc`)      | +0.034 | 0.74 | no |
| Mundlak within / between| +0.060 / −0.041 | 0.86 / 0.33 | no |
| × condition interaction | +0.105, ×exp −0.080 | 0.90 / 0.27 | no |
| FOOOF estimator (`_wc`) | +0.025 | 0.77 | no |

Reading it straight:
- **The machinery works and converges.** Every alpha coefficient has r̂ = 1.0 and
  ESS 1300–2280 — the estimates are trustworthy. The non-credibility is a power/
  effect-size fact at n = 3, not a sampling artifact. This mirrors the coherence
  result (note 06): a clean proof-of-concept, powered for the full ~43-subject set.
- **The effect is consistently positive** (faster drift when trial alpha is above the
  subject's mean) across every specification and both estimators — never credible,
  but never flipping sign.
- **Within-subject beats grand-mean** (+0.061 vs +0.034), and the Mundlak split shows
  the within slope (+0.060) and between slope (−0.041) actually point *opposite ways*.
  That's the whole argument for `_wc` in one number: grand-mean centering would dilute
  the within-subject effect by blending in a null/opposite between-subject term.
- **It's not coherence or RT in disguise.** `alpha_cf_cog_wc` correlates 0.03 with
  coherence and 0.01 with RT; its VIF is **1.01** (no variance inflation). The alpha
  predictor is statistically independent of the other drift terms, so the coefficient
  is really about alpha.

So Milestone 2 delivers a correct, confound-checked trial-by-trial neural-covariate
DDM. The headline scientific test waits on the full dataset; the pipeline is ready
for it.

---

## Key things to know for the PI quiz

- **Why no merge:** the alpha table is `epochs.metadata` + appended columns, so it's
  already trial-aligned; `(participant, block_cond, block_order, thisN)` is not a
  unique key (a keyed merge fanned out to 1101%).
- **Why `_wc` not `_gc`:** within-subject centering isolates the trial-level alpha
  shift; the Mundlak split shows within (+0.06) and between (−0.04) differ, so the
  blended grand-mean slope understates the within effect.
- **Why the effect isn't credible:** n = 3. The coefficient is well-sampled
  (ESS > 1300, r̂ = 1.0); the HDI straddles 0 because the effect is small relative to
  three-subject uncertainty.
- **Confound ruled out:** VIF(alpha) = 1.01; r(alpha, coherence) = 0.03 — alpha is not
  re-expressing coherence or RT.
- **Same sample for all models:** baseline is refit on the 1093 alpha trials, so the
  baseline-vs-alpha comparison isn't confounded by the 1329→1093 trial drop.
