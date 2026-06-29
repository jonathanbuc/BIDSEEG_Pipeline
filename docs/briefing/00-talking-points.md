# Talking points for the meeting with Jonathan

Use this live. Order roughly: **what I did → what it shows → what I need from you → logistics.**
Reminder to say once up front: *all results are from the 3-subject demo — plumbing/validation,
not findings ("don't interpret below 10 observations").*

---

## 1. Everything you asked for is done ✅

**From the meeting:**
- **Fixed non-decision time `t` to 0.2 s** — and it's a *true constant*, not a tight prior, so `t`
  no longer appears in the posterior at all (that's the proof it took). Reason: the ~500 ms
  response lockout isn't decision time.
- **Boundary `a` is now hierarchical** (`a ~ 1 + (1|participant)`) — per-subject offsets show up
  where there used to be one shared value. (Checked: with no formula, HSSM defaults `a` to a single
  `HalfNormal(2)` scalar with *no* random effects — so the formula was needed.)
- **Saved the model input** to `hssm_input_data.csv` — I can show you exactly what the sampler sees.
- **Re-fit converged**, `r_hat ≈ 1.00`.

**From your emails:**
- **`absl-py` added to the `.yml`** (it was installed but undeclared — that's why it broke for you).
- **`hssm_trace` plot is now readable** (the old one overlapped all the labels).
- **Recreated the Romei & Tarasi Fig 4C plot** from the colleague's code — coefficient-posterior
  histograms with a dashed line at 0.
- **Added condition-comparison posterior plots** (Franzen-style ridgelines) **and** a cleaned-up
  version of the diffusion-schematic from that code. All functions live in `plotting_module.py`,
  as you asked, so the analysis modules stay clean.

## 2. Results worth mentioning (n=3 — directional only)

- **The demo reproduces your pattern in direction:** drift `v` is modulated by **both** prior
  levels (`v_exp[lowlevel]` credibly positive, q=0.002; `v_exp[highlevel]` q≈0.04), while the start
  point `z` shifts **more for the low-level prior** than high-level (`z_exp[lowlevel]` ≈ −0.046 vs
  `z_exp[highlevel]` ≈ −0.014). So: **`z` ← low-level, `v` ← both** — exactly your prior finding,
  just not credible at n=3 (both `z` HDIs cross 0, as expected).
- **Alpha → drift** is positive but not credible at n=3 (`v_alpha` ≈ +0.04, q≈0.23) — right
  direction (Romei), underpowered.

## 3. Decisions I need from you

1. **How should the start point handle prior *level*?** I found the old model keyed `z` on
   "prior present vs absent," which **collapsed high vs low** — so it couldn't answer your question.
   I switched it to `z ~ exp` (gives separate low/high coefficients) as a quick test. **Do you want
   that single-model contrast (Option A), or the "separate experiments per prior block" approach you
   mentioned (Option B) for the real data?**
2. **Which alpha measure on drift?** The model currently uses the alpha *centre-of-gravity*, but the
   prereg defines IAF as the *FOOOF peak* frequency. **Switch the drift formula to the FOOOF peak,
   or keep CoG and report the peak as a robustness check?**
3. **Restricting the FOOOF window** to the prediction window (cue → stimulus) — you said this is a
   one-line change for the real data. Confirm you want it once the full dataset lands.

## 4. Questions you raised — now answered (from the HSSM source)

- **Missing trials:** HSSM would *error* on a NaN covariate, not drop it — so we must pre-filter.
  We already drop the ~8% no-alpha trials (no interpolation), which is the correct/required behavior.
- **Subject-level estimates:** yes — already produced via `(1|participant)` (per-subject offsets in
  the output). Note it's a random *intercept*; per-subject *slopes* would need `(1+exp|participant)`.
- **Fixing `t`:** a scalar value = a true constant (no sampled variable); a distribution would be a
  tight prior (still sampled). We used the constant.

## 5. Housekeeping / heads-up

- My branch's `environment_setup.yml` has drifted from `main` (env name, `hssm` version, a stray
  `pingouin` line on main) — I'll reconcile that when we merge; nothing broken now.
- All of the above is committed-ready but **not yet pushed** — tell me if you want it on the branch.

## 6. Logistics & next steps to confirm

- **Dataset:** any update on the ~40-subject upload? The pipeline's ready to run on it as soon as
  it's there.
- **July 1, 4 pm CET CFC/HGF meeting** — confirm you still want me to join (I'll wait for your go +
  the Zoom link).
- **Publications / HGF reanalysis** — did your chat with Guido land anywhere? Happy to start scoping
  the HGF reanalysis once the HSSM work is signed off.

---

*Deep-dive backup if he asks for detail: `01-meeting-jonathan.md`, `02-paper-summaries.md`,
`03-model-changes.md`, `04-hssm-questions.md`, `05-plots.md` in this folder.*
