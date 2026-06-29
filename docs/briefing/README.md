# Session briefing — HSSM hardening, plots, and the Natural Uncertainty context

These notes were written to help me (Aniket) **understand** the work done in the late-June 2026
session and **brief Jonathan/Guido** on it. They're reference docs, not the numbered teaching
notes in `../learning/` (those explain one change each; these give the wider picture).

> **Read me first, then go in order.** Everything here about *results* is from the **3-subject
> demo** dataset (the Hierarchical-Priors RDK task) — it's plumbing/validation, **not** scientific
> conclusions. The real study is the future "Natural Uncertainty" bear/dog dataset.

## The docs

**Going into the meeting? → [00-talking-points.md](./00-talking-points.md)** — the condensed "what
to say to Jonathan" sheet (status, results, decisions needed, logistics). The rest are deep-dive backup.

| # | File | What it covers |
|---|------|----------------|
| 1 | [01-meeting-jonathan.md](./01-meeting-jonathan.md) | What Jonathan wanted — his concerns, the 6/15 meeting decisions & action items, and his two follow-up emails |
| 2 | [02-paper-summaries.md](./02-paper-summaries.md) | Plain-language summaries of Romei & Tarasi (2026), Franzen et al. (2025), and the Natural Uncertainty preregistration — and why each matters |
| 3 | [03-model-changes.md](./03-model-changes.md) | The three HSSM model changes (fix `t`, hierarchical `a`, save `df_hssm`) + the Option-A `z` follow-up — exact code, where it lives, and how it works |
| 4 | [04-hssm-questions.md](./04-hssm-questions.md) | The "how does HSSM behave?" questions — how each arose at the meeting and the verified answer |
| 5 | [05-plots.md](./05-plots.md) | Every plot made this session, how to read it, and the paper it's modelled on (images embedded) |

`figures/` holds frozen copies of the five plots so `05-plots.md` renders self-contained.

## The 30-second version

- **Drift-diffusion / HSSM recap:** a decision is modelled as evidence accumulating at a **drift
  rate `v`** from a **start point `z`** to a **boundary `a`**, plus a **non-decision time `t`**
  (encoding + motor). HSSM fits these *hierarchically* and *Bayesianly* (posteriors, not point
  estimates).
- **What changed:** `t` is now a fixed constant (the task has a ~500 ms response lockout), `a`
  varies per participant, the model input is saved, and the start-point model was re-keyed so it
  can tell **which prior level** moves `z`.
- **The science thread:** Franzen → priors act as *origin* (`z`) vs *gain* (`v`); Romei → faster
  alpha → higher `v`. The demo *directionally* reproduces Jonathan's pattern (**`z` ← low-level
  prior, `v` ← both**) but is underpowered at n=3.
- **New plots:** a readable trace, Romei-style coefficient histograms, Franzen-style ridgelines,
  and a corrected drift-diffusion schematic.
