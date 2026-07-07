# 08 — Documenting the HSSM module in the README

*Teaching note for adding a "Module 5: HSSM" section to `README.md` so end-users (not just
developers reading `docs/learning/`) know the module exists and how to run it.*

---

## What I changed

- Bumped the intro from "4 modules" to "5 modules" and added a one-line HSSM entry to the overview
  list.
- Added a `### 2.5 - Module 5: HSSM` usage section: prerequisites (run module `d` first, install
  `hssm`), how to flip `perform.compute_hssm`, the run command, an annotated `Analysis.hssm` config
  block, and the output files.
- Fixed a pre-existing typo: the EEG-analysis and behavioral sections were *both* labelled
  "Module 4". EEG analysis is now correctly "Module 3".

No code changed — this is pure documentation.

## The CS concept — documentation layers and progressive disclosure

The repo already had three deep HSSM teaching notes (`03`, `04`, `06`), so why touch the README?
Because they serve different audiences. This is **progressive disclosure**: the README is the
*public interface* — the minimum a non-coder needs to run the thing — while `docs/learning/` is the
*reference layer* for the theory. Putting NUTS math in the README would violate the project's own
promise that "no extensive coding expertise is required"; omitting the run command would leave the
feature invisible. The fix is a thin README section that *links down* to the deep notes, so each
reader stops at the depth they need.

A second idea: the README is a **contract**, and the most important clause here is an **ordering
dependency**. Module `e` is not independent — it consumes a CSV that module `d` produces. The other
modules can be described in isolation; this one can't, so the docs have to make the
"run `d` first, or you get a `FileNotFoundError`" precondition explicit. Undocumented preconditions
are how good code produces bad first-run experiences.

## The psych/neuro concept — what the README has to convey in one paragraph

The substantive thing a user must grasp is *why this module is different from module 4*, not the
Bayesian machinery. Module 4 fits a drift-diffusion model **per subject** (no pooling); module 5
fits **one hierarchical model across all subjects** (partial pooling) and returns a **posterior
distribution** per parameter instead of a point estimate. That single contrast — point estimates +
t-test vs. a posterior you can interrogate directly — is the whole reason the module exists, so the
README leads with it and leaves the *how* (NUTS, R-hat, HDIs) to the linked notes.

I also surfaced the `formula_v` knob in plain terms, because it's the one config line that encodes a
scientific question: `v ~ 1 + exp * coh_level + (1|participant)` says drift rate depends on
condition, coherence, and their interaction, per participant — and swapping in `v ~ alpha_power + …`
is how you'd eventually test a trial-level EEG→behavior link.

## Why it helped / what improved

Before: the HSSM module was fully built and enabled (`compute_hssm: true`) but invisible in the
README — a new user following the Quick Start would run modules 1–4 and never know module 5 existed,
or would try to run it before module `d` and hit an opaque error. After: the module is discoverable,
its prerequisite is stated up front, every config field has an inline comment, and the duplicate
"Module 4" numbering no longer breaks the reading flow.
