# Design Spec — Workstream A: Safety Net (env + runnable demo + tests + hygiene)

**Date:** 2026-05-31
**Status:** Draft for review
**Author:** Aniket (with Claude Code)

---

## 0. Context

This git repo is the PI's **competition-submission** version of the BIDS-EEG pipeline
(Buchholz & Hesselmann, "Hierarchical Priors") — a polished, self-contained showcase that
ships 3 subjects already in BIDS format. The separate `HierarchicalPriors_pipeline/` folder is
the PI's original *working* delivery (raw data + `a–e` names); we keep it locally **only as a
data source** and it is gitignored. The goal of this workstream is to make the competition
repo **good**: reproducible, and **runnable end-to-end from a clean clone**.

We are hardening the pipeline across five workstreams; this spec is **Workstream A**, done
first because it introduces **zero numeric change** and gives us a safety net to verify later
changes:

- **A. Safety net** (this spec) — env + runnable demo + smoke test + hygiene
- **B. Correctness bug fixes**
- **C. Reproducibility refactor** (stop mutating `inputs.json`; seeds)
- **D. Performance** (PI explicitly asked to reduce ICA/autoreject resource use)
- **E. Structure & cleanup** (split monolith; PI asked to clear deprecation warnings)

### Verified facts

1. **Naming is intentional.** The `z/a/b/c/d` scheme is the PI's deliberate choice for this
   repo: BIDSification is `z_` (optional, since data ships already-BIDSified) and
   `a_preprocessing` is the real entry point. **We keep these names.**
2. **Clean clone can't currently run b/c/d.** Modules `b_ArtifactCorrection`,
   `c_EEGAnalysis`, `d_BehavAnalysis` read per-subject PsychoPy logs (and a trait CSV) from
   `sourcedata/`, which is **not shipped** — only the BIDSified output is. So a fresh clone
   breaks after preprocessing. **This is the main gap to fix.**
3. **The needed demo data is tiny (~1.8 MB).** Only these are required for the 3 demo subjects:
   `sourcedata/logfiles/sub-00{1,2,3}_RDKdeutsch_*.csv` (~0.6 MB each) and
   `sourcedata/hierPrior_traitVariables.csv` (37 KB). The 1.6 GB raw `.eeg` is **not** needed
   (the BIDSified EEG already ships as EDF via LFS) and stays out of git.
4. **The pinned environment can't run the code as-is.** `environment_setup.yml` pins Python
   **3.10**, but `b_ArtifactCorrection_module.py:193` uses a same-quote nested f-string that
   needs Python **≥3.12**. This is an accidental typo, fixable to be 3.10-legal.
5. **No conda installed** locally; Python 3.11/3.13 present. **No tests** exist.

### Decisions (with the user)

- **Conda only** (matches PI's instructions/package list).
- **Keep `z/a/b/c/d` naming** (do NOT rename); fix the copy-pasted wrong file headers.
- **Make a clean clone runnable end-to-end** by shipping the small demo behavioral CSVs.
- **Python floor: keep 3.10** to match this repo's own env file; **fix the f-string** so the
  code is actually 3.10-legal. (Faithful to the competition repo; revisit only if env
  resolution forces it.)
- `HierarchicalPriors_pipeline/` is gitignored (local data source only). ✅ done
- Reproducibility-exactness constraint is **unknown/cautious** pending the PI — so A makes
  **zero numeric change**; later behavior-changing fixes get flagged/gated.

---

## 1. Goals / Non-goals

**Goals**
- A conda env (Python 3.10, PI's package pins) that installs and imports the full stack.
- `z`(skipped)→`a`→`b`→`c`→`d` runs to completion **from a clean clone** on the 3 demo subjects.
- A test scaffold: fast deterministic unit tests + one `slow`-marked end-to-end smoke test.
- Clean repo: correct headers, no committed build artifacts, sane `.gitignore`, demo data
  tracked but raw EEG ignored.

**Non-goals (deferred)**
- No output-changing bug fixes (B), no `inputs.json`-mutation/seed work (C), no perf changes
  (D), no module splitting/dedup/dead-code removal (E).
- No numeric golden/regression snapshots yet (needs seeds from C + PI's constraint answer).
- No renaming; no shipping raw `.eeg`.

---

## 2. Plan

### A1 — Environment (conda, Python 3.10)

- **Prerequisite (user):** install Miniconda. Blocks env creation + smoke test only; all
  code-prep below proceeds without it.
- Fix `environment_setup.yml`: keep `python=3.10` and the PI's pins (`mne=1.9.0`,
  `mne-bids=0.16.0`, `autoreject=0.4.3`, `mne-icalabel=0.7.0`, `fooof=1.1.1`, `pyddm==0.9.0`);
  add explicit pins for numeric-affecting transitive deps (`numpy`, `scipy`, `pandas`,
  `statsmodels`, `pingouin`, `seaborn`, `matplotlib`, `h5py`) and `pytest` for dev.
- **Verify:** `conda env create -f environment_setup.yml` succeeds; an import probe passes for
  every package. If `fooof 1.1.1` forces `numpy<2`, capture that pin in the yml.

### A2 — Make a clean clone runnable

- **Ship demo behavioral data** (the ~1.8 MB): copy into the repo
  - `data/sourcedata/logfiles/sub-00{1,2,3}_RDKdeutsch_*.csv`
  - `data/sourcedata/hierPrior_traitVariables.csv`
  and track them in git. `inputs.json` already points `sourcedata` at `./data/sourcedata`.
- **`.gitignore` adjustment:** ignore only `data/sourcedata/raw/` (the big `.eeg`), not all of
  `sourcedata/`, so the small logfiles + trait CSV are tracked. (Update the rule added earlier.)
- **Fix the f-string** at `b_ArtifactCorrection_module.py:193` to be 3.10-legal (distinct quotes).
- **Fix copy-pasted file headers** so each header names its own file (`z/a/b/c/d` + plotting).
- **Reset `inputs.json` demo state:** `subject_ID` `043`→`001` (cosmetic; loops overwrite it).
- **Run** `a→b→c→d` on the 3 subjects from repo data; capture any execution-blocking errors as
  Workstream B candidates (fixed there, not here — unless a pure crash has an obviously safe,
  numerics-free fix).

### A3 — Test scaffold (`pytest`)

- `tests/` with **fast deterministic unit tests** on pure helpers (no heavy MNE run):
  `rt_cleansing`, SDT math, `get_bidspath`, `log_update`, FOOOF summary-row extraction.
- **One end-to-end smoke test** marked `@pytest.mark.slow`: runs each module on the 3 demo
  subjects (reduced "smoke" config if runtime demands — fewer ICA components / CBPT
  permutations) and asserts (a) no exception and (b) expected output files exist. **No numeric
  assertions** (deferred).
- Document `pytest -m "not slow"` (fast) and `pytest -m slow` (full).

### A4 — Repo hygiene

- `git rm --cached` the committed `__pycache__/*.pyc` and any tracked `.DS_Store`.
- `.gitignore` — present (PI folder, pycache, `.DS_Store`, venvs, `derivatives/`); adjust
  `sourcedata` rule per A2.

---

## 3. Risks / open questions

- **Env resolution:** `fooof 1.1.1` may conflict with `numpy 2.x` on py3.10; pin `numpy<2` if
  needed and record the working resolution. (Found during A1.)
- **Smoke-test runtime:** ICA + autoreject + CBPT on 3 subjects may be minutes; use a reduced
  smoke config in the test only.
- **CSV↔subject matching:** `behavdata_prep` matches `subject in csv_fname`; verify `'001'`
  matches `sub-001_RDKdeutsch_highleft.csv` (it does) and no false positives among the 3.
- **PI reproducibility constraint** still pending — does not block A.

---

## 4. Success criteria

1. `conda env create -f environment_setup.yml` yields an env where all pipeline imports succeed.
2. From a clean clone, `python a_preprocessing_module.py inputs.json` … through
   `d_BehavAnalysis_module.py inputs.json` complete on the 3 demo subjects with no edits.
3. `pytest -m "not slow"` passes; `pytest -m slow` completes the end-to-end smoke test.
4. `z/a/b/c/d` names kept; headers correct; no `.pyc`/`.DS_Store` tracked; raw `.eeg` + PI
   folder gitignored; demo CSVs tracked.
5. No analysis numerics changed by this workstream.
