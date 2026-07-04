# 15 — Reconciling two divergent branches into a clean, data-free PR

## What I changed

The old pull request (`anike-pipeline-hardening`, PR #5) could no longer be merged. While that
branch sat open, Jonathan committed his **spectral-parameterization** analysis straight to `main`
and, at the same time, added `/data/` (plus hidden/temp files) to `main`'s `.gitignore` and stopped
tracking the data folder. My branch, meanwhile, still *tracked* 292 files under `data/`. So the two
branches disagreed about (a) hundreds of data files and (b) several core modules we had both edited
independently. Git couldn't produce a clean merge, and neither could Jonathan by hand.

The fix was to rebuild my work as a **single clean commit on top of the current `main`**, containing
only code and docs — no data, no hidden files:

- Branched a fresh `anike-pipeline-hardening-v2` from `origin/main`.
- Ran a real 3-way merge of my branch into it, then resolved every conflict deliberately:
  - **`c_EEGAnalysis_module.py`** → kept Jonathan's version *verbatim*. His `spectral_parameterization`
    / `cbpt_spectral_parameterization` rewrite supersedes my older FOOOF path, and his `subject_tfr`
    doesn't define the variables my configurable-TFR tweak needs — grafting it would have thrown a
    `NameError`. So that tweak is deferred to a separate, tested change.
  - **`e_HSSM_module.py`** → kept my hardened version. It strictly supersedes the copy on `main`
    (`fix_t`, optional `z`/`t`/`a` formulas with a logit-linked `z`, `load_group_data`, `idata.nc`
    persistence, richer posterior plots).
  - **`plotting_module.py`** → *union* of both sides: `main`'s FOOOF helpers **and** my `hssm_*`
    posterior-plot helpers (which `e_HSSM` imports).
  - **`README` / `inputs.json` / `environment_setup.yml`** → merged (HSSM documented as module 5,
    `hssm==0.2.10`, OpenNeuro `ds008083` provenance kept).
- Dropped all `data/` from tracking (kept the files locally), and set `.gitignore` to ignore all of
  `/data/` — matching `main`'s policy — plus `openneuro data/`.
- Committed as **one** non-merge commit so the PR shows a clean net diff, not 20 commits of
  now-moot data churn.

## The CS concept — three-way merge, and *history* vs *tree*

A git merge is a **three-way** operation: it compares each side against their common ancestor (the
*merge base*, here `b3f4287`) to decide who changed what. When both sides change the same lines, git
can't choose and emits conflict markers. Two files here were **add/add** conflicts — both branches
created `e_HSSM_module.py` after the merge base, so git had no ancestor to diff against and marked
the whole file conflicted. Resolving those isn't mechanical; you need to know *why* each side looks
the way it does (which is where the memory of our HSSM work — `fix_t`, `z~prior` — actually paid off).

The second idea is separating a commit's **tree** (the snapshot of files) from its **history** (the
chain of parent commits). I could have committed the merge as-is — correct tree, but its history
would drag in every data-churn commit from the old branch, re-exposing Jonathan to the exact noise
that confused him. Instead I kept the resolved *tree* but reset the *history* to a single parent
(`origin/main`), so the PR reads as one clean delta. Same files, honest simpler story.

Finally: **`.gitignore` only stops *new* files from being tracked; it never untracks what git already
follows.** That's the whole root cause — my branch kept updating 292 data files that were tracked
before the ignore rule existed. `git rm --cached` is what actually removes them from the index while
leaving the real files on disk.

## The psych/neuro concept — two analyses that had to both survive

The merge wasn't abstract; it was two different *analyses of the same EEG* that both had to live:

- **Spectral parameterization (Jonathan's, in `c_EEGAnalysis`)** decomposes the power spectrum into
  an **aperiodic 1/f background** and true **oscillatory peaks** (FOOOF/specparam). This matters
  because raw alpha "power" conflates a genuine ~10 Hz rhythm with a shift in the broadband slope;
  parameterizing separates the oscillation from the background so a condition effect on *alpha* isn't
  really an artifact of the aperiodic tilt.
- **Hierarchical DDM (mine, in `e_HSSM`)** models the *behavior* those rhythms accompany — decomposing
  choices and RTs into drift rate (`v`), boundary (`a`), start-point bias (`z`), and non-decision time
  (`t`). The hardened version fixes `t` to a constant (the task blocks responses for ~500 ms, so that
  latency isn't decision time) and lets `z` carry a *pre-evidence* prior bias distinct from the
  *in-evidence* drift bias in `v`.

Keeping both is the point: the pipeline is being generalized so trial-level spectral markers can
eventually become covariates on the DDM drift rate — the "does the alpha clock move evidence
accumulation" question. A merge that dropped either side would have broken that arc.

## Why it helped

- **Before:** PR #5 unmergeable — 292 tracked data files + divergent `.gitignore` + 7 co-edited
  modules all conflicting; Jonathan couldn't resolve it.
- **After:** `anike-pipeline-hardening-v2` is a single commit on top of `main`, **0 data files**,
  no hidden/junk files, **3,711 insertions of code + docs**, all modules `py_compile`-clean, and it
  fast-forwards onto `main` with zero conflicts — while preserving *both* Jonathan's spectral work
  (`c_EEGAnalysis` is byte-identical to `main`) and the HSSM hardening.

## One integration caveat to follow up

`main`'s `c_EEGAnalysis` writes a single combined `EEG_iaf.csv`, whereas `e_HSSM.load_group_data`
looks for per-subject `sub-*_trial_alpha.csv` when a formula references an alpha covariate. The
**default** `formula_v` has no alpha term, so normal runs are unaffected — but the trial-alpha → DDM
covariate handoff needs one small reconciliation before that experimental path works end-to-end.
