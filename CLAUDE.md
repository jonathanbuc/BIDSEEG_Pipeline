# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working conventions (read this first)

**After finishing any task, write a teaching note.** Drop a Markdown file in `docs/learning/`
(named `NN-short-topic.md`, numbered in order) that explains the work to me as if teaching it. The
PI quizzes me on specifics, so each note must cover:
- **What I changed** — the concrete edit, in plain terms.
- **The CS concept** — the computer-science idea behind it (e.g. parallelism, caching,
  idempotency, dependency pinning) and why it applies here.
- **The psych/neuro concept** — the EEG / signal-processing / experimental idea it touches (e.g.
  why line-noise filtering matters, what ICA separates, what a drift-diffusion parameter means).
- **Why it helped / what improved** — the before→after, ideally with the measurable effect
  (faster, smaller, correct, reproducible).

Keep these notes tight and readable — they're for learning, not a changelog dump.

**Comment sparingly and naturally.** Don't blanket the code with explanatory comments. Add a
comment only where the *why* isn't obvious from the code; match the surrounding style; sound like a
human wrote it, not a tutorial. No narrating what the next line plainly does.

**Reproducibility of the *old* results is no longer a constraint.** The PI confirmed he does not
need previously published numbers to reproduce bit-for-bit, so we're free to fix correctness bugs
and change behavior to make the pipeline better. Still note any behavior-changing fix in its
teaching note so the change is traceable.

## What this is

A semi-automated EEG processing pipeline (Buchholz & Hesselmann) that takes BIDS-formatted EEG
data from raw recordings through preprocessing, artifact correction, EEG analysis, and behavioral
modeling. The bundled dataset (`data/BIDShierPriors`, 3 subjects) is from a Hierarchical Priors RDK
task — see `ExperimentGuide_HierarchicalPriors.md` for the experiment design and the
`epochs.metadata` column dictionary (essential for understanding condition coding like `exp`,
`prior`, `response_prior`, `coh`).

## Environment & running

```bash
conda env create -f environment_setup.yml   # creates env "mne-env"
conda activate mne-env
```

Each module is a standalone script that takes **`inputs.json` as its only argument**. Run them in order:

```bash
python z_BIDSification_module.py inputs.json   # ONLY if data is not already BIDS (e.g. BrainVision -> BIDS)
python a_preprocessing_module.py inputs.json   # downsample, reref, line noise, filter
python b_ArtifactCorrection_module.py inputs.json   # epoch, ICA, RANSAC interp, autoreject
python c_EEGAnalysis_module.py inputs.json     # TFR, cluster-perm tests (CBPT), FOOOF/specparam
python d_BehavAnalysis_module.py inputs.json   # SDT, generalized DDM (pyddm)
```

There are no tests, linters, or build steps. Stack: MNE-Python 1.9, mne-bids, autoreject,
mne-icalabel, fooof, pyddm. Python 3.10 (note: `.pyc` artifacts show 3.13 — match the conda env, not those).

## Critical architecture facts

**`inputs.json` is the single source of truth AND mutable pipeline state.** It holds every
parameter, but the modules also *rewrite* it as they run via `utils.update_inputs(...)`:
- `basic.subject_ID` is overwritten with the current subject inside the per-subject loop.
- `basic.current_step` is overwritten with the processing-stage tag after each save.

This means the file on disk after a run reflects the last subject/step, not your original config.
Re-running a later module relies on `current_step` pointing at the right stage. **Do not assume
`inputs.json` is read-only**, and be careful editing it while a module is running.

**`utils_module.py` executes top-level code at import time** (its bottom block reads
`sys.argv[1]`, loads inputs, builds `bids_path_preprocessing`, loads the log). Every other module
does `import utils_module as utils`, so **importing any module requires `inputs.json` as
`sys.argv[1]`** — you cannot import these modules in a plain REPL without faking argv. `log_msg`
and `log_load` even re-import `bids_path_preprocessing` from `utils_module` as a global.

**Processing stages are encoded in the BIDS filename via the `processing=` entity.** The loaders
and savers thread a stage tag through `BIDSPath`. The canonical chain is:

```
(BIDS raw, no tag) -> 01rawfilter -> 02ICA -> 03chInterp -> 04epochsCorr -> 05tfr
```

`current_step` in `inputs.json` selects which stage's file `utils.load_preprocessing_step` reads.
Continuous data is saved as EDF via `write_raw_bids`; epochs/TFRs as `.fif`; autoreject objects as
`.hdf5`/`.npz`; ICA/RANSAC as `.fif`/pickle.

**`utils.save_preprocessing_step(file, step, bidspath, subject)` is a type-dispatched saver** — a
large `match str(type(file))` that picks the right serialization per MNE/autoreject object type.
When adding a new artifact type, add a `case` there (the `case _` branch silently prints "NOT SAVED").

**`utils.get_bidspath(inputs, option, subjects)`** centralizes all path construction. Key options:
`'bids_proc'` (derivatives/BIDSprocessed root), `'epochs_list'`/`'tfr_list'` (lists of per-subject
file paths for group analysis).

**Two parallel logging systems:**
- `log_dataframe.csv` — one row per (subject, session, task), columns added on the fly by
  `utils.log_update(log_df, column, value)`. Carries quantitative provenance (filter cutoffs,
  % epochs removed, RT outlier counts, etc.). Loaded/saved with `log_load`/`log_save`.
- `<module-name>_log.txt` under a `logfiles/` dir — human-readable. `utils.log_msg(...)` prints to
  console then *redirects `sys.stdout` to the txt file* to write a timestamped line, keyed to the
  calling script's basename.

## Data flow & layout

- Input BIDS: `basic.bids_root_in` (`./data/BIDShierPriors`).
- Derivatives: `basic.bids_root_out` -> `.../BIDSprocessed/` (per-subject processed files,
  `diagnostics/` PSD plots, `autoreject/`) and `.../results/` (`groupEEG`, `groupBehavioral`,
  master CSVs like `EEG_alphapower_hierprior.csv`, `behav_results_hierprior.csv`).
- Behavioral source: modules `b` and `d` read raw **PsychoPy** logs from `basic.sourcedata`
  (`behavdata_prep`), not from BIDS — that's where trial metadata (`rt`, `coh`, `prior`, etc.)
  originates before being attached to `epochs.metadata`.

## Config structure (`inputs.json`)

- `basic` — IDs, paths, `event_dict` (trigger label -> code), `current_step` (mutated, see above).
- `perform` — boolean toggles; every major step (`ICA`, `channel_interpolation`, `perform_tfr`,
  `compute_ddm`, ...) is gated by a flag here. This is the primary way to turn pipeline stages on/off.
- `preprocessing` / `ArtifactCorrection` / `Analysis` — per-stage numeric parameters.
- `Analysis.conditions` is `{"<metadata_column>": [<level>, ...]}` — the first key names the
  `epochs.metadata` column used to split conditions (here `exp`: base/lowlevel/highlevel), and this
  contract is relied on across `c` and `d` (`cond_col, conditions = list(condition_dict.items())[0]`).

## Conventions

- Per-subject loops drive every module: iterate `utils.find_subjects(root)`, set
  `subject_ID`/`current_step` in `inputs.json`, load -> process -> save -> log, then a group-level
  pass aggregates across subjects.
- `inputs.json` files use object-key `match` statements heavily (Python 3.10 structural matching)
  for method dispatch (filter type, reref type, save type) — extend these rather than adding if/elif.
- Plotting lives in `plotting_module.py` (imported as `plotting`); shared helpers like
  `raincloud_plot`, `paired_plot`, `model_validation`, `rt_descriptive_plots`.
