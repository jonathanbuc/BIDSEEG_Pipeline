# 17 — Scaling the pipeline from 3 subjects to the full 43-subject sample

## What I changed

We finally have the full dataset: the OpenNeuro release of the Hierarchical Priors study
(`ds008083`, 43 subjects) landed in `openneuro data/default`. The whole point of the pipeline
hardening was to be ready for this moment — enough subjects to actually test hypotheses. Before
running anything I did three things:

- **Pointed the pipeline at the new data without disturbing the demo.** Instead of editing the
  canonical `inputs.json` (which drives the bundled 3-subject demo), I wrote a parallel
  `inputs_openneuro.json` with `bids_root_in` → `./openneuro data/default`, its own
  `bids_root_out` derivatives root, and a `sourcedata` slot pointing where the behavioral logs
  will go. The modules take their config path as `sys.argv[1]`, so this is a clean switch with the
  demo config left pristine.
- **Validated on one subject before committing the fleet.** Ran `a_preprocessing_module.py` on a
  single subject (temporarily trimming `participants.tsv` to one row, with a backup) to confirm
  paths, channels and montage were right, *then* ran all 42 usable subjects.
- **Sanity-checked the EEG across all 42** by reading each subject's BIDS `channels.tsv` /
  `events.tsv` directly (no need to load the multi-hundred-MB EDFs): 31 EEG channels, identical
  channel set/order everywhere, ROI electrodes (O1/Oz/O2) present, ~480 stimulus triggers/subject.
  Homogeneous — no per-subject quirks to patch.

One data gap: `sub-001`'s EEG never finished downloading (a 0-byte file), so the usable sample is
42. The behavioral half of the pipeline (`b`–`e`) is still gated on the PsychoPy source logs, which
the OpenNeuro upload doesn't include.

## The CS concept — validate-on-one, and reading the index instead of the payload

Two ideas did the work here. First, **incremental rollout / canary**: run the expensive batch job
on *one* item end-to-end before all 42, so a config or path bug surfaces in 24 seconds instead of
15 wasted minutes deep into the run. It's the same instinct as a canary deploy — cheap failure
first. Second, **read the metadata, not the data**: to check channel consistency and trigger counts
I parsed the small sidecar `.tsv` files (kilobytes) rather than opening 42 × ~150 MB EDFs. BIDS is
designed so the *index* (channels, events, sfreq in JSON) is separable from the *signal payload*;
exploiting that made a whole-dataset audit run in seconds. The audit also caught a nice invariant —
the demo's sub-001/002/003 trigger counts match the new sub-001/002/003 exactly, proving the
behavioral-log numbering will line up with the BIDS numbering (no remapping needed).

Keeping a separate `inputs_openneuro.json` instead of mutating `inputs.json` is basic **configuration
isolation**: the pipeline treats that file as mutable state (it rewrites `subject_ID`/`current_step`
into it as it runs), so pointing a *copy* at the new data means the run's bookkeeping scribbles on
the copy, not on the config that reproduces the demo.

## The psych/neuro concept — why sample size is the whole game here

The reason we waited for 43 subjects is **statistical power**. The scientific claims — that a
low-level prior shifts the drift-diffusion *starting point* `z` while evidence quality shifts the
*drift rate* `v`, and that individual alpha frequency tracks `v` — are subject-level effects
estimated with noise. With n=3 you can build the machinery but you can't distinguish a real effect
from sampling jitter; the hierarchical DDM (HSSM) in particular *shrinks* per-subject estimates
toward the group and only becomes trustworthy when the group has enough members to define a
sensible population distribution. Scaling to 42 is what turns the pipeline from a plumbing demo into
something that can actually support or reject the pre-registered hypotheses.

The EEG homogeneity check matters neurally too: because every subject shares the same 31-channel
montage and the same occipital ROI, the alpha-power and IAF measures are computed over comparable
scalp locations across people — a precondition for pooling them in one model rather than comparing
apples to oranges.

## Why it helped — before → after

- **Before:** the pipeline had only ever run on 3 bundled subjects (250 Hz demo data); no config
  existed for the real dataset, and its integrity was unverified.
- **After:** all 42 usable subjects are through stage `01rawfilter` (1000 Hz → 250 Hz, average
  ref, 0.1–40 Hz FIR) with zero errors, the EEG is confirmed homogeneous, and a reusable
  `inputs_openneuro.json` cleanly drives the new dataset. The only remaining gate to the full
  analysis is the behavioral source logs for subjects 4–43.

## Follow-up

`b`–`e` are blocked until the PsychoPy logs (`sub-0XX_RDKdeutsch_*.csv` + `hierPrior_traitVariables.csv`)
for all subjects arrive; they go in `openneuro data/sourcedata/`. Numbering is expected to match
BIDS (verified for 1–3). `sub-001` needs its EEG re-downloaded from OpenNeuro if we want the full 43.
