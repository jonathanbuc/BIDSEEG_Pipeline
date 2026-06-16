# 01 — Where the EEG data actually came from

*Teaching note for the "make the repo runnable from a clean clone" work.*

## The confusion: "I thought we had data"

You did — sort of. The repo had the **paperwork** for the data but not the **data itself**.

A BIDS dataset is split into many small sidecar files plus the big signal file:

| File | What it holds | Was it in the repo? |
|------|---------------|---------------------|
| `participants.tsv` | list of subjects | ✅ yes (but listed 43 phantom subjects) |
| `*_channels.tsv` | the 64 electrode names/types | ✅ yes |
| `*_events.tsv` | when each stimulus/response happened | ✅ yes |
| `*_electrodes.tsv` | 3D positions of electrodes on the scalp | ✅ yes |
| `*_eeg.json` | sampling rate, reference, filtering metadata | ✅ yes |
| **`*_eeg.edf`** | **the actual voltage-over-time brain recording** | ❌ **missing** |

So when you ran `python a_preprocessing_module.py inputs.json`, it had the menu but no food — it
crashed with `FileNotFoundError` because the one file that holds the actual signal was never
committed. The metadata made it *look* like the data was there.

## What I did

1. Found the real source recordings sitting locally in `data/sourcedata/raw/` — three BrainVision
   files per subject (`.eeg` = signal, `.vhdr` = header, `.vmrk` = event markers). These are the
   original ~1000 Hz recordings. They're large, so they're **gitignored** (never committed).
2. Wrote `scripts/make_demo_data.py`, which runs the BIDSification step (`z_BIDSification`) to:
   read each raw recording → **downsample it to 250 Hz** → write it back into the BIDS folder as an
   EDF file.
3. Committed those three EDFs (~50 MB each, ~150 MB total) using **Git LFS**.
4. Trimmed `participants.tsv` from 43 listed subjects down to the 3 we actually have data for.

Result: a fresh `git clone` now genuinely has runnable EEG data, and the pipeline goes
end-to-end (a → b → c → d) on it.

## The CS concept: data provenance + lossy, reproducible artifacts

**Provenance** = a documented, repeatable path from a trusted source to the artifact you ship.
We don't commit a mystery binary; we commit something we can regenerate on demand from
`sourcedata/raw` via a script. If anyone doubts the demo, `make_demo_data.py` is the receipt.

Two more ideas show up here:
- **Separating source-of-truth from derived artifacts.** Raw BrainVision = source of truth (big,
  gitignored). The committed EDF = a *derived* artifact (small, shippable). Mixing these up is how
  repos rot — people edit the derived file and lose the ability to regenerate it.
- **Git LFS (Large File Storage).** Git is built for small text diffs; a 50 MB binary bloats every
  clone forever because git keeps all history. LFS stores a tiny *pointer* in git and parks the real
  bytes in a side store, fetched only when needed. That's why the `.edf` files are LFS-tracked.

The **downsampling to 250 Hz is deliberately lossy** — a smaller demo that's good enough to run the
whole pipeline, but explicitly *not* the data behind any published result.

## The psych/neuro concept: sampling rate and the Nyquist limit

EEG is a continuous voltage signal; the recorder samples it many times per second. The
**sampling rate** (Hz) is how many snapshots per second.

The **Nyquist theorem** says you can faithfully represent frequencies only up to *half* the sampling
rate. So:
- 1000 Hz raw → can represent up to 500 Hz of brain/noise activity.
- 250 Hz demo → can represent up to 125 Hz.

For this pipeline that's fine: the science here lives in low frequencies — the **alpha band (≈8–12
Hz)** that the analysis (`c`) measures, plus everything the 40 Hz low-pass filter keeps. 125 Hz of
headroom comfortably covers all of it. Downsampling from 1000 → 250 Hz throws away high-frequency
content we were going to filter out anyway, while making every file 4× smaller and every processing
step roughly 4× faster. That's why it's the right trade for a demo.

## Why it helped / what improved

- **Before:** clone the repo → run the pipeline → instant crash, no data. Effectively a broken
  submission.
- **After:** clone → conda env → run a→b→c→d to completion on real (if downsampled) data, with a
  documented, regenerable provenance trail and a `participants.tsv` that matches reality.
