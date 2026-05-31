"""
Regenerate the shipped demo BIDS dataset at a reduced sampling rate.

Provenance / reproducibility helper. The competition repo ships a DOWNSAMPLED
(250 Hz) 3-subject demo so that a clean clone can run the whole pipeline
end-to-end while staying within GitHub's Git-LFS quota. This script documents
exactly how that shipped demo data was produced:

    raw BrainVision  (data/sourcedata/raw/, NOT shipped - too large)
      --> z_BIDSification_module.py            (full 1000 Hz BIDS)
      --> scripts/make_demo_data.py            (resample to 250 Hz -> shipped)

250 Hz is the rate the preprocessing module downsamples to anyway
(inputs.json -> preprocessing.samplingrate_down), so the demo is scientifically
equivalent to running the full pipeline on the 1000 Hz data for these subjects.

Run inside the `mne-env` conda environment from the repo root:
    python scripts/make_demo_data.py inputs.json [target_sfreq]
"""
import sys
import os
import json

import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids


def main():
    inputs_path = sys.argv[1] if len(sys.argv) > 1 else "inputs.json"
    target_sfreq = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0

    with open(inputs_path) as f:
        inputs = json.load(f)

    root = inputs["basic"]["bids_root_in"]
    task = inputs["basic"]["task"]
    session = inputs["basic"]["session"]
    event_dict = inputs["basic"]["event_dict"]

    parts = pd.read_csv(os.path.join(root, "participants.tsv"), sep="\t")
    subjects = [s.replace("sub-", "") for s in parts["participant_id"]]

    for sub in subjects:
        bp = BIDSPath(subject=sub, task=task, session=session, root=root,
                      datatype="eeg")
        raw = read_raw_bids(bp, verbose=False)
        raw.load_data(verbose=False)
        sf = raw.info["sfreq"]
        if sf <= target_sfreq:
            print(f"sub-{sub}: already {sf:g} Hz (<= {target_sfreq:g}), skipping")
            continue
        print(f"sub-{sub}: resampling {sf:g} -> {target_sfreq:g} Hz")
        raw.resample(target_sfreq, verbose=False)
        write_raw_bids(raw, bp, event_id=event_dict, format="EDF",
                       overwrite=True, allow_preload=True, verbose=False)
    print("done")


if __name__ == "__main__":
    main()
