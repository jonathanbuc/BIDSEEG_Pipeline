"""
End-to-end smoke test: run the whole pipeline on the shipped 3-subject demo
data and assert it completes and produces the expected outputs.

This is the safety net for refactoring: it does NOT check numeric values (that
needs fixed random seeds, which come in a later workstream), only that the
modules run to completion and emit their key artifacts.

Marked `slow` (runs ICA / TFR / FOOOF / DDM — several minutes). Run with:
    pytest -m slow
Must be run inside the `mne-env` conda environment, from the repo root.

The pipeline mutates inputs.json at runtime, so this test backs it up and
restores it afterwards to keep the working tree clean.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INPUTS = REPO / "inputs.json"
DERIV = REPO / "data" / "BIDShierPriors" / "derivatives" / "BIDSprocessed"
RESULTS = DERIV / "results"

MODULES = [
    "a_preprocessing_module.py",
    "b_ArtifactCorrection_module.py",
    "c_EEGAnalysis_module.py",
    "d_BehavAnalysis_module.py",
]

# A representative output from each stage (relative to the derivatives root).
EXPECTED_OUTPUTS = [
    DERIV / "sub-001" / "ses-01" / "eeg"
        / "sub-001_ses-01_task-HierPrior_proc-01rawfilter_eeg.edf",   # a
    DERIV / "sub-001" / "ses-01" / "eeg"
        / "sub-001_ses-01_task-HierPrior_proc-04epochsCorr.fif",      # b
    RESULTS / "EEG_alphapower_hierprior.csv",                          # c
    RESULTS / "groupBehavioral" / "behav_results_hierprior.csv",      # d
]


@pytest.fixture
def preserve_inputs_json():
    """Modules rewrite inputs.json at runtime; snapshot and restore it."""
    backup = INPUTS.read_bytes()
    try:
        yield
    finally:
        INPUTS.write_bytes(backup)


@pytest.mark.slow
def test_full_pipeline_runs_on_demo_data(preserve_inputs_json):
    for module in MODULES:
        result = subprocess.run(
            [sys.executable, module, "inputs.json"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{module} exited {result.returncode}\n"
            f"--- stdout tail ---\n{result.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{result.stderr[-2000:]}"
        )

    missing = [str(p.relative_to(REPO)) for p in EXPECTED_OUTPUTS if not p.exists()]
    assert not missing, f"pipeline finished but expected outputs are missing: {missing}"
