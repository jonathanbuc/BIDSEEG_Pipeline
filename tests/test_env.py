"""
Fast environment sanity checks. Run inside the `mne-env` conda environment.

These guard against the kind of reproducibility gap that shipped in this repo:
a module (c_EEGAnalysis) imported `pingouin`, but it was never listed in
environment_setup.yml, so a clean `conda env create` produced an env that
crashed at runtime. This test fails loudly if any required package is missing.
"""
import importlib

import pytest

# (import_name, expected_version_or_None) — versions pinned in environment_setup.yml
REQUIRED = [
    ("mne", "1.9.0"),
    ("mne_bids", "0.16.0"),
    ("autoreject", "0.4.3"),
    ("mne_icalabel", "0.7.0"),
    ("fooof", "1.1.1"),
    ("pyddm", "0.9.0"),
    ("pingouin", None),
    ("numpy", None),
    ("scipy", None),
    ("pandas", None),
    ("matplotlib", None),
    ("h5py", None),
    ("statsmodels", None),
    ("seaborn", None),
]


@pytest.mark.parametrize("name,expected", REQUIRED, ids=[r[0] for r in REQUIRED])
def test_required_package_importable(name, expected):
    mod = importlib.import_module(name)
    if expected is not None:
        assert mod.__version__ == expected, (
            f"{name} version {mod.__version__} != pinned {expected}"
        )
