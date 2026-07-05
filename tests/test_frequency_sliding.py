"""
Unit tests for the Hilbert "frequency sliding" instantaneous-alpha-frequency
estimator (Cohen 2014; Romei & Tarasi 2026, Nat Commun 17:3384).

These use synthetic signals with a KNOWN frequency, so correctness is checked
against ground truth without needing EEG data. The estimator is pure
numpy/scipy, so these run in any env (no mne / inputs.json needed).
"""
import numpy as np
import pytest

from frequency_sliding import instantaneous_alpha_frequency

SFREQ = 250.0


def _sine(freq, dur=2.0, sfreq=SFREQ, amp=1.0, phase=0.0):
    t = np.arange(int(dur * sfreq)) / sfreq
    return amp * np.sin(2 * np.pi * freq * t + phase)


def test_recovers_known_frequency_of_pure_sine():
    # A clean 10.5 Hz oscillation must be recovered as ~10.5 Hz.
    data = _sine(10.5)[None, :]                      # (1 epoch, n_times)
    mean_if, _ = instantaneous_alpha_frequency(data, SFREQ, 8.5, 12.5)
    assert mean_if.shape == (1,)
    assert abs(mean_if[0] - 10.5) < 0.3


def test_returns_a_finite_value_for_every_epoch():
    # The whole point vs FOOOF: an estimate on EVERY trial, never NaN.
    rng = np.random.default_rng(0)
    clean = np.array([_sine(9.0), _sine(11.0), _sine(12.0)])
    data = clean + 0.5 * rng.standard_normal(clean.shape)
    mean_if, sd_if = instantaneous_alpha_frequency(data, SFREQ, 7.0, 13.0)
    assert mean_if.shape == (3,)
    assert np.all(np.isfinite(mean_if))
    assert np.all(np.isfinite(sd_if))


def test_tracks_relative_frequency_differences():
    # A slower vs a faster oscillation must come out ordered and near-truth.
    data = np.array([_sine(9.0), _sine(12.0)])
    mean_if, _ = instantaneous_alpha_frequency(data, SFREQ, 7.0, 13.0)
    assert mean_if[0] < mean_if[1]
    assert abs(mean_if[0] - 9.0) < 0.5
    assert abs(mean_if[1] - 12.0) < 0.5


def test_is_unbiased_by_aperiodic_1_over_f_trend():
    # A strong 1/f-like drift added under the oscillation must not pull the
    # estimate (the derivative pre-whitening step is what buys this).
    t = np.arange(int(2.0 * SFREQ)) / SFREQ
    aperiodic = 5.0 * np.exp(-t / 0.5)               # big low-frequency ramp
    data = (_sine(10.0) + aperiodic)[None, :]
    mean_if, _ = instantaneous_alpha_frequency(data, SFREQ, 8.0, 12.0)
    assert abs(mean_if[0] - 10.0) < 0.3
