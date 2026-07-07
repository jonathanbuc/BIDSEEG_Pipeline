"""
Hilbert "frequency sliding" estimator of instantaneous alpha frequency (IAF).

FOOOF-free individual-alpha-frequency method of Romei & Tarasi (2026,
Nat Commun 17:3384), after Cohen (2014): derivative pre-whitening -> zero-phase
band-pass around the alpha peak -> Hilbert instantaneous phase -> temporal
derivative of the unwrapped phase -> multi-window median denoising.

Unlike the per-trial FOOOF peak (which returns NaN whenever it fails to fit a
clean peak on a short, noisy single-trial spectrum), this returns a frequency
estimate on *every* trial, because it never has to detect a peak in a fitted
model — it reads the frequency straight out of the phase in the time domain.

Deliberately pure numpy/scipy (no mne, no inputs.json), so the numerical core is
unit-testable against synthetic signals with a known frequency.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import median_filter


def instantaneous_alpha_frequency(data, sfreq, l_freq, h_freq,
                                  median_windows_ms=(10.0, 400.0), n_median=10,
                                  edge_ms=100.0, filter_order=4):
    """
    Per-epoch instantaneous alpha frequency via frequency sliding.

    Parameters
    ----------
    data : array, shape (n_epochs, n_times) or (n_times,)
        Single-channel (or ROI-averaged) time series, one row per epoch.
    sfreq : float
        Sampling rate (Hz).
    l_freq, h_freq : float
        Band-pass edges (Hz) — the individual alpha band, e.g. IAF +/- 2 Hz.
    median_windows_ms : (float, float)
        Min/max median-filter window lengths (ms). Cohen's denoising applies
        several median filters spanning this range and takes their median.
    n_median : int
        Number of median-filter window lengths sampled across that range.
    edge_ms : float
        Milliseconds trimmed from each end before averaging, to drop the
        filter/Hilbert edge transients.
    filter_order : int
        Butterworth order, applied zero-phase via ``filtfilt``.

    Returns
    -------
    mean_if, sd_if : arrays, shape (n_epochs,)
        Per-epoch mean and SD of the instantaneous frequency over the
        (edge-trimmed) window.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_times = data.shape[-1]

    # 1) Derivative pre-whitening: differentiating flattens the aperiodic 1/f
    #    slope so it cannot bias the frequency estimate (their FOOOF-free trick).
    deriv = np.gradient(data, axis=-1)

    # 2) Zero-phase Butterworth band-pass around the individual alpha band
    #    (scipy filtfilt is the paper's MATLAB filtfilt).
    nyq = sfreq / 2.0
    b, a = butter(filter_order, [l_freq / nyq, h_freq / nyq], btype='band')
    filt = filtfilt(b, a, deriv, axis=-1)

    # 3) Analytic signal -> unwrapped instantaneous phase.
    phase = np.unwrap(np.angle(hilbert(filt, axis=-1)), axis=-1)

    # 4) Instantaneous frequency = d(phase)/dt, converted to Hz.
    inst = np.gradient(phase, axis=-1) * sfreq / (2.0 * np.pi)

    # 5) Denoise: apply median filters of several window lengths and take the
    #    median across them (Cohen 2014), suppressing phase-slip spikes.
    orders = np.unique(np.round(
        np.linspace(median_windows_ms[0], median_windows_ms[1], n_median)
        / 1000.0 * sfreq).astype(int))
    orders = orders[orders >= 1]
    stack = np.stack([median_filter(inst, size=(1, o), mode='nearest')
                      for o in orders], axis=0)
    inst = np.median(stack, axis=0)

    # 6) Trim edge transients, then summarise per epoch.
    edge = int(round(edge_ms / 1000.0 * sfreq))
    if 2 * edge < n_times:
        inst = inst[:, edge:n_times - edge]
    return inst.mean(axis=-1), inst.std(axis=-1)
