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

The Hilbert IAF core below is pure numpy/scipy (no mne, no inputs.json) so it
stays unit-testable against synthetic signals with a known frequency.
Electrode-cluster selection (TFR or FOOOF) imports mne/fooof only inside
those helpers.
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


def select_max_alpha_roi_cluster(
    epochs,
    roi,
    n_electrodes,
    alpha_band=(7.0, 13.0),
    method="fooof",
    tmin=None,
    tmax=None,
    # TFR
    freqs=None,
    n_cycles=None,
    decim=1,
    # FOOOF (aligned with utils._fit_channel)
    f_range=(1.0, 40.0),
    peak_threshold=1.0,
    max_n_peaks=6,
    peak_width_limits=(1.4, 8.0),
):
    """
    Select ``n_electrodes`` ROI channels with the highest alpha power.

    ``method='tfr'``
        Epoch-averaged Morlet power in ``alpha_band`` × ``[tmin, tmax]``.
    ``method='fooof'``
        Per channel: Welch PSD averaged over all epochs (as in
        ``utils._fit_channel``), FOOOF fit, rank by strongest alpha-peak
        power (PW) inside ``alpha_band``. Channels with no alpha peak get PW=0.

    Returns
    -------
    cluster : list of str
        Channel names, highest power first.
    powers : ndarray, shape (n_electrodes,)
        Ranking metric (TFR mean power or FOOOF peak PW) in the same order.
    """
    method = str(method).lower()
    lo, hi = alpha_band
    roi = list(roi)

    if method == "tfr":
        mean_power, ch_names = _alpha_power_tfr(
            epochs, roi, lo, hi, tmin, tmax, freqs, n_cycles, decim
        )
    elif method == "fooof":
        mean_power, ch_names = _alpha_power_fooof(
            epochs, roi, lo, hi, tmin, tmax, f_range,
            peak_threshold, max_n_peaks, peak_width_limits,
        )
    else:
        raise ValueError(f"method must be 'tfr' or 'fooof', got {method!r}")

    top = np.argsort(mean_power)[::-1][:n_electrodes]
    return [ch_names[i] for i in top], mean_power[top]


def _alpha_power_tfr(epochs, roi, lo, hi, tmin, tmax, freqs, n_cycles, decim):
    """Epoch-mean Morlet alpha power per channel."""
    if freqs is None:
        freqs = np.arange(lo, hi + 1e-9, 0.5)
    freqs = np.asarray(freqs, float)
    if n_cycles is None:
        n_cycles = freqs / 2.0

    tfr = epochs.compute_tfr(
        method="morlet", picks=roi, freqs=freqs, n_cycles=n_cycles,
        average=True, return_itc=False, decim=decim, verbose=False,
    )
    power = np.asarray(tfr.data)
    if power.ndim == 4:
        power = power[0]
    fmask = (tfr.freqs >= lo) & (tfr.freqs <= hi)
    t0 = tfr.times[0] if tmin is None else tmin
    t1 = tfr.times[-1] if tmax is None else tmax
    tmask = (tfr.times >= t0) & (tfr.times <= t1)
    return power[:, fmask][:, :, tmask].mean(axis=(1, 2)), list(tfr.ch_names)


def _alpha_power_fooof(epochs, roi, lo, hi, tmin, tmax, f_range,
                       peak_threshold, max_n_peaks, peak_width_limits):
    """
    FOOOF alpha-peak power per channel on the experiment-wide mean PSD.

    Mirrors ``utils._fit_channel``: ``spectrum = psd[:, ch, :].mean(axis=0)``,
    then FOOOF; ranking uses the tallest peak PW inside ``[lo, hi]``.
    """
    from fooof import FOOOF

    ep = epochs.copy()
    if tmin is not None or tmax is not None:
        ep.crop(tmin=0.0 if tmin is None else tmin,
                tmax=ep.tmax if tmax is None else tmax)
    ep = ep.pick(roi)

    psd = ep.compute_psd(method="welch", fmin=f_range[0], fmax=f_range[1],
                         verbose=False)
    freqs = psd.freqs
    # (n_epochs, n_channels, n_freqs) → trial-mean spectrum per channel
    spectra = psd.get_data().mean(axis=0)
    ch_names = list(psd.ch_names)

    powers = np.zeros(len(ch_names))
    for i, spectrum in enumerate(spectra):
        try:
            fm = FOOOF(
                aperiodic_mode="fixed", peak_threshold=peak_threshold,
                max_n_peaks=max_n_peaks, peak_width_limits=peak_width_limits,
                verbose=False,
            )
            fm.fit(freqs, spectrum, f_range)
            in_alpha = fm.peak_params_[
                (fm.peak_params_[:, 0] >= lo) & (fm.peak_params_[:, 0] <= hi)
            ]
            if len(in_alpha):
                powers[i] = in_alpha[:, 1].max()  # FOOOF PW (log10 power)
        except Exception:
            pass
    return powers, ch_names
