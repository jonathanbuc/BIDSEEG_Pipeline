"""
Corcoran et al. (2018) Savitzky–Golay individual alpha frequency (IAF).

Estimates peak alpha frequency (PAF) and centre of gravity (CoG) from EEG
power spectra using Savitzky–Golay smoothing of the normalised PSD and its
derivatives (Psychophysiology, e13064; https://doi.org/10.1111/psyp.13064).

Pipeline (per channel)
----------------------
1. Welch PSD (or use a precomputed spectrum)
2. Trim → normalise (P / mean P) → log-linear regression → minP threshold
3. SGF smooth + 1st/2nd derivatives → candidate peaks in ``wa``
4. Keep peaks above minP; require a dominant peak (pDiff)
5. Inflection-based Q, IAW bounds, optional FWHM
6. Cross-channel: Q-weighted mean PAF; mean IAW → CoG

Public API
----------
- ``corcoran_iaf(raw_or_epochs, ...)`` — compute Welch per channel
- ``corcoran_iaf_from_psd(freqs, psd, ...)`` — use existing spectra
- ``subject_iaf_hz(result)`` — scalar Hz for Hilbert band centering

Example
-------
>>> result = corcoran_iaf_from_psd(freqs, mean_psd, wa=(7, 13), c_min=1)
>>> iaf = subject_iaf_hz(result)  # PAF, else CoG, else NaN
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, welch
from scipy.stats import linregress


# ── public API ───────────────────────────────────────────────────────────────

def corcoran_iaf(
    inst,
    picks=None,
    welch_n_fft=1024,
    welch_overlap=0.50,
    sgf_fw=11,
    sgf_k=5,
    wa=(7.0, 13.0),
    p_diff=0.20,
    c_min=3,
    freq_trim=(1.0, 40.0),
):
    """
    Estimate cross-channel PAF / CoG from an MNE Raw or Epochs object.

    Epochs are concatenated in time before Welch (long continuous record).
    Prefer :func:`corcoran_iaf_from_psd` if spectra are already available.

    Parameters
    ----------
    inst : mne.io.BaseRaw or mne.BaseEpochs
    picks : sequence of str / int or None
        Channels to use. ``None`` → all EEG channels.
    welch_n_fft, welch_overlap : int, float
        Welch segment length and fractional overlap (paper: 1024, 0.5).
    sgf_fw, sgf_k : int
        Savitzky–Golay frame width (odd) and polynomial order.
    wa : (float, float)
        Alpha search window [Hz].
    p_diff : float
        Dominant-peak criterion (fractional height advantage).
    c_min : int
        Minimum channels needed for cross-channel means (capped at n_channels).
    freq_trim : (float, float)
        Frequency range kept for normalisation / regression [Hz].

    Returns
    -------
    dict with keys
        'paf_mean'  : cross-channel weighted mean PAF (Hz) or NaN
        'cog_mean'  : cross-channel mean CoG (Hz) or NaN
        'iaw_mean'  : (f1, f2) mean IAW bounds or (NaN, NaN)
        'channels'  : list of per-channel dicts (see _process_psd)
        'n_paf'     : number of channels contributing to PAF
        'n_cog'     : number of channels contributing to CoG / IAW
    """
    import mne  # only needed for Raw/Epochs path

    ch_names, data = _channel_data(inst, picks, mne)
    n_overlap = int(welch_n_fft * welch_overlap)

    channels = []
    for name, signal in zip(ch_names, data):
        res = _from_signal(signal, inst.info["sfreq"], welch_n_fft, n_overlap,
                           sgf_fw, sgf_k, wa, p_diff, freq_trim)
        res["channel"] = name
        channels.append(res)

    return _aggregate(channels, min(c_min, len(channels)))


def corcoran_iaf_from_psd(
    freqs,
    psd,
    ch_names=None,
    sgf_fw=11,
    sgf_k=5,
    wa=(7.0, 13.0),
    p_diff=0.20,
    c_min=3,
    freq_trim=(1.0, 40.0),
):
    """
    Same estimation as :func:`corcoran_iaf`, starting from precomputed PSDs.

    Parameters
    ----------
    freqs : array-like, shape (n_freqs,)
    psd : array-like, shape (n_freqs,) or (n_channels, n_freqs)
        1-D → single channel (use ``c_min=1``).
    ch_names : sequence of str or None
    Remaining kwargs
        As in :func:`corcoran_iaf`.

    Returns
    -------
    dict
        Same keys as :func:`corcoran_iaf`.
    """
    freqs = np.asarray(freqs, float)
    psd = np.atleast_2d(np.asarray(psd, float))
    n_ch = psd.shape[0]
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]

    channels = []
    for name, spectrum in zip(ch_names, psd):
        res = _from_psd(freqs, spectrum, sgf_fw, sgf_k, wa, p_diff, freq_trim)
        res["channel"] = name
        channels.append(res)

    return _aggregate(channels, min(c_min, n_ch))


def subject_iaf_hz(result, prefer="paf"):
    """Return a single IAF in Hz: preferred estimate, else the other, else NaN."""
    a, b = (result["paf_mean"], result["cog_mean"]) if prefer.lower() == "paf" \
        else (result["cog_mean"], result["paf_mean"])
    if np.isfinite(a):
        return float(a)
    if np.isfinite(b):
        return float(b)
    return np.nan


def print_iaf_result(result, subject_id=""):
    """Pretty-print a :func:`corcoran_iaf` / :func:`corcoran_iaf_from_psd` result."""
    prefix = f"[{subject_id}] " if subject_id else ""
    paf, cog = result["paf_mean"], result["cog_mean"]
    print(f"{prefix}PAF  = {paf:.3f} Hz  (n={result['n_paf']})" if np.isfinite(paf)
          else f"{prefix}PAF  = NaN  (n={result['n_paf']})")
    print(f"{prefix}CoG  = {cog:.3f} Hz  (n={result['n_cog']})" if np.isfinite(cog)
          else f"{prefix}CoG  = NaN  (n={result['n_cog']})")
    f1, f2 = result["iaw_mean"]
    print(f"{prefix}IAW  = [{f1:.2f}, {f2:.2f}] Hz" if np.isfinite(f1)
          else f"{prefix}IAW  = not estimated")
    for ch in result["channels"]:
        paf_s = f"{ch['paf']:.3f}" if ch["paf"] is not None else "–"
        q_s = f"Q={ch['q']:.4f}" if ch["q"] is not None else ""
        iaw_s = (f"IAW=[{ch['iaw'][0]:.2f},{ch['iaw'][1]:.2f}]"
                 if ch["iaw"] is not None else "no IAW")
        print(f"  {ch['channel']:>6s}  PAF={paf_s:>8s}  {q_s:>12s}  {iaw_s}")


# ── data I/O ─────────────────────────────────────────────────────────────────

def _channel_data(inst, picks, mne):
    """Return ``(ch_names, data)`` with ``data`` shape (n_channels, n_times)."""
    if picks is None:
        picks = [ch for ch, t in zip(inst.ch_names, inst.get_channel_types())
                 if t == "eeg"]
    elif not isinstance(picks[0], str):
        picks = [inst.ch_names[i] for i in picks]
    else:
        picks = list(picks)

    if isinstance(inst, mne.BaseEpochs):
        # (n_ep, n_ch, n_t) → concatenate epochs along time
        x = inst.get_data(picks=picks)
        data = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    else:
        data = inst.get_data(picks=picks)

    return picks, np.asarray(data, float)


# ── per-channel pipeline ─────────────────────────────────────────────────────

def _from_signal(signal, sfreq, n_fft, n_overlap, fw, k, wa, p_diff, freq_trim):
    """Welch → SGF IAF pipeline for one channel time series."""
    nperseg = min(int(n_fft), len(signal))
    noverlap = min(int(n_overlap), max(nperseg - 1, 0))
    freqs, psd = welch(signal, fs=sfreq, window="hann",
                       nperseg=nperseg, noverlap=noverlap)
    return _from_psd(freqs, psd, fw, k, wa, p_diff, freq_trim)


def _from_psd(freqs, psd, fw, k, wa, p_diff, freq_trim):
    """
    Core SGF peak / IAW / Q estimation on one spectrum.

    Returns a channel dict; ``paf`` / ``iaw`` are ``None`` if no peak is found.
    """
    freqs = np.asarray(freqs, float)
    psd = np.asarray(psd, float)
    keep = (freqs >= freq_trim[0]) & (freqs <= freq_trim[1])
    freqs, psd = freqs[keep], psd[keep]

    empty = _empty(freqs)
    if freqs.size < 5 or psd.mean() <= 0:
        return empty

    # Normalise and set minP = exp(regression + 1 SD residual) on log PSD
    psd_norm = psd / psd.mean()
    log_psd = np.log(psd_norm)
    slope, intercept, *_ = linregress(freqs, log_psd)
    predicted = slope * freqs + intercept
    min_p = np.exp(predicted + (log_psd - predicted).std())

    fw = fw if fw % 2 else fw + 1
    if fw >= len(psd_norm) or fw <= k:
        return empty

    smooth = savgol_filter(psd_norm, fw, k, deriv=0)
    d1 = savgol_filter(psd_norm, fw, k, deriv=1)
    d2 = savgol_filter(psd_norm, fw, k, deriv=2)

    wa_idx = np.where((freqs >= wa[0]) & (freqs <= wa[1]))[0]
    if wa_idx.size < 3:
        return empty

    # Peaks = downward zero-crossings of d1 inside Wa, above minP
    cands = _peak_indices(d1, wa_idx)
    cands = [i for i in cands if psd_norm[i] >= min_p[i]]
    peak = _dominant_peak(cands, psd_norm, p_diff)
    if peak is None:
        return empty

    i1, i2 = _inflections(d2, peak, wa_idx)
    f1, f2 = _iaw_bounds(d1, peak, len(freqs))
    iaw = (float(freqs[f1]), float(freqs[f2]))

    return {
        "paf": float(freqs[peak]),
        "q": _quality(psd_norm, freqs, i1, i2),
        "iaw": iaw,
        "paf_bounds": (float(freqs[i1]), float(freqs[i2])),
        "peak_height": float(psd[peak]),
        "peak_width_fwhm": _fwhm(smooth, freqs, peak, i1, i2),
        "freqs": freqs,
        "psd_norm": psd_norm,
        "psd_smooth": smooth,
        "psd_raw": psd,
    }


def _empty(freqs=None):
    return {
        "paf": None, "q": None, "iaw": None, "paf_bounds": None,
        "peak_height": None, "peak_width_fwhm": None,
        "freqs": freqs, "psd_norm": None, "psd_smooth": None, "psd_raw": None,
    }


# ── cross-channel means ──────────────────────────────────────────────────────

def _aggregate(channels, c_min):
    """Q-weighted PAF (Eq. 8); mean IAW → CoG (Eq. 5)."""
    with_paf = [r for r in channels if r["paf"] is not None]
    with_iaw = [r for r in channels if r["iaw"] is not None]

    paf_mean = np.nan
    if len(with_paf) >= c_min:
        pafs = np.array([r["paf"] for r in with_paf])
        q = np.array([r["q"] for r in with_paf], float)
        w = q / q.max()
        paf_mean = float(np.sum(pafs * w) / w.sum())

    iaw_mean, cog_mean = (np.nan, np.nan), np.nan
    if len(with_iaw) >= c_min:
        f1 = float(np.mean([r["iaw"][0] for r in with_iaw]))
        f2 = float(np.mean([r["iaw"][1] for r in with_iaw]))
        iaw_mean = (f1, f2)
        cogs = [_cog(r["psd_norm"], r["freqs"], f1, f2)
                for r in channels
                if r["freqs"] is not None and r["psd_norm"] is not None]
        cogs = [c for c in cogs if c is not None]
        if cogs:
            cog_mean = float(np.mean(cogs))

    heights = [r["peak_height"] for r in with_paf if r["peak_height"] is not None]
    widths = [r["peak_width_fwhm"] for r in with_paf if r["peak_width_fwhm"] is not None]

    return {
        "paf_mean": paf_mean,
        "cog_mean": cog_mean,
        "iaw_mean": iaw_mean,
        "peak_height_mean": float(np.mean(heights)) if heights else np.nan,
        "peak_width_mean": float(np.mean(widths)) if widths else np.nan,
        "channels": channels,
        "n_paf": len(with_paf),
        "n_cog": len(with_iaw),
    }


# ── peak / bound helpers ─────────────────────────────────────────────────────

def _peak_indices(d1, search_idx):
    """Indices of downward-going first-derivative zero crossings in Wa."""
    peaks = []
    for j in range(1, len(search_idx)):
        a, b = search_idx[j - 1], search_idx[j]
        if d1[a] > 0 and d1[b] < 0:
            peaks.append(a if abs(d1[a]) <= abs(d1[b]) else b)
    return peaks


def _dominant_peak(candidates, psd_norm, p_diff):
    """Tallest peak must beat the next by ``p_diff`` of its own height."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ranked = sorted(((psd_norm[i], i) for i in candidates), reverse=True)
    (p1, i1), (p2, _) = ranked[0], ranked[1]
    return i1 if (p1 - p2) / p1 >= p_diff else None


def _inflections(d2, peak, search_idx):
    """Nearest 2nd-derivative sign changes around the peak (else Wa edges)."""
    lo, hi = search_idx[0], search_idx[-1]
    i1 = next((i for i in range(peak - 1, lo - 1, -1) if d2[i] * d2[i + 1] <= 0), lo)
    i2 = next((i + 1 for i in range(peak + 1, hi + 1)
               if i + 1 < len(d2) and d2[i] * d2[i + 1] <= 0), hi)
    return i1, i2


def _iaw_bounds(d1, peak, n):
    """
    IAW edges: prefer local minimum in d1, else |d1| < 1 (Eq. 6–7).
    Falls back to spectrum edges.
    """
    f1 = 0
    for i in range(peak - 1, 0, -1):
        if d1[i - 1] < 0 < d1[i] or abs(d1[i]) < 1.0:
            f1 = i
            break
    f2 = n - 1
    for i in range(peak + 1, n - 1):
        if d1[i] < 0 < d1[i + 1] or abs(d1[i]) < 1.0:
            f2 = i + 1 if d1[i] < 0 < d1[i + 1] else i
            break
    return f1, f2


def _quality(psd_norm, freqs, i1, i2):
    """Q = ∫ PSD_norm df / (f2 − f1)  [Eq. 4]."""
    if i2 <= i1:
        return None
    span = freqs[i2] - freqs[i1]
    if span <= 0:
        return None
    return float(np.trapezoid(psd_norm[i1:i2 + 1], freqs[i1:i2 + 1]) / span)


def _cog(psd_norm, freqs, f1, f2):
    """Centre of gravity over [f1, f2]  [Eq. 5]."""
    m = (freqs >= f1) & (freqs <= f2)
    if m.sum() < 2:
        return None
    f, p = freqs[m], psd_norm[m]
    den = np.trapezoid(p, f)
    return None if den == 0 else float(np.trapezoid(p * f, f) / den)


def _fwhm(smooth, freqs, peak, i1, i2):
    """FWHM of the smoothed peak between inflection points [Hz], or None."""
    if i2 <= i1:
        return None
    seg_f = freqs[i1:i2 + 1]
    seg = smooth[i1:i2 + 1]
    loc = peak - i1
    if not (0 <= loc < len(seg)):
        return None

    peak_val = seg[loc]
    baseline = min(seg[0], seg[-1])
    if peak_val <= baseline:
        return None
    half = baseline + 0.5 * (peak_val - baseline)

    def _cross(xs, fs, start, step):
        for i in range(start, 0 if step < 0 else len(xs) - 1, step):
            a, b = (i - 1, i) if step < 0 else (i, i + 1)
            lo_v, hi_v = sorted((xs[a], xs[b]))
            if lo_v <= half <= hi_v:
                denom = xs[b] - xs[a]
                t = 0.0 if denom == 0 else (half - xs[a]) / denom
                return fs[a] + t * (fs[b] - fs[a])
        return None

    left = _cross(seg, seg_f, loc, -1)
    right = _cross(seg, seg_f, loc, 1)
    if left is None or right is None:
        return None
    return float(right - left)
