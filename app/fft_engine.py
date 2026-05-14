from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def compute_raw_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hanning-windowed single-sided FFT. Returns (freqs_hz, magnitude)."""
    n = len(signal)
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * win)) / (n / 2) / np.mean(win)
    spectrum[0] /= 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spectrum


def average_spectra_interp(
    spectra: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """RMS-average (freqs, mag) pairs onto a common grid."""
    if not spectra:
        return np.array([0.0]), np.array([0.0])
    if len(spectra) == 1:
        return spectra[0]
    valid = [(f, m) for f, m in spectra if len(f) > 1]
    if not valid:
        return np.array([0.0]), np.array([0.0])
    f_max = min(f[-1] for f, _ in valid)
    df = max(f[1] - f[0] for f, _ in valid)
    n_pts = max(2, int(f_max / df) + 1)
    common_freqs = np.linspace(0.0, f_max, n_pts)
    interped = []
    for freqs, mag in valid:
        fn = interp1d(freqs, mag, kind="linear", bounds_error=False, fill_value=0.0)
        interped.append(fn(common_freqs))
    stacked = np.vstack(interped)
    avg = np.sqrt(np.mean(stacked ** 2, axis=0))
    return common_freqs, avg
