from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def compute_order_spectrum(
    signal: np.ndarray, total_revs: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Hanning-windowed FFT -> (orders, magnitude, order_resolution)."""
    win   = np.hanning(len(signal))
    N     = len(signal)
    fft_m = np.abs(np.fft.rfft(signal * win)) / (N / 2) / np.mean(win)
    fft_m[0] /= 2
    order_res = 1.0 / total_revs
    orders    = np.arange(len(fft_m)) * order_res
    return orders, fft_m, order_res


def interpolate_to_common_orders(
    orders: np.ndarray, mag: np.ndarray, common_orders: np.ndarray
) -> np.ndarray:
    f = interp1d(orders, mag, kind='linear', bounds_error=False, fill_value=0.0)
    return f(common_orders)


def average_spectra_rms(spectra: list[np.ndarray]) -> np.ndarray:
    if not spectra:
        return np.zeros(1)
    stacked = np.vstack(spectra)
    return np.sqrt(np.mean(stacked ** 2, axis=0))


def time_synchronous_average(
    signal: np.ndarray, samples_per_rev: int, revs_per_avg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    block = samples_per_rev * revs_per_avg
    n_avg = len(signal) // block
    if n_avg < 1:
        orders, mag, _ = compute_order_spectrum(signal, float(revs_per_avg))
        return signal, orders, mag, 1
    trimmed = signal[:n_avg * block].reshape(n_avg, block)
    tsa     = trimmed.mean(axis=0)
    orders, mag, _ = compute_order_spectrum(tsa, float(revs_per_avg))
    return tsa, orders, mag, n_avg


def residual_after_tsa(signal: np.ndarray, tsa_waveform: np.ndarray) -> np.ndarray:
    block  = len(tsa_waveform)
    n_tile = int(np.ceil(len(signal) / block))
    tiled  = np.tile(tsa_waveform, n_tile)[:len(signal)]
    return signal - tiled
