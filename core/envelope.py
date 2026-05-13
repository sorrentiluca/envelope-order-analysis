from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, stft

from .config import Config
from .resampling import resample_to_uniform_angle
from .spectrum import compute_order_spectrum


def bandpass_filter(
    signal: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int = 4
) -> np.ndarray:
    nyq  = fs / 2.0
    f_lo = max(f_lo, 1.0)
    f_hi = min(f_hi, nyq * 0.95)
    if f_lo >= f_hi:
        raise ValueError(f"Invalid band: f_lo={f_lo:.1f} >= f_hi={f_hi:.1f}")
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return filtfilt(b, a, signal)


def hilbert_envelope(signal_band: np.ndarray) -> np.ndarray:
    env = np.abs(hilbert(signal_band))
    return env - np.mean(env)


def _stft_spectral_kurtosis(signal: np.ndarray, fs: float, nperseg: int) -> tuple:
    noverlap = nperseg // 2
    f, _, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     window='hann', return_onesided=True)
    mag2 = np.abs(Zxx) ** 2
    mag4 = mag2 ** 2
    m2   = np.mean(mag2, axis=1)
    m4   = np.mean(mag4, axis=1)
    eps  = 1e-30
    sk   = np.clip(m4 / (m2 ** 2 + eps) - 2.0, -2.0, 100.0)
    bw   = f[1] - f[0] if len(f) > 1 else fs / 2.0
    return f, sk, bw


def fast_kurtogram(signal: np.ndarray, fs: float, n_levels: int = 6) -> tuple:
    """Returns (kurt_rows, f_rows, bw_rows, best) where best=(f_lo, f_hi, level, kurtosis)."""
    nfft_sizes = [max(32, int(len(signal) // (2 ** i))) for i in range(1, n_levels + 1)]
    nfft_sizes = sorted({min(n, len(signal) // 4) for n in nfft_sizes}, reverse=True)

    kurt_rows, f_rows, bw_rows = [], [], []
    best_kurt = -np.inf
    best = (100.0, 1000.0, 0, 0.0)

    for lvl_i, nperseg in enumerate(nfft_sizes):
        try:
            f, sk, bw = _stft_spectral_kurtosis(signal, fs, nperseg)
        except Exception:
            continue
        kurt_rows.append(sk)
        f_rows.append(f)
        bw_rows.append(bw)
        valid = (f > bw * 2) & (f < fs * 0.45)
        if not np.any(valid):
            continue
        sk_v   = sk[valid]
        f_v    = f[valid]
        peak_i = int(np.argmax(sk_v))
        if sk_v[peak_i] > best_kurt:
            best_kurt = float(sk_v[peak_i])
            fc   = float(f_v[peak_i])
            best = (max(0.0, fc - bw), min(fs / 2.0, fc + bw), lvl_i, best_kurt)

    return kurt_rows, f_rows, bw_rows, best


def auto_select_band(
    signal: np.ndarray, fs: float, cfg: Config, max_rpm: float
) -> tuple[float, float]:
    if cfg.envelope_band_hz is not None:
        return cfg.envelope_band_hz

    max_rev_per_s = max(max_rpm / 60.0, 1.0)
    min_bw = cfg.n_bpf_harmonics * cfg.bpf_fundamental_order * max_rev_per_s * 2.0

    if cfg.kurtogram_enable:
        _, _, _, best = fast_kurtogram(signal, fs, cfg.kurtogram_n_levels)
        f_lo, f_hi, _, kurt_val = best
        if f_hi - f_lo < min_bw:
            fc   = (f_lo + f_hi) / 2.0
            f_lo = max(fc - min_bw / 2.0, 10.0)
            f_hi = min(fc + min_bw / 2.0, fs * 0.45)
        f_lo = max(f_lo, 10.0)
        f_hi = min(f_hi, fs * 0.45)
        if cfg.verbose:
            print(f"  Kurtogram -> band {f_lo:.0f}-{f_hi:.0f} Hz  (SK_max={kurt_val:.2f})")
        return f_lo, f_hi

    f_lo = max(min_bw, 500.0)
    f_hi = min(fs * 0.4, f_lo * 8.0)
    return f_lo, f_hi


def envelope_order_spectrum(
    accel_time: np.ndarray,
    angle: np.ndarray,
    rpm: np.ndarray,
    time: np.ndarray,
    fs: float,
    f_lo: float,
    f_hi: float,
    cfg: Config,
) -> Optional[tuple]:
    """Band-pass -> Hilbert envelope -> angle resample -> FFT.
    Returns (orders, magnitude, band_used) or None."""
    try:
        sig_band = bandpass_filter(accel_time, fs, f_lo, f_hi, cfg.envelope_filter_order)
        env      = hilbert_envelope(sig_band)
        res = resample_to_uniform_angle(angle, env, time, rpm, fs, cfg)
        if res is None:
            return None
        _, env_angle, _, total_revs, _ = res
        orders, mag, _ = compute_order_spectrum(env_angle, total_revs)
        return orders, mag, (f_lo, f_hi)
    except Exception as exc:
        warnings.warn(f"Envelope analysis failed: {exc}")
        return None
