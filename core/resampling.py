from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from .config import Config
from .segmentation import unwrap_angle, sort_dedup_angle


def antialias_lowpass(
    signal: np.ndarray, fs: float, max_rev_per_s: float, cfg: Config
) -> np.ndarray:
    cutoff = cfg.antialias_margin * (cfg.samples_per_rev / 2.0) * max_rev_per_s
    nyq    = fs / 2.0
    if cutoff >= nyq:
        return signal
    Wn = cutoff / nyq
    b, a = butter(cfg.antialias_filter_order, Wn, btype='low')
    return filtfilt(b, a, signal)


def resample_to_uniform_angle(
    angle: np.ndarray,
    accel: np.ndarray,
    time: np.ndarray,
    rpm: np.ndarray,
    fs: float,
    cfg: Config,
) -> Optional[tuple]:
    """Returns (uniform_angle, accel_resampled, rpm_resampled, total_revs, order_res) or None."""
    angle = unwrap_angle(angle)
    result = sort_dedup_angle(angle, accel, time, rpm)
    angle, accel, time_s, rpm_s = result[0], result[1], result[2], result[3]

    total_revs = abs((angle[-1] - angle[0]) / 360.0)
    if total_revs < 0.5:
        return None

    max_rev_per_s = max(np.abs(rpm_s).max() / 60.0, 1e-3)
    if cfg.antialias:
        accel = antialias_lowpass(accel, fs, max_rev_per_s, cfg)

    n = max(4, int(round(total_revs * cfg.samples_per_rev)))
    uniform_angle = np.linspace(angle[0], angle[-1], n)

    interp_fn = interp1d(angle, accel, kind='linear', bounds_error=False, fill_value=0.0)
    accel_res = interp_fn(uniform_angle)

    rpm_fn  = interp1d(angle, rpm_s, kind='linear', bounds_error=False, fill_value=0.0)
    rpm_res = rpm_fn(uniform_angle)

    if cfg.detrend:
        accel_res -= np.mean(accel_res)

    order_res = 1.0 / total_revs
    return uniform_angle, accel_res, rpm_res, total_revs, order_res
