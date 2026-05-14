from __future__ import annotations

from typing import Optional

import numpy as np

from core.config import Config
from core.resampling import resample_to_uniform_angle
from core.spectrum import compute_order_spectrum, interpolate_to_common_orders, average_spectra_rms


def compute_order_spectrum_from_arrays(
    angle: np.ndarray,
    accel: np.ndarray,
    time: np.ndarray,
    rpm: np.ndarray,
    fs: float,
    samples_per_rev: int = 1024,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Returns (orders, magnitude) or None if resampling fails."""
    cfg = Config()
    cfg.samples_per_rev = samples_per_rev
    result = resample_to_uniform_angle(angle, accel, time, rpm, fs, cfg)
    if result is None:
        return None
    _, accel_res, _, total_revs, _ = result
    orders, mag, _ = compute_order_spectrum(accel_res, total_revs)
    return orders, mag


def average_order_spectra(
    spectra: list[tuple[np.ndarray, np.ndarray]],
    order_max: float = 100.0,
    order_res: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate to common order grid and RMS average."""
    common = np.arange(0.0, order_max + order_res, order_res)
    if not spectra:
        return common, np.zeros_like(common)
    interped = [interpolate_to_common_orders(o, m, common) for o, m in spectra]
    return common, average_spectra_rms(interped)
