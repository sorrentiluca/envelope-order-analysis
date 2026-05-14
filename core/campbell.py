from __future__ import annotations

import numpy as np


def compute_campbell(
    accel_resampled: np.ndarray,
    rpm_resampled: np.ndarray,
    samples_per_rev: int,
    revs_per_block: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (order_axis, rpm_axis_sorted, magnitude_map)."""
    block_size = int(samples_per_rev * revs_per_block)
    step_size  = max(1, int(block_size * (1.0 - overlap)))
    window     = np.hanning(block_size)
    order_spectra, block_rpms = [], []

    for i in range(0, len(accel_resampled) - block_size, step_size):
        block = accel_resampled[i:i + block_size]
        block = block - np.mean(block)
        fft_m = np.abs(np.fft.rfft(block * window)) / (block_size / 2) / np.mean(window)
        order_spectra.append(fft_m)
        block_rpms.append(float(np.mean(rpm_resampled[i:i + block_size])))

    if not order_spectra:
        return np.array([]), np.array([]), np.zeros((0, 0))

    Z      = np.vstack(order_spectra)
    Y_rpm  = np.array(block_rpms)
    sort_i = np.argsort(Y_rpm)
    order_res = 1.0 / revs_per_block
    X_orders  = np.arange(Z.shape[1]) * order_res
    return X_orders, Y_rpm[sort_i], Z[sort_i]
