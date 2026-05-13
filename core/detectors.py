from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .config import Config, Candidate


def fit_noise_floor(orders: np.ndarray, mag: np.ndarray, cfg: Config) -> np.ndarray:
    """Rolling percentile in log10 space -> smooth noise baseline."""
    log_mag  = np.log10(np.maximum(mag, 1e-30))
    win_pts  = max(3, int(cfg.noise_floor_smooth_orders / (orders[1] - orders[0] + 1e-30)))
    pad      = (win_pts - 1) // 2
    padded   = np.pad(log_mag, (pad, win_pts - 1 - pad), mode='edge')
    windows  = np.lib.stride_tricks.sliding_window_view(padded, win_pts)
    floor_log = uniform_filter1d(
        np.percentile(windows, cfg.noise_floor_percentile, axis=1),
        size=win_pts,
    )
    return 10.0 ** floor_log[:len(log_mag)]


def snr_spectrum(mag: np.ndarray, floor: np.ndarray) -> np.ndarray:
    return mag / np.maximum(floor, 1e-30)


def harmonic_sum_score(
    orders: np.ndarray, snr: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    f_grid = np.arange(cfg.detector_fmin_order, cfg.detector_fmax_order, cfg.detector_step_order)
    score  = np.zeros(len(f_grid))
    for i, f in enumerate(f_grid):
        vals, n_harm = [], 0
        for k in range(1, cfg.detector_n_harmonics + 1):
            target = k * f
            if target > cfg.order_xlim:
                break
            n_harm += 1
            snr_at = float(np.interp(target, orders, snr, left=0.0, right=0.0))
            vals.append(max(snr_at, 0.0))
        if n_harm == 0:
            continue
        if cfg.detector_combine == 'logsum':
            raw_score = sum(np.log1p(v) for v in vals)
        elif cfg.detector_combine == 'product':
            raw_score = float(np.prod([v + 1.0 for v in vals]))
        else:
            raw_score = sum(vals)
        score[i] = raw_score / n_harm
    return f_grid, score


def real_cepstrum(orders: np.ndarray, mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real cepstrum; a harmonic family at order f0 -> rahmonic at quefrency 1/f0 revs."""
    log_mag   = np.log(np.maximum(mag, 1e-30))
    cep       = np.fft.irfft(log_mag)
    n         = len(cep)
    order_res = orders[1] - orders[0] if len(orders) > 1 else 1.0
    order_max = orders[-1] if len(orders) > 0 else 1.0
    quefrency = np.arange(n) / (order_max + order_res)
    return quefrency[:n // 2], cep[:n // 2]


def detect_fundamentals(
    f_grid: np.ndarray,
    score: np.ndarray,
    orders: np.ndarray,
    snr: np.ndarray,
    source: str,
    cfg: Config,
) -> list[Candidate]:
    if np.all(score == 0):
        return []
    min_dist = max(1, int(0.3 / cfg.detector_step_order))
    peak_idx, _ = find_peaks(
        score,
        height=np.mean(score) + np.std(score) * 0.5,
        distance=min_dist,
    )
    if len(peak_idx) == 0:
        return []
    peak_idx = sorted(peak_idx, key=lambda i: score[i], reverse=True)
    accepted: list[Candidate] = []
    for pi in peak_idx:
        f = float(f_grid[pi])
        redundant = False
        for acc in accepted:
            ratio = f / acc.order
            if abs(ratio - round(ratio)) < 0.05 and round(ratio) >= 2:
                redundant = True
                break
            ratio2 = acc.order / f
            if abs(ratio2 - round(ratio2)) < 0.05 and round(ratio2) >= 2:
                redundant = True
                break
        if redundant:
            continue
        n_h3 = 0
        for k in range(1, cfg.detector_n_harmonics + 1):
            target = k * f
            if target > cfg.order_xlim:
                break
            if float(np.interp(target, orders, snr)) >= 3.0:
                n_h3 += 1
        accepted.append(Candidate(order=round(f, 4), score=float(score[pi]),
                                   source=source, n_harmonics_snr3=n_h3))
        if len(accepted) >= 10:
            break
    return accepted
