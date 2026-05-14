from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class SegmentWindow:
    t_start: float
    t_end: float
    i_start: int
    i_end: int
    mean_rpm: float
    plateau_t_start: Optional[float] = None
    plateau_t_end: Optional[float] = None
    enabled: bool = True


def _find_plateau(rpm_seg: np.ndarray, speed_tol: float = 0.05) -> Optional[tuple]:
    n = len(rpm_seg)
    if n < 10:
        return None
    smooth = uniform_filter1d(np.abs(rpm_seg), size=max(3, n // 15))
    peak_rpm = smooth.max()
    if peak_rpm < 10.0:
        return None
    stable = np.abs(smooth - peak_rpm) / peak_rpm < speed_tol
    padded = np.concatenate([[False], stable, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) == 0:
        return None
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def detect_events(
    time: np.ndarray,
    rpm: np.ndarray,
    rpm_threshold: float = 100.0,
    min_duration: float = 0.1,
    speed_tol: float = 0.05,
) -> list[SegmentWindow]:
    """Return contiguous spans where |rpm| > threshold, with plateau sub-windows."""
    above = np.abs(rpm) > rpm_threshold
    padded = np.concatenate([[False], above, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    windows: list[SegmentWindow] = []
    for s, e in zip(starts, ends):
        e = min(int(e), len(time) - 1)
        s = int(s)
        if time[e] - time[s] < min_duration:
            continue
        rpm_seg = rpm[s:e]
        w = SegmentWindow(
            t_start=float(time[s]),
            t_end=float(time[e]),
            i_start=s,
            i_end=e,
            mean_rpm=float(np.mean(np.abs(rpm_seg))),
        )
        plateau = _find_plateau(rpm_seg, speed_tol=speed_tol)
        if plateau is not None:
            p0, p1 = plateau
            p1 = min(p1, len(rpm_seg) - 1)
            w.plateau_t_start = float(time[s + p0])
            w.plateau_t_end = float(time[s + p1])
        windows.append(w)
    return windows
