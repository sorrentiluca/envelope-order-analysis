from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentWindow:
    t_start: float
    t_end: float
    i_start: int
    i_end: int
    mean_trigger: float
    enabled: bool = True


def detect_events(
    time: np.ndarray,
    trigger: np.ndarray,
    threshold: float = 100.0,
    pre_window_s: float = 0.0,
    post_window_s: float = 0.0,
) -> list[SegmentWindow]:
    """Find spans where |trigger| > threshold, optionally padded by pre/post seconds."""
    above = np.abs(trigger) > threshold
    padded = np.concatenate([[False], above, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    windows: list[SegmentWindow] = []
    for s, e in zip(starts, ends):
        s = int(s)
        e = min(int(e), len(time) - 1)
        t_s = max(float(time[0]), float(time[s]) - pre_window_s)
        t_e = min(float(time[-1]), float(time[e]) + post_window_s)
        i0 = int(np.searchsorted(time, t_s))
        i1 = int(np.searchsorted(time, t_e))
        if i1 - i0 < 4:
            continue
        trig_seg = trigger[i0:i1]
        windows.append(SegmentWindow(
            t_start=t_s,
            t_end=t_e,
            i_start=i0,
            i_end=i1,
            mean_trigger=float(np.mean(np.abs(trig_seg))),
        ))
    return windows
