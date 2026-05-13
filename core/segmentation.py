from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import Config, Segment


def unwrap_angle(angle_deg: np.ndarray) -> np.ndarray:
    """Cumulative +/-360 correction; handles shaft reversals."""
    diff = np.diff(angle_deg)
    corr = -360.0 * np.round(diff / 360.0)
    return np.concatenate([[angle_deg[0]], angle_deg[1:] + np.cumsum(corr)])


def sort_dedup_angle(angle: np.ndarray, *signals) -> tuple:
    """Sort by angle; drop duplicate angle samples. Returns (angle, *signals)."""
    sort_idx = np.argsort(angle, kind='stable')
    angle = angle[sort_idx]
    signals = tuple(s[sort_idx] for s in signals)
    _, unique_idx = np.unique(angle, return_index=True)
    return (angle[unique_idx],) + tuple(s[unique_idx] for s in signals)


def _longest_run(mask: np.ndarray) -> Optional[tuple]:
    if not np.any(mask):
        return None
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best])


def find_event_windows(time: np.ndarray, rpm: np.ndarray, cfg: Config) -> list[tuple]:
    """RPM-threshold up-crossings, debounced -> list of (i0, i1) index pairs."""
    above     = (np.abs(rpm) > cfg.rpm_threshold).astype(int)
    crossings = np.where(np.diff(above) == 1)[0]
    debounced, last_t = [], -np.inf
    for idx in crossings:
        if time[idx] - last_t >= cfg.min_event_gap_s:
            debounced.append(idx)
            last_t = time[idx]
    windows = []
    for idx in debounced:
        t0 = time[idx] - cfg.pre_window_s
        t1 = time[idx] + cfg.post_window_s
        i0 = np.searchsorted(time, t0)
        i1 = np.searchsorted(time, t1)
        if i1 - i0 > 4:
            windows.append((i0, int(min(i1, len(time) - 1))))
    return windows


def find_plateau_in_window(rpm_seg: np.ndarray, cfg: Config) -> Optional[tuple]:
    """Longest near-constant-RPM stretch within a short event window."""
    n = len(rpm_seg)
    if n < 10:
        return None
    smooth   = uniform_filter1d(np.abs(rpm_seg), size=max(3, n // 15))
    peak_rpm = smooth.max()
    if peak_rpm < cfg.min_segment_rpm:
        return None
    stable = (
        (np.abs(smooth - peak_rpm) / peak_rpm < cfg.plateau_speed_tol)
        & (smooth > cfg.min_segment_rpm)
    )
    return _longest_run(stable)


def find_constant_speed_segments(time: np.ndarray, rpm: np.ndarray, cfg: Config) -> list[tuple]:
    """Scan entire array for contiguous spans where RPM is stable."""
    smooth = uniform_filter1d(np.abs(rpm), size=max(3, len(rpm) // 200))
    above  = smooth > cfg.min_segment_rpm
    if not np.any(above):
        return []
    win        = max(20, len(smooth) // 100)
    local_mean = uniform_filter1d(smooth, size=win)
    local_std  = np.sqrt(uniform_filter1d((smooth - local_mean) ** 2, size=win))
    cv     = np.where(local_mean > 0, local_std / local_mean, 1.0)
    stable = above & (cv < cfg.whole_file_speed_tol)
    windows = []
    padded  = np.concatenate([[False], stable, [False]])
    changes = np.diff(padded.astype(int))
    starts  = np.where(changes == 1)[0]
    ends    = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        if e - s > 4:
            windows.append((int(s), int(e)))
    return windows


def _revs_in_window(angle_unwrapped: np.ndarray, i0: int, i1: int) -> float:
    return abs(angle_unwrapped[i1 - 1] - angle_unwrapped[i0]) / 360.0


def pick_segments(
    time: np.ndarray,
    angle_unwrapped: np.ndarray,
    rpm: np.ndarray,
    file_name: str,
    cfg: Config,
) -> list[Segment]:
    candidates = []
    if cfg.use_whole_file_segments:
        for i0, i1 in find_constant_speed_segments(time, rpm, cfg):
            candidates.append((i0, i1))
    else:
        for i0, i1 in find_event_windows(time, rpm, cfg):
            seg_rpm = rpm[i0:i1]
            if cfg.prefer_plateau:
                plateau = find_plateau_in_window(seg_rpm, cfg)
                if plateau is not None:
                    p0, p1 = plateau
                    candidates.append((i0 + p0, i0 + p1))
                else:
                    candidates.append((i0, i1))
            else:
                candidates.append((i0, i1))

    segments, skipped = [], 0
    for i0, i1 in candidates:
        total_revs = _revs_in_window(angle_unwrapped, i0, i1)
        if total_revs < cfg.min_revs_per_segment:
            skipped += 1
            continue
        angle_span = angle_unwrapped[i1 - 1] - angle_unwrapped[i0]
        direction  = 'POS' if angle_span >= 0 else 'NEG'
        if cfg.direction == 'POSITIVE' and direction != 'POS':
            continue
        if cfg.direction == 'NEGATIVE' and direction != 'NEG':
            continue
        mean_rpm = float(np.mean(np.abs(rpm[i0:i1])))
        segments.append(Segment(file_name, i0, i1, direction, mean_rpm, total_revs))

    if cfg.verbose:
        print(f"  Segments: {len(segments)} kept, {skipped} discarded (< {cfg.min_revs_per_segment:.1f} rev)")
    return segments


def concatenate_matched_segments(
    segments: list[Segment],
    channels: dict,
    active_axes: list[str],
    cfg: Config,
) -> list[Segment]:
    if not segments:
        return segments
    rpm_vals = np.array([s.mean_rpm for s in segments])
    bins     = np.arange(0, rpm_vals.max() + cfg.rpm_match_tol, cfg.rpm_match_tol)
    bin_ids  = np.digitize(rpm_vals, bins)
    extra = []
    for b in np.unique(bin_ids):
        group = [s for s, bid in zip(segments, bin_ids) if bid == b and not s.stitched]
        if len(group) < 2:
            continue
        total_revs = sum(s.total_revs for s in group)
        mean_rpm   = float(np.mean([s.mean_rpm for s in group]))
        rep = group[0]
        stitched_seg = Segment(
            file_name=rep.file_name, i0=-1, i1=-1,
            direction=rep.direction, mean_rpm=mean_rpm,
            total_revs=total_revs, stitched=True,
        )
        stitched_seg._constituents = [(s.i0, s.i1) for s in group]  # type: ignore[attr-defined]
        extra.append(stitched_seg)
        if cfg.verbose:
            print(f"  Stitched {len(group)} segments at ~{mean_rpm:.0f} RPM -> {total_revs:.1f} rev")
    return segments + extra


def extract_segment_signal(
    channels: dict, seg: Segment, axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (angle, accel, time, rpm) for a segment."""
    if seg.stitched:
        parts_a, parts_s, parts_t, parts_r = [], [], [], []
        for i0, i1 in seg._constituents:  # type: ignore[attr-defined]
            parts_a.append(channels['angle'][i0:i1])
            parts_s.append(channels[axis][i0:i1])
            parts_t.append(channels['time'][i0:i1])
            parts_r.append(channels['rpm'][i0:i1])
        angle_pieces, offset = [], 0.0
        for a in parts_a:
            uw = unwrap_angle(a)
            angle_pieces.append(uw - uw[0] + offset)
            offset = angle_pieces[-1][-1] + 360.0 / 1024.0
        return (
            np.concatenate(angle_pieces),
            np.concatenate(parts_s),
            np.concatenate(parts_t),
            np.concatenate(parts_r),
        )
    i0, i1 = seg.i0, seg.i1
    return (
        channels['angle'][i0:i1],
        channels[axis][i0:i1],
        channels['time'][i0:i1],
        channels['rpm'][i0:i1],
    )
