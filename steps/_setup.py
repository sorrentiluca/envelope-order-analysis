"""Shared loader used by all step scripts.

Call load_event(FILE_PATH, EVENT_INDEX, AXIS) to get a dict containing
every raw array needed by any step: full file channels, event window
slices, and plateau window slices.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

# Ensure repo root is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.io import load_data, check_columns, extract_channels, estimate_fs, report_encoder_resolution
from core.segmentation import unwrap_angle, find_event_windows, find_constant_speed_segments


def load_event(file_path: str, event_index: int, axis: str, cfg: Config = None) -> dict:
    """Load one CSV, detect events, select one event and its plateau window.

    Returns a dict with keys:
      cfg, df, channels, active_axes, axis, time, rpm, angle (unwrapped), fs,
      n_events, event_index,
      i0_ev/i1_ev, t_ev, r_ev, ang_ev, accel_ev, axis_data_ev,
      i0_pl/i1_pl, t_pl, r_pl, ang_pl, accel_pl, axis_data_pl,
      plateau_found
    """
    if cfg is None:
        cfg = Config()

    df          = load_data(file_path)
    active_axes = check_columns(df, cfg)
    if axis not in active_axes:
        warnings.warn(f"Axis {axis} not found; falling back to {active_axes[0]}")
        axis = active_axes[0]

    channels = extract_channels(df, cfg, active_axes)
    time  = channels['time']
    rpm   = channels['rpm']
    angle = channels['angle']
    fs    = estimate_fs(time)
    report_encoder_resolution(angle, cfg)

    angle_uw = unwrap_angle(angle)

    # ── Event detection ───────────────────────────────────────────────────────
    windows = find_event_windows(time, rpm, cfg)
    if not windows:
        raise RuntimeError("No events found.  Check rpm_threshold in Config.")
    if event_index >= len(windows):
        print(f"  EVENT_INDEX={event_index} out of range ({len(windows)} events); using 0")
        event_index = 0

    i0_ev, i1_ev = windows[event_index]
    t_ev   = time[i0_ev:i1_ev]
    r_ev   = rpm[i0_ev:i1_ev]
    ang_ev = angle_uw[i0_ev:i1_ev]

    # ── Plateau detection within the event ───────────────────────────────────
    t_slice = t_ev - t_ev[0]
    css = find_constant_speed_segments(t_slice, r_ev, cfg)
    if css:
        best_p0, best_p1 = max(css, key=lambda s: s[1] - s[0])
        i0_pl, i1_pl = i0_ev + best_p0, i0_ev + best_p1
        plateau_found = True
    else:
        i0_pl, i1_pl = i0_ev, i1_ev
        plateau_found = False

    t_pl   = time[i0_pl:i1_pl]
    r_pl   = rpm[i0_pl:i1_pl]
    ang_pl = angle_uw[i0_pl:i1_pl]

    axis_data_pl = {ax: channels[ax][i0_pl:i1_pl] for ax in active_axes}
    axis_data_ev = {ax: channels[ax][i0_ev:i1_ev] for ax in active_axes}

    return dict(
        cfg=cfg,
        df=df,
        channels=channels,
        active_axes=active_axes,
        axis=axis,
        time=time,
        rpm=rpm,
        angle=angle_uw,
        fs=fs,
        n_events=len(windows),
        event_index=event_index,
        # event window
        i0_ev=i0_ev, i1_ev=i1_ev,
        t_ev=t_ev, r_ev=r_ev, ang_ev=ang_ev,
        accel_ev=axis_data_ev[axis],
        axis_data_ev=axis_data_ev,
        # plateau / analysis window
        i0_pl=i0_pl, i1_pl=i1_pl,
        t_pl=t_pl, r_pl=r_pl, ang_pl=ang_pl,
        accel_pl=axis_data_pl[axis],
        axis_data_pl=axis_data_pl,
        plateau_found=plateau_found,
    )
