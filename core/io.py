from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from .config import Config


def load_data(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith(".dxd"):
        df = _load_dxd(file_path)
    else:
        df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows  <- {os.path.basename(file_path)}")
    return df


def _load_dxd(file_path: str) -> pd.DataFrame:
    import h5py
    channels: dict[str, np.ndarray] = {}
    time_array: np.ndarray | None = None

    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                channels[key] = item[:]
            elif isinstance(item, h5py.Group):
                # DEWEsoft channel group: expect 'data' dataset; 'time' for timestamps
                if "data" in item:
                    channels[key] = item["data"][:]
                if time_array is None and "time" in item:
                    time_array = item["time"][:]
                elif time_array is None and "Time" in item:
                    time_array = item["Time"][:]

    if not channels:
        raise ValueError(f"No readable channel data found in DXD file: {os.path.basename(file_path)}")

    # Align all channels to the minimum length (channels may differ by 1 sample at EOF)
    min_len = min(len(v) for v in channels.values())
    df = pd.DataFrame({k: v[:min_len] for k, v in channels.items()})

    if time_array is not None and "time" not in df.columns and "Time" not in df.columns:
        df.insert(0, "time", time_array[:min_len])

    return df


def load_many(base_path: str, files: tuple) -> list[tuple[str, pd.DataFrame]]:
    results = []
    for fname in files:
        path = os.path.join(base_path, fname)
        if not os.path.exists(path):
            warnings.warn(f"File not found, skipping: {path}")
            continue
        results.append((fname, load_data(path)))
    return results


def check_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Return active axes after dropping any whose accel column is missing."""
    active_axes = []
    for ax in cfg.axes:
        col = cfg.accel_cols.get(ax)
        if col and col in df.columns:
            active_axes.append(ax)
        else:
            warnings.warn(f"Axis {ax}: column '{col}' not found — skipping.")
    for col in (cfg.time_col, cfg.rpm_col, cfg.angle_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")
    return active_axes


def extract_channels(df: pd.DataFrame, cfg: Config, active_axes: list[str]) -> dict:
    """Return dict of clean float arrays; NaN rows dropped consistently."""
    cols = [cfg.time_col, cfg.rpm_col, cfg.angle_col] + [cfg.accel_cols[ax] for ax in active_axes]
    clean = df[cols].dropna().reset_index(drop=True)
    ch = {
        'time':  clean[cfg.time_col].values.astype(float),
        'rpm':   clean[cfg.rpm_col].values.astype(float),
        'angle': clean[cfg.angle_col].values.astype(float),
    }
    for ax in active_axes:
        ch[ax] = clean[cfg.accel_cols[ax]].values.astype(float)
    return ch


def estimate_fs(time: np.ndarray) -> float:
    dt_all = np.diff(time)
    fs = 1.0 / np.median(dt_all)
    dt_min, dt_max = dt_all.min(), dt_all.max()
    if dt_max / dt_min > 10:
        warnings.warn(f"Large dt variation (min={dt_min:.6f}s, max={dt_max:.6f}s) — dropped samples?")
    return fs


def detect_encoder_ppr(angle_raw: np.ndarray) -> int:
    """Estimate encoder pulses-per-revolution from angle step size."""
    steps = np.abs(np.diff(angle_raw))
    valid = steps[steps > 0]
    if len(valid) == 0:
        return 1024
    effective_step = float(np.median(valid))
    if effective_step <= 0:
        return 1024
    return int(round(360.0 / effective_step))


def report_encoder_resolution(angle_raw: np.ndarray, cfg: Config) -> None:
    steps = np.abs(np.diff(angle_raw))
    effective_step = np.median(steps[steps > 0])
    effective_ppr  = round(360.0 / effective_step)
    print(f"  Encoder: effective step ~{effective_step:.4f}deg  ->  ~{effective_ppr} PPR")
    if cfg.samples_per_rev > effective_ppr * 1.1:
        warnings.warn(
            f"samples_per_rev={cfg.samples_per_rev} > encoder PPR~{effective_ppr}: "
            "resampling above encoder resolution — consider lowering samples_per_rev."
        )
