from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from .config import Config


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows  <- {os.path.basename(file_path)}")
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
