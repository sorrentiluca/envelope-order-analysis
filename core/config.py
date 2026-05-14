from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Config:
    # ── IO ────────────────────────────────────────────────────────────────────
    base_path: str = (
        r'C:\Users\sorrelca\OneDrive - Schaeffler\Accelerated Leadership Program'
        r'\Rotation 4 (Troy)\Ballscrew Testing\Test Data\PlasticvsMetal'
    )
    files: tuple = (
        '2300-30k-metal635-1.csv',
        '2300-30k-metal635-2.csv',
        '2300-30k-metal635-3.csv',
    )
    time_col: str  = 'Time (s)'
    rpm_col: str   = 'CNT 1/Frequency (RPM)'
    angle_col: str = 'CNT 1/Angle (Degrees)'
    accel_cols: dict = field(default_factory=lambda: {
        'X': 'X Accel (m/s2)',
        'Y': 'Y Accel (m/s2)',
        'Z': 'Z Accel (m/s2)',
    })
    axes: tuple = ('X', 'Y', 'Z')

    # ── SEGMENTATION ──────────────────────────────────────────────────────────
    rpm_threshold: float   = 100.0
    pre_window_s: float    = 0.15
    post_window_s: float   = 0.35
    min_event_gap_s: float = 0.5
    prefer_plateau: bool      = True
    plateau_speed_tol: float  = 0.05
    min_segment_rpm: float    = 200.0
    use_whole_file_segments: bool = False
    whole_file_speed_tol: float   = 0.05
    min_revs_per_segment: float   = 8.0
    direction: str = 'ALL'
    concatenate_matched: bool  = False
    rpm_match_tol: float       = 200.0

    # ── ANGLE RESAMPLING / SPECTRUM ───────────────────────────────────────────
    samples_per_rev: int       = 1024
    antialias: bool            = True
    antialias_filter_order: int = 6
    antialias_margin: float    = 0.9
    order_xlim: float          = 60.0
    common_order_resolution: float = 0.01
    detrend: bool              = True
    avg_mode: str              = 'rms'
    tsa_revs_per_average: int  = 1

    # ── ENVELOPE ──────────────────────────────────────────────────────────────
    envelope_enable: bool      = True
    envelope_band_hz: Optional[tuple] = None
    envelope_filter_order: int = 4
    kurtogram_enable: bool     = True
    kurtogram_n_levels: int    = 6

    # ── DETECTORS ─────────────────────────────────────────────────────────────
    detector_fmin_order: float  = 1.0
    detector_fmax_order: float  = 25.0
    detector_step_order: float  = 0.005
    detector_n_harmonics: int   = 6
    detector_combine: str       = 'logsum'
    noise_floor_percentile: float     = 40.0
    noise_floor_smooth_orders: float  = 2.0
    cepstrum_enable: bool       = True

    # ── BPF REFERENCE ─────────────────────────────────────────────────────────
    bpf_fundamental_order: float = 5.35
    n_bpf_harmonics: int         = 8

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    do_plots: bool   = True
    show_plots: bool = True
    verbose: bool    = True
    fs_override_hz: Optional[float] = None

    # ── CAMPBELL ──────────────────────────────────────────────────────────────
    campbell_revs_per_block: int = 4
    campbell_overlap: float      = 0.5
    campbell_order_xlim: float   = 60.0


@dataclass
class Segment:
    file_name: str
    i0: int
    i1: int
    direction: str
    mean_rpm: float
    total_revs: float
    stitched: bool = False


@dataclass
class SegmentResult:
    axis: str
    total_revs: float
    order_resolution: float
    mean_rpm: float
    orders: np.ndarray
    mag: np.ndarray
    env_mag: Optional[np.ndarray] = None
    band_hz: Optional[tuple]      = None
    tsa_orders: Optional[np.ndarray]   = None
    tsa_mag: Optional[np.ndarray]      = None
    residual_mag: Optional[np.ndarray] = None
    uniform_angle: Optional[np.ndarray] = None
    angle_accel: Optional[np.ndarray]   = None


@dataclass
class AggResult:
    axis: str
    n_events: int
    common_orders: np.ndarray
    avg_mag: np.ndarray
    noise_floor: np.ndarray
    snr: np.ndarray
    avg_env_mag: Optional[np.ndarray]   = None
    env_noise_floor: Optional[np.ndarray] = None
    env_snr: Optional[np.ndarray]       = None


@dataclass
class Candidate:
    order: float
    score: float
    source: str
    n_harmonics_snr3: int


@dataclass
class Results:
    agg: dict
    common_orders: np.ndarray
    combined_mag: np.ndarray
    combined_snr: np.ndarray
    combined_env_mag: Optional[np.ndarray]
    combined_env_snr: Optional[np.ndarray]
    raw_score_f: np.ndarray
    raw_score: np.ndarray
    env_score: Optional[np.ndarray]
    z_raw_score: Optional[np.ndarray]
    raw_cepstrum_q: Optional[np.ndarray]
    raw_cepstrum: Optional[np.ndarray]
    env_cepstrum_q: Optional[np.ndarray]
    env_cepstrum: Optional[np.ndarray]
    candidates: list
    campbell_X: Optional[np.ndarray]    = None
    campbell_Y_rpm: Optional[np.ndarray] = None
    campbell_Z_map: Optional[np.ndarray] = None
    campbell_axis: str                  = 'X'
    sample_time: Optional[np.ndarray]   = None
    sample_rpm: Optional[np.ndarray]    = None
    sample_accel: Optional[np.ndarray]  = None
    sample_uniform_angle: Optional[np.ndarray] = None
    sample_angle_accel: Optional[np.ndarray]   = None
    kurtogram_data: Optional[tuple]     = None
