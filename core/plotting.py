from __future__ import annotations

from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .config import Config, Results


AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}


def bpf_reference_orders(cfg: Config) -> list[float]:
    return [cfg.bpf_fundamental_order * k for k in range(1, cfg.n_bpf_harmonics + 1)]


def add_bpf_reference_lines(ax, cfg: Config, ymax: float) -> None:
    for i, order in enumerate(bpf_reference_orders(cfg)):
        if order > cfg.order_xlim:
            break
        ax.axvline(order, color='green', lw=0.8, linestyle=':', alpha=0.6,
                   label='BPF reference' if i == 0 else None)
        ax.text(order, ymax * 0.98, f' {order:.2f}', color='green',
                fontsize=6, va='top', ha='left', rotation=90)


def add_bpf_vlines(ax, cfg: Config, ymax: float) -> None:
    """Alias used by step scripts."""
    add_bpf_reference_lines(ax, cfg, ymax)


def annotate_box(ax, text: str, x: float = 0.02, y: float = 0.97, fontsize: int = 9) -> None:
    """Add a yellow explanation box to an axes."""
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffacc',
                  edgecolor='#ccaa00', alpha=0.92),
        zorder=10,
    )


def plot_raw_and_angle_waveform(results: Results, cfg: Config) -> Optional[plt.Figure]:
    if results.sample_time is None:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(16, 7))
    ax = axes[0]
    ax.plot(results.sample_time, results.sample_accel, lw=0.5, color='steelblue')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Acceleration (m/s2)')
    ax.set_title('Raw Accelerometer Signal (sample event)')
    ax2 = ax.twinx()
    ax2.plot(results.sample_time, results.sample_rpm, lw=0.8, color='grey', alpha=0.5)
    ax2.set_ylabel('RPM', color='grey')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax = axes[1]
    if results.sample_uniform_angle is not None:
        ax.plot(results.sample_uniform_angle / 360.0, results.sample_angle_accel,
                lw=0.5, color='darkorange')
        ax.set_xlabel('Revolutions'); ax.set_ylabel('Acceleration (m/s2)')
        ax.set_title('Angle-Domain Resampled Signal')
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_order_spectrum(results: Results, cfg: Config) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 6))
    for axis, agg in results.agg.items():
        ax.plot(agg.common_orders, agg.avg_mag,
                color=AXIS_COLORS.get(axis, 'grey'), lw=0.8, alpha=0.35, linestyle='--',
                label=f'{axis}  (n={agg.n_events})')
    ax.plot(results.common_orders, results.combined_mag,
            color='#222', lw=1.3, alpha=0.9, label='Combined')
    ymax = float(results.combined_mag.max()) * 1.2 or 1.0
    add_bpf_reference_lines(ax, cfg, ymax)
    ax.set_xlim(0, cfg.order_xlim); ax.set_ylim(0, ymax)
    ax.set_xlabel('Order (events per revolution)', fontsize=11)
    ax.set_ylabel('Magnitude (m/s2)', fontsize=11)
    ax.set_title(f'Order Spectrum - Raw  BPF reference: {cfg.bpf_fundamental_order} order',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    f0 = cfg.bpf_fundamental_order
    x1, x2 = max(0, f0 - 3), min(cfg.order_xlim, f0 + 10)
    axins = ax.inset_axes([0.02, 0.55, 0.22, 0.40])
    for axis, agg in results.agg.items():
        axins.plot(agg.common_orders, agg.avg_mag,
                   color=AXIS_COLORS.get(axis, 'grey'), lw=0.7, alpha=0.4)
    axins.plot(results.common_orders, results.combined_mag, color='#222', lw=1.0)
    add_bpf_reference_lines(axins, cfg, float(results.combined_mag.max()) * 1.1 or 1.0)
    axins.set_xlim(x1, x2)
    axins.set_title(f'Zoom: {x1:.0f}-{x2:.0f} orders', fontsize=7)
    axins.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_envelope_order_spectrum(results: Results, cfg: Config) -> Optional[plt.Figure]:
    if results.combined_env_mag is None:
        return None
    fig, ax = plt.subplots(figsize=(18, 6))
    for axis, agg in results.agg.items():
        if agg.avg_env_mag is not None:
            ax.plot(agg.common_orders, agg.avg_env_mag,
                    color=AXIS_COLORS.get(axis, 'grey'), lw=0.8, alpha=0.35, linestyle='--',
                    label=f'{axis} envelope')
    ax.plot(results.common_orders, results.combined_env_mag,
            color='darkred', lw=1.3, alpha=0.9, label='Combined envelope')
    ymax = float(results.combined_env_mag.max()) * 1.2 or 1.0
    add_bpf_reference_lines(ax, cfg, ymax)
    ax.set_xlim(0, cfg.order_xlim); ax.set_ylim(0, ymax)
    ax.set_xlabel('Order (events per revolution)', fontsize=11)
    ax.set_ylabel('Envelope magnitude (m/s2)', fontsize=11)
    ax.set_title('Envelope Order Spectrum  <- VERDICT PLOT  '
                 f'(BPF family at {cfg.bpf_fundamental_order} order should appear here)',
                 fontsize=12, fontweight='bold', color='darkred')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    f0 = cfg.bpf_fundamental_order
    x1, x2 = max(0, f0 - 2), min(cfg.order_xlim, f0 + 12)
    axins = ax.inset_axes([0.02, 0.55, 0.22, 0.40])
    axins.plot(results.common_orders, results.combined_env_mag, color='darkred', lw=1.0)
    add_bpf_reference_lines(axins, cfg, float(results.combined_env_mag.max()) * 1.1 or 1.0)
    axins.set_xlim(x1, x2)
    axins.set_title(f'Zoom: {x1:.0f}-{x2:.0f} orders', fontsize=7)
    axins.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_kurtogram(results: Results, cfg: Config) -> Optional[plt.Figure]:
    if results.kurtogram_data is None:
        return None
    kurt_rows, f_rows, bw_rows, best = results.kurtogram_data
    if not kurt_rows:
        return None
    fig, ax = plt.subplots(figsize=(14, 5))
    cmap_k   = cm.get_cmap('hot_r')
    all_kurt = np.concatenate(kurt_rows)
    vmin, vmax = 0.0, float(np.percentile(all_kurt, 99))
    sc = None
    for i, (sk, f, bw) in enumerate(zip(kurt_rows, f_rows, bw_rows)):
        sc = ax.scatter(f, np.full_like(f, i), c=sk, cmap=cmap_k,
                        vmin=vmin, vmax=vmax, s=15, marker='s', alpha=0.8)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label='Spectral Kurtosis')
    ax.axvline(best[0], color='cyan', lw=2, linestyle='--', label=f'Band lo {best[0]:.0f} Hz')
    ax.axvline(best[1], color='cyan', lw=2, linestyle='-',  label=f'Band hi {best[1]:.0f} Hz')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Level (largest window = level 0)', fontsize=11)
    ax.set_title(f'Kurtogram - selected band {best[0]:.0f}-{best[1]:.0f} Hz  (SK={best[3]:.2f})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_campbell(results: Results, cfg: Config) -> Optional[plt.Figure]:
    if results.campbell_X is None or results.campbell_Z_map is None:
        return None
    X, Y, Z = results.campbell_X, results.campbell_Y_rpm, results.campbell_Z_map
    fig, ax = plt.subplots(figsize=(16, 7))
    vmax_val = float(np.percentile(Z, 98)) if Z.size > 0 else 1.0
    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap='jet', vmin=0, vmax=vmax_val)
    plt.colorbar(mesh, ax=ax, label=f'Magnitude (m/s2) - axis: {results.campbell_axis}')
    for order in bpf_reference_orders(cfg):
        if order <= cfg.campbell_order_xlim:
            ax.axvline(order, color='white', lw=0.8, linestyle=':', alpha=0.7)
    ax.set_xlim(0, cfg.campbell_order_xlim)
    ax.set_xlabel('Order (events per revolution)', fontsize=11)
    ax.set_ylabel('Speed (RPM)', fontsize=11)
    ax.set_title(f'Order-RPM Heatmap (Campbell) - {results.campbell_axis} axis\n'
                 'White dotted lines = BPF reference orders',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_detector_score(results: Results, cfg: Config) -> plt.Figure:
    has_cep = results.raw_cepstrum is not None
    n_rows  = 2 if has_cep else 1
    fig, axes_list = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes_list = [axes_list]
    ax = axes_list[0]
    ax.plot(results.raw_score_f, results.raw_score,
            color='steelblue', lw=1.3, label='Harmonic-sum score (raw combined)')
    if results.env_score is not None:
        ax.plot(results.raw_score_f, results.env_score,
                color='darkred', lw=1.3, alpha=0.8, label='Harmonic-sum score (envelope combined)')
    if results.z_raw_score is not None:
        ax.plot(results.raw_score_f, results.z_raw_score,
                color=AXIS_COLORS.get('Z', 'green'), lw=1.0, alpha=0.7, linestyle='--',
                label='Harmonic-sum score (Z raw)')
    ax.axvline(cfg.bpf_fundamental_order, color='green', lw=1.5, linestyle='--',
               label=f'BPF reference: {cfg.bpf_fundamental_order}')
    ax.set_xlabel('Candidate fundamental order', fontsize=11)
    ax.set_ylabel('Normalised score', fontsize=11)
    ax.set_title('Harmonic-Sum Score (peaks = detected fundamentals)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    if has_cep and n_rows > 1:
        ax2 = axes_list[1]
        ax2.plot(results.raw_cepstrum_q, results.raw_cepstrum,
                 color='steelblue', lw=0.9, label='Real cepstrum (raw combined)')
        if results.env_cepstrum is not None:
            ax2.plot(results.env_cepstrum_q, results.env_cepstrum,
                     color='darkred', lw=0.9, alpha=0.8, label='Real cepstrum (envelope)')
        q_bpf = 1.0 / cfg.bpf_fundamental_order
        ax2.axvline(q_bpf, color='green', lw=1.5, linestyle='--',
                    label=f'BPF rahmonic: 1/{cfg.bpf_fundamental_order:.2f} = {q_bpf:.3f} rev')
        ax2.set_xlim(0, min(results.raw_cepstrum_q[-1], 2.0))
        ax2.set_xlabel('Quefrency (revolutions)', fontsize=11)
        ax2.set_ylabel('Cepstrum amplitude', fontsize=11)
        ax2.set_title('Real Cepstrum  (rahmonic at 1/f0 confirms a harmonic family)',
                      fontsize=11)
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig
