"""
Full batch analysis pipeline.

Loads every file listed in Config.files, processes all events across
all files, aggregates the results, prints a diagnostics table, and
displays all summary plots.

Edit core/config.py (Config.base_path and Config.files) then run:
    python run_analysis.py
"""
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from core.config import Config, Segment, SegmentResult, AggResult, Candidate, Results
from core.io import load_many, check_columns, extract_channels, estimate_fs, report_encoder_resolution
from core.segmentation import (
    unwrap_angle, pick_segments, concatenate_matched_segments, extract_segment_signal
)
from core.resampling import resample_to_uniform_angle
from core.spectrum import (
    compute_order_spectrum, interpolate_to_common_orders,
    average_spectra_rms, time_synchronous_average, residual_after_tsa,
)
from core.envelope import auto_select_band, envelope_order_spectrum, fast_kurtogram
from core.detectors import (
    fit_noise_floor, snr_spectrum, harmonic_sum_score,
    real_cepstrum, detect_fundamentals,
)
from core.campbell import compute_campbell
from core.plotting import (
    plot_raw_and_angle_waveform, plot_order_spectrum,
    plot_envelope_order_spectrum, plot_kurtogram,
    plot_campbell, plot_detector_score,
)


def _analyze_segment(
    seg: Segment,
    channels: dict,
    axis: str,
    fs: float,
    common_orders: np.ndarray,
    cfg: Config,
    band_hz: Optional[tuple] = None,
    store_waveform: bool = False,
) -> Optional[SegmentResult]:
    angle_raw, accel_raw, time_raw, rpm_raw = extract_segment_signal(channels, seg, axis)
    res = resample_to_uniform_angle(angle_raw, accel_raw, time_raw, rpm_raw, fs, cfg)
    if res is None:
        return None
    uniform_angle, accel_res, rpm_res, total_revs, order_res = res

    orders, mag, _ = compute_order_spectrum(accel_res, total_revs)
    mag_common     = interpolate_to_common_orders(orders, mag, common_orders)

    sr = SegmentResult(
        axis=axis, total_revs=total_revs, order_resolution=order_res,
        mean_rpm=float(np.mean(np.abs(rpm_res))),
        orders=common_orders, mag=mag_common,
    )
    if store_waveform:
        sr.uniform_angle = uniform_angle
        sr.angle_accel   = accel_res

    if cfg.avg_mode in ('tsa', 'both'):
        tsa_w, tsa_o, tsa_m, _ = time_synchronous_average(
            accel_res, cfg.samples_per_rev, cfg.tsa_revs_per_average
        )
        sr.tsa_orders = tsa_o
        sr.tsa_mag    = tsa_m
        residual = residual_after_tsa(accel_res, tsa_w)
        r_ord, r_mag, _ = compute_order_spectrum(residual, total_revs)
        sr.residual_mag = interpolate_to_common_orders(r_ord, r_mag, common_orders)

    if cfg.envelope_enable and band_hz is not None:
        f_lo, f_hi = band_hz
        env_res = envelope_order_spectrum(
            accel_raw, angle_raw, rpm_raw, time_raw, fs, f_lo, f_hi, cfg
        )
        if env_res is not None:
            env_orders, env_mag_raw, band_used = env_res
            sr.env_mag = interpolate_to_common_orders(env_orders, env_mag_raw, common_orders)
            sr.band_hz = band_used
    return sr


def _analyze_file(
    file_name: str,
    df,
    cfg: Config,
    common_orders: np.ndarray,
) -> tuple[dict, Optional[tuple]]:
    print(f"\n{'='*55}\n  {file_name}\n{'='*55}")
    active_axes = check_columns(df, cfg)
    channels    = extract_channels(df, cfg, active_axes)
    fs = cfg.fs_override_hz or estimate_fs(channels['time'])
    print(f"  Sample rate: {fs:.1f} Hz")
    report_encoder_resolution(channels['angle'], cfg)

    angle_uw = unwrap_angle(channels['angle'])
    segments = pick_segments(channels['time'], angle_uw, channels['rpm'], file_name, cfg)
    if not segments:
        print("  No valid segments found.")
        return {ax: [] for ax in active_axes}, None

    if cfg.concatenate_matched and len(segments) >= 2:
        segments = concatenate_matched_segments(segments, channels, active_axes, cfg)

    band_hz = kurtogram_d = None
    if cfg.envelope_enable and active_axes:
        ax0  = active_axes[0]
        seg0 = next((s for s in segments if not s.stitched), segments[0])
        if not seg0.stitched and seg0.i0 >= 0:
            raw_sig = channels[ax0][seg0.i0:seg0.i1]
            max_rpm = float(np.max(np.abs(channels['rpm'][seg0.i0:seg0.i1])))
            try:
                band_hz = auto_select_band(raw_sig, fs, cfg, max_rpm)
                if cfg.kurtogram_enable:
                    kurtogram_d = fast_kurtogram(raw_sig, fs, cfg.kurtogram_n_levels)
            except Exception as exc:
                warnings.warn(f"Band selection failed: {exc}")

    results_by_axis: dict[str, list[SegmentResult]] = {ax: [] for ax in active_axes}
    store_wfm = True
    for seg_i, seg in enumerate(segments):
        for axis in active_axes:
            sr = _analyze_segment(
                seg, channels, axis, fs, common_orders, cfg,
                band_hz=band_hz, store_waveform=store_wfm,
            )
            if sr is not None:
                results_by_axis[axis].append(sr)
                store_wfm = False
        if seg_i == 0:
            store_wfm = False

    campbell = None
    if active_axes:
        best_seg = max(
            (s for s in segments if not s.stitched and s.i0 >= 0),
            key=lambda s: s.total_revs, default=None,
        )
        if best_seg is not None:
            camp_axis  = active_axes[0]
            camp_res = resample_to_uniform_angle(
                channels['angle'][best_seg.i0:best_seg.i1],
                channels[camp_axis][best_seg.i0:best_seg.i1],
                channels['time'][best_seg.i0:best_seg.i1],
                channels['rpm'][best_seg.i0:best_seg.i1],
                fs, cfg,
            )
            if camp_res is not None:
                _, accel_r, rpm_r, _, _ = camp_res
                X, Y, Z = compute_campbell(
                    accel_r, rpm_r,
                    cfg.samples_per_rev, cfg.campbell_revs_per_block, cfg.campbell_overlap,
                )
                campbell = (X, Y, Z, camp_axis)

    return results_by_axis, (campbell, kurtogram_d)


def _aggregate(all_file_results: list, cfg: Config) -> Results:
    common_orders = np.arange(
        0, cfg.order_xlim + cfg.common_order_resolution, cfg.common_order_resolution
    )
    pooled: dict[str, list[SegmentResult]] = {}
    for results_by_axis, _ in all_file_results:
        for ax, seg_list in results_by_axis.items():
            pooled.setdefault(ax, []).extend(seg_list)

    active_axes = [ax for ax in cfg.axes if ax in pooled and pooled[ax]]
    if not active_axes:
        raise RuntimeError("No valid segments found across all files.")

    agg: dict[str, AggResult] = {}
    for axis in active_axes:
        segs     = pooled[axis]
        raw_mags = [s.mag for s in segs if s.mag is not None]
        avg_mag  = average_spectra_rms(raw_mags) if raw_mags else np.zeros_like(common_orders)
        floor    = fit_noise_floor(common_orders, avg_mag, cfg)
        snr_v    = snr_spectrum(avg_mag, floor)
        env_mags = [s.env_mag for s in segs if s.env_mag is not None]
        avg_env  = average_spectra_rms(env_mags) if env_mags else None
        env_floor = env_snr = None
        if avg_env is not None:
            env_floor = fit_noise_floor(common_orders, avg_env, cfg)
            env_snr   = snr_spectrum(avg_env, env_floor)
        agg[axis] = AggResult(
            axis=axis, n_events=len(segs), common_orders=common_orders,
            avg_mag=avg_mag, noise_floor=floor, snr=snr_v,
            avg_env_mag=avg_env, env_noise_floor=env_floor, env_snr=env_snr,
        )

    combined_mag = sum(agg[ax].avg_mag for ax in active_axes)
    combined_env_mag = (
        sum(agg[ax].avg_env_mag for ax in active_axes if agg[ax].avg_env_mag is not None)
        if any(agg[ax].avg_env_mag is not None for ax in active_axes) else None
    )
    combined_floor   = fit_noise_floor(common_orders, combined_mag, cfg)
    combined_snr     = snr_spectrum(combined_mag, combined_floor)
    combined_env_snr = None
    if combined_env_mag is not None:
        combined_env_snr = snr_spectrum(
            combined_env_mag, fit_noise_floor(common_orders, combined_env_mag, cfg)
        )

    f_grid, raw_score = harmonic_sum_score(common_orders, combined_snr, cfg)
    env_score = z_raw_score = None
    if combined_env_snr is not None:
        _, env_score = harmonic_sum_score(common_orders, combined_env_snr, cfg)
    if 'Z' in agg:
        _, z_raw_score = harmonic_sum_score(common_orders, agg['Z'].snr, cfg)

    raw_cep_q = raw_cep = env_cep_q = env_cep = None
    if cfg.cepstrum_enable:
        raw_cep_q, raw_cep = real_cepstrum(common_orders, combined_mag)
        if combined_env_mag is not None:
            env_cep_q, env_cep = real_cepstrum(common_orders, combined_env_mag)

    candidates: list[Candidate] = []
    if env_score is not None:
        candidates += detect_fundamentals(f_grid, env_score, common_orders, combined_env_snr,
                                          'envelope_combined', cfg)
    candidates += detect_fundamentals(f_grid, raw_score, common_orders, combined_snr,
                                      'raw_combined', cfg)
    if z_raw_score is not None and 'Z' in agg:
        candidates += detect_fundamentals(f_grid, z_raw_score, common_orders, agg['Z'].snr,
                                          'Z_raw', cfg)
    seen: set[float] = set()
    unique_cands: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: x.score, reverse=True):
        key = round(c.order, 2)
        if key not in seen:
            seen.add(key)
            unique_cands.append(c)
    candidates = unique_cands[:10]

    campbell_X = campbell_Y = campbell_Z = None
    campbell_axis_name = active_axes[0]
    kurtogram_data = None
    for _, extras in all_file_results:
        if extras is None:
            continue
        camp_tuple, kurt_d = extras
        if camp_tuple is not None and campbell_X is None:
            campbell_X, campbell_Y, campbell_Z, campbell_axis_name = camp_tuple
        if kurt_d is not None and kurtogram_data is None:
            kurtogram_data = kurt_d

    sample_ua = sample_aa = None
    if active_axes and pooled[active_axes[0]]:
        sr0 = pooled[active_axes[0]][0]
        if sr0.uniform_angle is not None:
            sample_ua = sr0.uniform_angle
            sample_aa = sr0.angle_accel

    return Results(
        agg=agg, common_orders=common_orders,
        combined_mag=combined_mag, combined_snr=combined_snr,
        combined_env_mag=combined_env_mag, combined_env_snr=combined_env_snr,
        raw_score_f=f_grid, raw_score=raw_score,
        env_score=env_score, z_raw_score=z_raw_score,
        raw_cepstrum_q=raw_cep_q, raw_cepstrum=raw_cep,
        env_cepstrum_q=env_cep_q, env_cepstrum=env_cep,
        candidates=candidates,
        campbell_X=campbell_X, campbell_Y_rpm=campbell_Y,
        campbell_Z_map=campbell_Z, campbell_axis=campbell_axis_name,
        sample_uniform_angle=sample_ua, sample_angle_accel=sample_aa,
        kurtogram_data=kurtogram_data,
    )


def print_diagnostics(results: Results, cfg: Config) -> None:
    print(f"\n{'='*70}\n  DIAGNOSTICS\n{'='*70}")
    for axis, agg in results.agg.items():
        print(f"\n  [{axis}]  n_events={agg.n_events}")
        snr_raw_list = [
            float(np.interp(k * cfg.bpf_fundamental_order, agg.common_orders, agg.snr))
            for k in range(1, cfg.n_bpf_harmonics + 1)
            if k * cfg.bpf_fundamental_order <= cfg.order_xlim
        ]
        print(f"    SNR BPF harmonics (raw): " + "  ".join(f"{v:.1f}" for v in snr_raw_list))
        if agg.env_snr is not None:
            snr_env_list = [
                float(np.interp(k * cfg.bpf_fundamental_order, agg.common_orders, agg.env_snr))
                for k in range(1, cfg.n_bpf_harmonics + 1)
                if k * cfg.bpf_fundamental_order <= cfg.order_xlim
            ]
            print(f"    SNR BPF harmonics (env): " + "  ".join(f"{v:.1f}" for v in snr_env_list))
    print(f"\n  Top candidates:")
    print(f"  {'Order':>8} | {'Score':>7} | {'Source':<25} | {'Harmonics SNR>=3'}")
    print(f"  {'-'*62}")
    for c in results.candidates[:8]:
        print(f"  {c.order:8.3f} | {c.score:7.3f} | {c.source:<25} | {c.n_harmonics_snr3}")
    bpf_snr_raw = float(np.interp(
        cfg.bpf_fundamental_order, results.common_orders, results.combined_snr
    ))
    bpf_snr_env = (
        float(np.interp(
            cfg.bpf_fundamental_order, results.common_orders, results.combined_env_snr
        )) if results.combined_env_snr is not None else None
    )
    print(f"\n  BPF order {cfg.bpf_fundamental_order}:  SNR(raw)={bpf_snr_raw:.2f}  "
          + (f"SNR(envelope)={bpf_snr_env:.2f}" if bpf_snr_env is not None else "envelope=N/A"))
    bpf_in_top = any(abs(c.order - cfg.bpf_fundamental_order) < 0.1 for c in results.candidates)
    print(f"  {'OK' if bpf_in_top else 'NOT'} BPF order in top candidates.")
    print(f"{'='*70}\n")


def main() -> None:
    cfg = Config()
    print(f"\nLoading {len(cfg.files)} file(s) from {cfg.base_path}")
    file_pairs = load_many(cfg.base_path, cfg.files)
    if not file_pairs:
        raise RuntimeError("No files loaded -- check base_path and file names in Config.")

    common_orders = np.arange(
        0, cfg.order_xlim + cfg.common_order_resolution, cfg.common_order_resolution
    )
    all_results = []
    for fname, df in file_pairs:
        res_by_axis, extras = _analyze_file(fname, df, cfg, common_orders)
        all_results.append((res_by_axis, extras))

    results = _aggregate(all_results, cfg)
    print_diagnostics(results, cfg)

    if cfg.do_plots:
        plot_raw_and_angle_waveform(results, cfg)
        plot_order_spectrum(results, cfg)
        plot_envelope_order_spectrum(results, cfg)
        plot_kurtogram(results, cfg)
        plot_campbell(results, cfg)
        plot_detector_score(results, cfg)
        if cfg.show_plots:
            plt.show()


if __name__ == '__main__':
    main()
