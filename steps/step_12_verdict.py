"""
Step 12 - Verdict
 =================
Summary SNR table for all BPF harmonics + the "money figure":
raw order spectrum vs combined multi-axis envelope spectrum side by side.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_12_verdict.py
"""
# ── CONFIG ───────────────────────────────────────────────────────────────────
FILE_PATH   = r'C:\path\to\your\data\2300-30k-metal635-1.csv'
EVENT_INDEX = 0
AXIS        = 'Z'
# ───────────────────────────────────────────────────────────────────────────

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from steps._setup import load_event
from core.config import Config
from core.resampling import resample_to_uniform_angle
from core.spectrum import compute_order_spectrum, interpolate_to_common_orders
from core.envelope import auto_select_band, envelope_order_spectrum
from core.detectors import fit_noise_floor, snr_spectrum
from core.plotting import annotate_box, add_bpf_vlines, bpf_reference_orders

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

ang_pl       = d['ang_pl']; t_pl = d['t_pl']; r_pl = d['r_pl']
fs           = d['fs']; axis = d['axis']
active_axes  = d['active_axes']
axis_data_pl = d['axis_data_pl']
accel        = d['accel_pl']

# Raw order spectrum (primary axis)
res = resample_to_uniform_angle(ang_pl, accel, t_pl, r_pl, fs, cfg)
if res is None:
    raise RuntimeError("Resampling failed.")
_, accel_angle, _, total_revs, order_res = res
orders_raw, mag_raw, _ = compute_order_spectrum(accel_angle, total_revs)
floor_raw = fit_noise_floor(orders_raw, mag_raw, cfg)
snr_raw   = snr_spectrum(mag_raw, floor_raw)

# Envelope band
f_lo, f_hi = auto_select_band(accel, fs, cfg, float(np.abs(r_pl).max()))

# Multi-axis envelope spectra
common_orders = np.arange(0, cfg.order_xlim, cfg.common_order_resolution)
per_axis_env: dict = {}
for ax_name in active_axes:
    result = envelope_order_spectrum(
        axis_data_pl[ax_name], ang_pl, r_pl, t_pl, fs, f_lo, f_hi, cfg
    )
    if result is not None:
        o_ax, m_ax, _ = result
        per_axis_env[ax_name] = interpolate_to_common_orders(o_ax, m_ax, common_orders)

combined_env   = sum(per_axis_env.values()) if per_axis_env else np.zeros_like(common_orders)
combined_floor = fit_noise_floor(common_orders, combined_env, cfg)
combined_snr   = snr_spectrum(combined_env, combined_floor)

bpf_orders = bpf_reference_orders(cfg)
bpf0       = cfg.bpf_fundamental_order
N_HARM     = cfg.n_bpf_harmonics

# ── Verdict table ─────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"  FILE : {FILE_PATH}")
print(f"  Axis  : {axis}   Event: {d['event_index']}")
print(f"  Order res: {order_res:.4f}  ({total_revs:.1f} revs)")
print(f"  Carrier band: {f_lo:.0f}-{f_hi:.0f} Hz")
print()
print(f"  {'Harmonic':>9}  {'Order':>7}  {'SNR':>7}  Status")
for k, o in enumerate(bpf_orders[:N_HARM], 1):
    if o > cfg.order_xlim:
        break
    snr_here = float(np.interp(o, common_orders, combined_snr))
    flag = '  <- DETECTED' if snr_here >= 3.0 else ''
    print(f"  {k:>9}x  {o:>7.2f}  {snr_here:>7.2f}{flag}")
print("=" * 72)

# ── Money figure ─────────────────────────────────────────────────────────────────
xlim2 = min(cfg.order_xlim, bpf0 * (N_HARM + 1))
AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}
ax_color = AXIS_COLORS.get(axis, 'steelblue')

fig, axes_p = plt.subplots(1, 2, figsize=(18, 7))

ax = axes_p[0]
mask_r = orders_raw <= xlim2
ymax_r = float(mag_raw[mask_r].max()) * 1.3 or 1e-9
ax.plot(orders_raw[mask_r], mag_raw[mask_r], lw=0.8, color=ax_color, label='Raw')
ax.plot(orders_raw[mask_r], floor_raw[mask_r], lw=1.2, color='grey', ls='--',
        label='Noise floor')
add_bpf_vlines(ax, cfg, ymax_r)
ax.set_xlim(0, xlim2); ax.set_ylim(0, ymax_r)
ax.set_xlabel('Order'); ax.set_ylabel('Magnitude (m/s2)')
ax.set_title(f'Raw order spectrum -- {axis} axis', fontsize=12)
ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)

ax = axes_p[1]
mask_v = common_orders <= xlim2
ymax_v = float(combined_env[mask_v].max()) * 1.3 or 1e-9
ax.plot(common_orders[mask_v], combined_env[mask_v], lw=1.0, color='darkred',
        label='Combined envelope (X+Y+Z)')
ax.plot(common_orders[mask_v], combined_floor[mask_v], lw=1.2, color='grey', ls='--',
        label='Noise floor')
add_bpf_vlines(ax, cfg, ymax_v)
for o in bpf_orders[:N_HARM]:
    if o > xlim2:
        break
    snr_here = float(np.interp(o, common_orders, combined_snr))
    if snr_here >= 3.0:
        ax.plot(o, float(np.interp(o, common_orders, combined_env)), 'g^', ms=9, zorder=5)
ax.set_xlim(0, xlim2); ax.set_ylim(0, ymax_v)
ax.set_xlabel('Order'); ax.set_ylabel('Envelope magnitude (m/s2)')
ax.set_title('Combined envelope  <-- VERDICT',
             fontsize=12, color='darkred', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    f"Green triangles = BPF harmonics with SNR >= 3.\n"
    f"A comb at orders {bpf0}, {2*bpf0:.2f}, {3*bpf0:.2f} ... = BALL-PASS CONFIRMED.\n"
    f"If nothing stands out: try Z axis, or set cfg.envelope_band_hz manually.",
    fontsize=8)

fig.suptitle('Step 12 -- VERDICT (raw vs combined envelope)',
             fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 12 done.  Walkthrough complete.")
