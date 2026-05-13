"""
Step 8 - Envelope Order Spectrum  (THE REVEAL)
===============================================
Pipeline: band-pass -> Hilbert envelope -> angle resample -> FFT.
Shows the raw order spectrum alongside the envelope order spectrum.
BPF harmonics that are buried in the raw spectrum emerge clearly here.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_08_envelope_spectrum.py
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
from core.spectrum import compute_order_spectrum
from core.envelope import auto_select_band, envelope_order_spectrum
from core.detectors import fit_noise_floor, snr_spectrum
from core.plotting import annotate_box, add_bpf_vlines, bpf_reference_orders

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

ang_pl = d['ang_pl']
t_pl   = d['t_pl']
r_pl   = d['r_pl']
accel  = d['accel_pl']
fs     = d['fs']
axis   = d['axis']

# Raw order spectrum
res = resample_to_uniform_angle(ang_pl, accel, t_pl, r_pl, fs, cfg)
if res is None:
    raise RuntimeError("Resampling failed -- try a different EVENT_INDEX.")
_, accel_angle, _, total_revs, order_res = res
orders_raw, mag_raw, _ = compute_order_spectrum(accel_angle, total_revs)
floor_raw = fit_noise_floor(orders_raw, mag_raw, cfg)
snr_raw   = snr_spectrum(mag_raw, floor_raw)

# Envelope order spectrum
f_lo, f_hi = auto_select_band(accel, fs, cfg, float(np.abs(r_pl).max()))
env_result = envelope_order_spectrum(accel, ang_pl, r_pl, t_pl, fs, f_lo, f_hi, cfg)
if env_result is None:
    raise RuntimeError("Envelope analysis failed -- check band selection.")
orders_env, mag_env, band_used = env_result
floor_env = fit_noise_floor(orders_env, mag_env, cfg)
snr_env   = snr_spectrum(mag_env, floor_env)

bpf_orders = bpf_reference_orders(cfg)

print(f"  Order resolution : {order_res:.4f}  ({total_revs:.1f} revs)")
print(f"  Carrier band     : {f_lo:.0f}-{f_hi:.0f} Hz")
print(f"  BPF harmonic SNR table (envelope):")
for k, o in enumerate(bpf_orders[:cfg.n_bpf_harmonics], 1):
    if o > cfg.order_xlim:
        break
    snr_here = float(np.interp(o, orders_env, snr_env))
    flag = '  <- DETECTED' if snr_here >= 3.0 else ''
    print(f"    {k}x  order {o:.2f}  SNR={snr_here:.2f}{flag}")

AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}
ax_color = AXIS_COLORS.get(axis, 'steelblue')
bpf0 = cfg.bpf_fundamental_order

fig, axes_p = plt.subplots(1, 2, figsize=(18, 7))

ax = axes_p[0]
mask_r = orders_raw <= cfg.order_xlim
ymax_r = float(mag_raw[mask_r].max()) * 1.25 or 1e-9
ax.plot(orders_raw[mask_r], mag_raw[mask_r], lw=0.7, color=ax_color, label='Raw spectrum')
ax.plot(orders_raw[mask_r], floor_raw[mask_r], lw=1.2, color='grey', ls='--', label='Noise floor')
add_bpf_vlines(ax, cfg, ymax_r)
ax.set_xlim(0, cfg.order_xlim); ax.set_ylim(0, ymax_r)
ax.set_xlabel('Order'); ax.set_ylabel('Magnitude (m/s2)')
ax.set_title(f'Raw order spectrum -- {axis} axis')
ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax, "Raw spectrum: BPF family buried under noise floor.")

ax = axes_p[1]
mask_e = orders_env <= cfg.order_xlim
ymax_e = float(mag_env[mask_e].max()) * 1.25 or 1e-9
ax.plot(orders_env[mask_e], mag_env[mask_e], lw=0.9, color='darkred',
        label='Envelope spectrum')
ax.plot(orders_env[mask_e], floor_env[mask_e], lw=1.2, color='grey', ls='--',
        label='Noise floor')
add_bpf_vlines(ax, cfg, ymax_e)
for o in bpf_orders[:4]:
    if o > cfg.order_xlim:
        break
    snr_here = float(np.interp(o, orders_env, snr_env))
    ax.text(o + 0.3, float(np.interp(o, orders_env, mag_env)) * 1.4 + ymax_e * 0.01,
            f'SNR={snr_here:.1f}', fontsize=6, color='green')
ax.set_xlim(0, cfg.order_xlim); ax.set_ylim(0, ymax_e)
ax.set_xlabel('Order'); ax.set_ylabel('Envelope magnitude (m/s2)')
ax.set_title(f'Envelope order spectrum  <-- THE VERDICT\n'
             f'BPF @ order {bpf0}  (band: {f_lo:.0f}-{f_hi:.0f} Hz)',
             fontweight='bold', color='darkred')
ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    f"Envelope of band-passed ({f_lo:.0f}-{f_hi:.0f} Hz) signal.\n"
    f"A comb at orders {bpf0}, {2*bpf0:.2f}, {3*bpf0:.2f} ... confirms BPF.\n"
    "If peaks align with green lines AND SNR >= 3 -> CONFIRMED.",
    fontsize=8)

fig.suptitle('Step 8 -- ENVELOPE ORDER SPECTRUM (THE REVEAL)',
             fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 8 done.")
