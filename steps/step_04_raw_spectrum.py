"""
Step 4 - Raw Order Spectrum
===========================
Computes and plots the order spectrum directly from the angle-resampled
signal.  BPF reference lines are overlaid to show why the raw spectrum
alone is not sufficient to confirm ball-pass activity.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_04_raw_spectrum.py
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
from core.plotting import annotate_box, add_bpf_vlines, bpf_reference_orders

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

t_pl   = d['t_pl']
r_pl   = d['r_pl']
ang_pl = d['ang_pl']
accel  = d['accel_pl']
fs     = d['fs']
axis   = d['axis']

res = resample_to_uniform_angle(ang_pl, accel, t_pl, r_pl, fs, cfg)
if res is None:
    raise RuntimeError("Resampling failed -- try a different EVENT_INDEX.")
_, accel_angle, _, total_revs, order_res = res

orders, mag, _ = compute_order_spectrum(accel_angle, total_revs)

bpf0 = cfg.bpf_fundamental_order
bpf_orders = bpf_reference_orders(cfg)

print(f"  Order resolution: {order_res:.4f}  ({total_revs:.1f} revs)")
for k, o in enumerate(bpf_orders[:4], 1):
    idx = np.argmin(np.abs(orders - o))
    print(f"  BPF harmonic {k}x  (order {o:.2f}): magnitude = {mag[idx]:.5f} m/s2")

AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}
ax_color = AXIS_COLORS.get(axis, 'steelblue')

fig, ax = plt.subplots(figsize=(16, 6))
mask = orders <= cfg.order_xlim
ymax = float(mag[mask].max()) * 1.25 or 1e-9

ax.plot(orders[mask], mag[mask], lw=0.7, color=ax_color, label=f'{axis} axis')

# Shaft harmonics
for k in [1, 2, 3, 5, 10, 20]:
    if k <= cfg.order_xlim:
        ax.axvline(k, color='#aaaaaa', lw=0.8, ls='-', alpha=0.5)
        ax.text(k + 0.2, ymax * 0.88, f'{k}x', fontsize=7, color='#666')

add_bpf_vlines(ax, cfg, ymax)
ax.set_xlim(0, cfg.order_xlim); ax.set_ylim(0, ymax)
ax.set_xlabel('Order (events per revolution)', fontsize=11)
ax.set_ylabel('Magnitude (m/s2)', fontsize=11)
ax.set_title(f'Raw order spectrum -- {axis} axis  '
             f'(order res = {order_res:.3f}, from {total_revs:.1f} revs)',
             fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    f"Grey lines = shaft-synchronous harmonics (1x, 2x, 3x ...).\n"
    f"Green dotted lines = BPF family at {bpf0}, {2*bpf0:.2f}, {3*bpf0:.2f} ...\n"
    f"At order resolution {order_res:.3f}, the BPF family is buried under noise.\n"
    f"We need a completely different approach -- envelope demodulation.")

fig.suptitle('Step 4 -- Raw Order Spectrum', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 4 done.")
