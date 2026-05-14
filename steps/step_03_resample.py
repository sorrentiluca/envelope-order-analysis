"""
Step 3 - Angle-Domain Resampling
================================
Compares three representations of the same data:
  1. Time domain  (uniform time, non-uniform angle)
  2. Raw vs angle (same samples plotted vs shaft angle)
  3. Angle-resampled (uniform angle spacing -> correct FFT -> order spectrum)

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_03_resample.py
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
from core.plotting import annotate_box

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

t_pl    = d['t_pl']
r_pl    = d['r_pl']
ang_pl  = d['ang_pl']
accel   = d['accel_pl']
fs      = d['fs']
axis    = d['axis']

AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}
ax_color = AXIS_COLORS.get(axis, 'steelblue')

res = resample_to_uniform_angle(ang_pl, accel, t_pl, r_pl, fs, cfg)
if res is None:
    raise RuntimeError("Resampling failed -- try a different EVENT_INDEX.")
uniform_angle, accel_angle, _, total_revs, order_res = res
revolutions = (uniform_angle - uniform_angle[0]) / 360.0

print(f"  Total revs      : {total_revs:.2f}")
print(f"  Order resolution: {order_res:.4f}")
print(f"  Resampled points: {len(accel_angle):,}  ({cfg.samples_per_rev} samp/rev)")

fig, axes_p = plt.subplots(3, 1, figsize=(16, 11))

ax = axes_p[0]
ax.plot(t_pl - t_pl[0], accel, lw=0.5, color=ax_color)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Accel (m/s2)')
ax.set_title('Time domain -- uniform time spacing, non-uniform angle spacing')
ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "TIME domain: samples equally spaced in TIME, not angle.\n"
    "When speed varies, equal-time samples represent unequal shaft angles.\n"
    "An FFT on this gives a FREQUENCY spectrum, not an ORDER spectrum.")

ax = axes_p[1]
ang_plot = ang_pl - ang_pl[0]
ax.plot(ang_plot, accel, lw=0.5, color='darkorange')
ax.set_xlabel('Shaft angle from start (deg)'); ax.set_ylabel('Accel (m/s2)')
ax.set_title('Same data plotted vs angle -- samples still non-uniform in angle')
ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "Plotting vs. angle reveals mechanical periodicity better, but the\n"
    "samples are still IRREGULARLY spaced in angle -- FFT still wrong.")

ax = axes_p[2]
ax.plot(revolutions, accel_angle, lw=0.5, color='seagreen')
ax.set_xlabel('Shaft revolutions'); ax.set_ylabel('Accel (m/s2)')
ax.set_title(f'Angle-domain resampled  ({cfg.samples_per_rev} samples/rev, uniform spacing)')
ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    f"After resampling to {cfg.samples_per_rev} samples/rev (uniform angle spacing),\n"
    "an FFT gives an ORDER spectrum: x-axis = events per revolution.\n"
    f"Order resolution = 1 / {total_revs:.1f} rev = {order_res:.4f} order/bin.")

fig.suptitle('Step 3 -- Angle Resampling', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 3 done.")
