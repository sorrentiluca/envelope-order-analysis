"""
Step 6 - Band-Pass Filter
=========================
Applies a zero-phase Butterworth band-pass filter centred on the
kurtogram-selected resonance band.  Shows a 100 ms zoom comparing
the raw and filtered signals.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_06_bandpass.py
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
from core.envelope import auto_select_band, bandpass_filter
from core.plotting import annotate_box

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

accel = d['accel_pl']
t_pl  = d['t_pl']
fs    = d['fs']
r_pl  = d['r_pl']
axis  = d['axis']

f_lo, f_hi = auto_select_band(accel, fs, cfg, float(np.abs(r_pl).max()))
accel_band = bandpass_filter(accel, fs, f_lo, f_hi, cfg.envelope_filter_order)

n_show = min(len(t_pl), int(fs * 0.1))   # 100 ms zoom
t_zoom = t_pl[:n_show] - t_pl[0]

AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}
ax_color = AXIS_COLORS.get(axis, 'steelblue')

print(f"  Band selected: {f_lo:.0f}-{f_hi:.0f} Hz")
print(f"  Raw RMS     : {accel.std():.5f} m/s2")
print(f"  Filtered RMS: {accel_band.std():.5f} m/s2")

fig, axes_p = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax = axes_p[0]
ax.plot(t_zoom, accel[:n_show], lw=0.5, color=ax_color, label='Raw')
ax.set_ylabel('Accel (m/s2)')
ax.set_title(f'Raw {axis}-axis signal -- 100 ms window')
ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "Broadband raw signal: dominated by low-frequency shaft content\n"
    "and structural modes.  Ball-pass impulses are hidden in the noise.")

ax = axes_p[1]
ax.plot(t_zoom, accel_band[:n_show], lw=0.5, color='darkorange',
        label=f'Band-passed {f_lo:.0f}-{f_hi:.0f} Hz')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Accel (m/s2)')
ax.set_title('Band-passed signal -- retaining only the impulsive carrier band')
ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    f"Band-pass: {f_lo:.0f}-{f_hi:.0f} Hz (zero-phase Butterworth, order {cfg.envelope_filter_order}).\n"
    "Low-frequency shaft content removed.  Resonance ring-down now visible.\n"
    "The AMPLITUDE of this signal is modulated at the BPF rate.")

fig.suptitle('Step 6 -- Band-Pass Filter', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 6 done.")
