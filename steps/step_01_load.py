"""
Step 1 - Load Data
==================
Shows the full raw acceleration trace with RPM overlay, then the RPM
profile that drives event detection.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_01_load.py
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
from core.plotting import annotate_box

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

time     = d['time']
rpm      = d['rpm']
axis     = d['axis']
accel    = d['channels'][axis]
fs       = d['fs']

AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}
ax_color = AXIS_COLORS.get(axis, 'steelblue')

print(f"  Sample rate : {fs:.1f} Hz")
print(f"  Total samples: {len(time):,}")
print(f"  Duration    : {time[-1] - time[0]:.2f} s")
print(f"  Events found: {d['n_events']}")

fig, axes_p = plt.subplots(2, 1, figsize=(16, 7), sharex=False)

ax = axes_p[0]
ax.plot(time, accel, lw=0.4, color=ax_color, label=f'{axis} accel')
ax.set_ylabel('Acceleration (m/s2)')
ax.set_xlabel('Time (s)')
ax.set_title(f'Raw {axis}-axis acceleration -- full file')
ax2 = ax.twinx()
ax2.plot(time, rpm, lw=0.8, color='grey', alpha=0.5, label='RPM')
ax2.set_ylabel('RPM', color='grey')
ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    f"Full file raw signal.\n"
    f"Each spike cluster = one actuation (motor on -> coast -> off).\n"
    f"We cannot see ball-pass events in this view.")

ax = axes_p[1]
ax.plot(time, np.abs(rpm), lw=0.8, color='grey', label='|RPM|')
ax.axhline(cfg.rpm_threshold, color='red', lw=1, ls='--',
           label=f'RPM threshold ({cfg.rpm_threshold})')
ax.set_ylabel('|RPM|')
ax.set_xlabel('Time (s)')
ax.set_title('RPM profile -- used to find actuation events')
ax.legend(fontsize=8)
ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "We find events as upward crossings of the RPM threshold.\n"
    "Each crossing = start of one actuation cycle.")

fig.suptitle('Step 1 -- Raw Data', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 1 done.")
