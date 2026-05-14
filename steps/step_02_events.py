"""
Step 2 - Event Detection and Plateau Selection
==============================================
Highlights the constant-speed plateau within one actuation event and
explains why we prefer plateau segments for order analysis.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_02_events.py
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

t_ev          = d['t_ev']
r_ev          = d['r_ev']
t_pl          = d['t_pl']
r_pl          = d['r_pl']
ang_pl        = d['ang_pl']
plateau_found = d['plateau_found']

total_revs_pl = abs(ang_pl[-1] - ang_pl[0]) / 360.0
order_res     = 1.0 / max(total_revs_pl, 0.01)

print(f"  Events found      : {d['n_events']}")
print(f"  Analysing event   : {d['event_index']}")
print(f"  Plateau found     : {plateau_found}")
print(f"  Window duration   : {t_pl[-1] - t_pl[0]:.3f} s")
print(f"  Total revs        : {total_revs_pl:.1f}")
print(f"  Order resolution  : {order_res:.4f}")
print(f"  Mean speed        : {np.abs(r_pl).mean():.0f} RPM")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_ev - t_ev[0], np.abs(r_ev), lw=1.2, color='grey', label='|RPM|')
ax.axvspan(
    t_pl[0]  - t_ev[0],
    t_pl[-1] - t_ev[0],
    color='skyblue', alpha=0.35,
    label='Analysis window' + (' (plateau)' if plateau_found else ' (full event)'),
)
ax.set_xlabel('Time within event (s)')
ax.set_ylabel('RPM')
ax.set_title(f'Event {d["event_index"]} -- RPM profile with analysis window highlighted')
ax.legend(fontsize=9)
ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "WHY prefer the plateau?\n"
    "Order resolution = 1 / total_revs.  During the plateau the shaft speed\n"
    "is nearly constant, so orders do not smear across the frequency axis.\n"
    f"Window: {t_pl[-1]-t_pl[0]:.3f} s  ->  {total_revs_pl:.1f} rev  ->  "
    f"resolution {order_res:.3f} order")

fig.suptitle('Step 2 -- Event & Plateau', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 2 done.")
