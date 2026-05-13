"""
Step 7 - Hilbert Envelope
=========================
Extracts the instantaneous amplitude (envelope) of the band-passed signal
using the Hilbert transform.  After mean-removal, the envelope oscillates
at the ball-pass rate -- the BPF information is now visible in the
amplitude modulation.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_07_hilbert_envelope.py
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
from scipy.signal import hilbert

from steps._setup import load_event
from core.config import Config
from core.envelope import auto_select_band, bandpass_filter, hilbert_envelope
from core.plotting import annotate_box

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

accel = d['accel_pl']
t_pl  = d['t_pl']
fs    = d['fs']
r_pl  = d['r_pl']
axis  = d['axis']

f_lo, f_hi   = auto_select_band(accel, fs, cfg, float(np.abs(r_pl).max()))
accel_band   = bandpass_filter(accel, fs, f_lo, f_hi, cfg.envelope_filter_order)
envelope_sig = hilbert_envelope(accel_band)

n_show = min(len(t_pl), int(fs * 0.1))
t_zoom = t_pl[:n_show] - t_pl[0]

AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}
ax_color = AXIS_COLORS.get(axis, 'steelblue')

print(f"  Band : {f_lo:.0f}-{f_hi:.0f} Hz")
print(f"  Envelope RMS (mean-removed): {envelope_sig.std():.6f} m/s2")

fig, axes_p = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax = axes_p[0]
ax.plot(t_zoom, accel_band[:n_show], lw=0.5, color='darkorange', alpha=0.6,
        label='Band-passed')
env_abs_zoom = np.abs(hilbert(accel_band[:n_show]))
ax.plot(t_zoom,  env_abs_zoom, lw=1.5, color='red', label='Hilbert envelope')
ax.plot(t_zoom, -env_abs_zoom, lw=1.5, color='red', alpha=0.5)
ax.set_ylabel('Accel (m/s2)')
ax.set_title('Band-passed signal with Hilbert envelope overlay (100 ms window)')
ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "The Hilbert transform gives the analytic signal whose magnitude is the\n"
    "instantaneous envelope -- the slowly-varying amplitude of the carrier.\n"
    "Red curve = |Hilbert(band-passed signal)|.")

ax = axes_p[1]
ax.plot(t_zoom, envelope_sig[:n_show], lw=0.8, color='red',
        label='Envelope (mean-removed)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Envelope (m/s2)')
ax.set_title('Mean-removed envelope -- modulation at ball-pass rate')
ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "After removing the mean, the envelope oscillates at the BPF rate.\n"
    "Each peak corresponds to a ball passing the load zone.\n"
    "We now resample this envelope to the angle domain and FFT it.")

fig.suptitle('Step 7 -- Hilbert Envelope', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 7 done.")
