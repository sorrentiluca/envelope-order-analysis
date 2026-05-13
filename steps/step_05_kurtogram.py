"""
Step 5 - Kurtogram
==================
Runs the fast (STFT-based) kurtogram on the plateau segment to find
which frequency band contains the most impulsive (peaky) energy.
That band is where ball-pass impacts excite a structural resonance.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_05_kurtogram.py
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
import matplotlib.cm as cm

from steps._setup import load_event
from core.config import Config
from core.envelope import fast_kurtogram, auto_select_band
from core.plotting import annotate_box

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

accel = d['accel_pl']
fs    = d['fs']
r_pl  = d['r_pl']

kurt_rows, f_rows, bw_rows, best = fast_kurtogram(accel, fs, cfg.kurtogram_n_levels)
f_lo_auto, f_hi_auto, _, kurt_max = best

f_lo, f_hi = auto_select_band(accel, fs, cfg, float(np.abs(r_pl).max()))

print(f"  Kurtogram best band (raw)  : {f_lo_auto:.0f}-{f_hi_auto:.0f} Hz  SK_max={kurt_max:.2f}")
print(f"  Band after min-BW check    : {f_lo:.0f}-{f_hi:.0f} Hz")
print(f"  (Override via cfg.envelope_band_hz = ({f_lo:.0f}, {f_hi:.0f}))")

fig, ax = plt.subplots(figsize=(14, 5))
cmap_k = cm.get_cmap('hot_r')
all_sk = np.concatenate(kurt_rows) if kurt_rows else np.array([0.0])
vmax_k = max(float(np.percentile(all_sk, 99)), 0.1)
sc = None
for i, (sk, f, bw) in enumerate(zip(kurt_rows, f_rows, bw_rows)):
    sc = ax.scatter(f, np.full_like(f, i), c=sk, cmap=cmap_k,
                    vmin=0.0, vmax=vmax_k, s=18, marker='s', alpha=0.85)
if sc is not None:
    plt.colorbar(sc, ax=ax, label='Spectral Kurtosis (SK)')
ax.axvline(f_lo, color='cyan', lw=2.5, ls='--', label=f'Band lo {f_lo:.0f} Hz')
ax.axvline(f_hi, color='cyan', lw=2.5, ls='-',  label=f'Band hi {f_hi:.0f} Hz')
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('Level (larger window = lower level index)', fontsize=10)
ax.set_title(f'Kurtogram -- selected band: {f_lo:.0f}-{f_hi:.0f} Hz  (SK={kurt_max:.2f})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, ls='--', alpha=0.2)
annotate_box(ax,
    "SPECTRAL KURTOSIS (SK) measures how impulsive the signal energy is\n"
    "at each frequency.  Impulsive (peaky in time) = high kurtosis.\n"
    "Ball-pass impacts excite a resonance -> high SK near that resonance.\n"
    "Cyan lines = band selected for band-pass before demodulation.\n"
    f"Override: set cfg.envelope_band_hz = ({f_lo:.0f}, {f_hi:.0f})")

fig.suptitle('Step 5 -- Kurtogram', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 5 done.")
