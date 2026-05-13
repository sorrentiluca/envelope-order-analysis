"""
Step 9 - Cross-Checks: Harmonic-Sum Score and Cepstrum
=======================================================
Two independent techniques that confirm (or deny) a harmonic comb
without relying on visual inspection of the spectrum.

  Harmonic-Sum (HPS) score: accumulates SNR at k*f for every candidate f.
  Real cepstrum: a rahmonic at quefrency 1/f0 confirms a harmonic family.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_09_cross_checks.py
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
from core.detectors import fit_noise_floor, snr_spectrum, harmonic_sum_score, real_cepstrum
from core.plotting import annotate_box

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

ang_pl = d['ang_pl']; t_pl = d['t_pl']; r_pl = d['r_pl']
accel  = d['accel_pl']; fs = d['fs']

# Raw order spectrum
res = resample_to_uniform_angle(ang_pl, accel, t_pl, r_pl, fs, cfg)
if res is None:
    raise RuntimeError("Resampling failed.")
_, accel_angle, _, total_revs, order_res = res
orders_raw, mag_raw, _ = compute_order_spectrum(accel_angle, total_revs)
floor_raw = fit_noise_floor(orders_raw, mag_raw, cfg)
snr_raw   = snr_spectrum(mag_raw, floor_raw)

# Envelope order spectrum
f_lo, f_hi = auto_select_band(accel, fs, cfg, float(np.abs(r_pl).max()))
env_result = envelope_order_spectrum(accel, ang_pl, r_pl, t_pl, fs, f_lo, f_hi, cfg)
if env_result is None:
    raise RuntimeError("Envelope analysis failed.")
orders_env, mag_env, _ = env_result
floor_env = fit_noise_floor(orders_env, mag_env, cfg)
snr_env   = snr_spectrum(mag_env, floor_env)

# Harmonic-sum scores
f_grid_raw, score_raw = harmonic_sum_score(orders_raw, snr_raw, cfg)
f_grid_env, score_env = harmonic_sum_score(orders_env, snr_env, cfg)

# Cepstra
cep_q, cep_vals = real_cepstrum(orders_env, mag_env)
bpf_quefrency = 1.0 / cfg.bpf_fundamental_order
bpf0 = cfg.bpf_fundamental_order

score_at_bpf_raw = float(np.interp(bpf0, f_grid_raw, score_raw))
score_at_bpf_env = float(np.interp(bpf0, f_grid_env, score_env))
print(f"  HPS score at BPF order {bpf0}: raw={score_at_bpf_raw:.4f}  envelope={score_at_bpf_env:.4f}")

fig, axes_p = plt.subplots(2, 1, figsize=(16, 10))

ax = axes_p[0]
ax.plot(f_grid_raw, score_raw, lw=1.2, color='steelblue', label='Raw spectrum SNR')
ax.plot(f_grid_env, score_env, lw=1.2, color='darkred', alpha=0.85, label='Envelope SNR')
ax.axvline(bpf0, color='green', lw=2.0, ls='--', label=f'BPF reference {bpf0}')
ax.set_xlim(cfg.detector_fmin_order, min(cfg.detector_fmax_order, cfg.order_xlim))
ax.set_xlabel('Candidate fundamental order')
ax.set_ylabel('Normalised harmonic-sum score')
ax.set_title('Harmonic-Sum Detector Score  '
             '(peaks = orders with a full harmonic comb in the SNR spectrum)')
ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "The HPS detector accumulates SNR at k*f for k=1,2,... for every candidate f.\n"
    f"A harmonic family gives a HIGH score.  A peak near order {bpf0} (green line)\n"
    "= independent confirmation of BPF.")

ax = axes_p[1]
cep_mask = cep_q < (1.0 / max(cfg.detector_fmin_order, 0.1))
ax.plot(cep_q[cep_mask], np.abs(cep_vals[cep_mask]), lw=0.9, color='purple',
        label='Real cepstrum (envelope spectrum)')
ax.axvline(bpf_quefrency, color='green', lw=2.0, ls='--',
           label=f'1/{bpf0} = {bpf_quefrency:.3f} rev (expected rahmonic)')
ax.set_xlabel('Quefrency (revolutions)  [= 1 / order]')
ax.set_ylabel('Cepstrum magnitude')
ax.set_title('Real Cepstrum -- a rahmonic at quefrency 1/BPF confirms harmonic family')
ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.3)
annotate_box(ax,
    "The CEPSTRUM is the IFFT of log(|spectrum|).  A harmonic family at order f0\n"
    f"produces a rahmonic at quefrency 1/f0 = 1/{bpf0} = {bpf_quefrency:.3f} rev.\n"
    "This is an independent, threshold-light check for periodicity.")

fig.suptitle('Step 9 -- Harmonic-Sum Score & Cepstrum',
             fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 9 done.")
