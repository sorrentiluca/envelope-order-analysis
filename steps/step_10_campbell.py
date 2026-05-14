"""
Step 10 - Campbell Diagram (Order-RPM Heatmap)
===============================================
Uses the full event window (ramp up -> plateau -> ramp down) so the
speed axis spans a wide RPM range.  A true mechanical order appears as
a vertical stripe; a structural resonance appears as a horizontal band.

Edit FILE_PATH, EVENT_INDEX, AXIS, then run:
    python steps/step_10_campbell.py
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
from core.campbell import compute_campbell
from core.plotting import annotate_box, bpf_reference_orders

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

# Use the full event window for the widest RPM range
ang_ev  = d['ang_ev']
t_ev    = d['t_ev']
r_ev    = d['r_ev']
accel_ev = d['accel_ev']
fs      = d['fs']
axis    = d['axis']
bpf0    = cfg.bpf_fundamental_order

res_event = resample_to_uniform_angle(ang_ev, accel_ev, t_ev, r_ev, fs, cfg)

fig, ax = plt.subplots(figsize=(16, 7))

if res_event is None:
    ax.text(0.5, 0.5, 'Resampling failed -- try a different EVENT_INDEX.',
            ha='center', va='center', transform=ax.transAxes, fontsize=13, color='grey')
else:
    _, accel_r, rpm_r, total_revs_ev, _ = res_event
    revs_block = max(1, min(cfg.campbell_revs_per_block, int(total_revs_ev / 10)))
    X_ord, Y_rpm, Z_map = compute_campbell(
        accel_r, rpm_r, cfg.samples_per_rev, revs_block, cfg.campbell_overlap
    )
    camp_ok = (Z_map.size > 0 and Y_rpm.size >= 2 and len(np.unique(Y_rpm)) >= 2)

    if camp_ok:
        pos_mask = Y_rpm > 10.0
        Y_pos = Y_rpm[pos_mask]
        Z_pos = Z_map[pos_mask]
        if Z_pos.size > 0 and np.any(Z_pos > 0):
            vmax_val = max(float(np.percentile(Z_pos[Z_pos > 0], 98)), 1e-9)
            mesh = ax.pcolormesh(X_ord, Y_pos, Z_pos,
                                 shading='auto', cmap='jet', vmin=0.0, vmax=vmax_val)
            plt.colorbar(mesh, ax=ax, label=f'Magnitude (m/s2) -- {axis} axis')
        for o in bpf_reference_orders(cfg):
            if o <= cfg.campbell_order_xlim:
                ax.axvline(o, color='white', lw=0.9, ls=':', alpha=0.8)
        ax.set_xlim(0, cfg.campbell_order_xlim)
        ax.set_xlabel('Order (events per revolution)', fontsize=11)
        ax.set_ylabel('Speed (RPM)', fontsize=11)
        ax.set_title(f'Order-RPM Heatmap (Campbell) -- {axis} axis\n'
                     'White dotted lines = BPF reference orders',
                     fontsize=12, fontweight='bold')
        annotate_box(ax,
            "Campbell: colour = order spectrum magnitude at each RPM.\n"
            "A TRUE mechanical order = VERTICAL stripe (same order, all speeds).\n"
            "A structural RESONANCE = HORIZONTAL band (constant Hz, not order).\n"
            f"BPF at order {bpf0} should be a vertical stripe.",
            fontsize=8)
        print(f"  Campbell computed: {Z_map.shape[0]} RPM blocks x {Z_map.shape[1]} orders")
    else:
        ax.text(0.5, 0.5,
                'Campbell not available\n(too few speed-varying points in this event)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='grey')
        print("  Campbell skipped -- not enough RPM variation.")

fig.suptitle('Step 10 -- Campbell Diagram', fontsize=10, color='#666', x=0.99, ha='right')
plt.tight_layout()
plt.show()
print("Step 10 done.")
