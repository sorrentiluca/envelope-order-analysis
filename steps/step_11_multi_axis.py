"""
Step 11 - Multi-Axis Combination
 =================================
Computes the envelope order spectrum for X, Y, and Z independently,
then sums them into a combined spectrum.  Mechanical orders present in
multiple axes add; axis-specific artefacts do not.

Edit FILE_PATH, EVENT_INDEX, AXIS (used only for band selection), then run:
    python steps/step_11_multi_axis.py
"""
# ── CONFIG ───────────────────────────────────────────────────────────────────
FILE_PATH   = r'C:\path\to\your\data\2300-30k-metal635-1.csv'
EVENT_INDEX = 0
AXIS        = 'Z'   # used for band selection; all active axes are plotted
# ───────────────────────────────────────────────────────────────────────────

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from steps._setup import load_event
from core.config import Config
from core.envelope import auto_select_band, envelope_order_spectrum
from core.detectors import fit_noise_floor, snr_spectrum
from core.spectrum import interpolate_to_common_orders
from core.plotting import annotate_box, add_bpf_vlines, bpf_reference_orders

cfg = Config()
d   = load_event(FILE_PATH, EVENT_INDEX, AXIS, cfg)

ang_pl       = d['ang_pl']; t_pl = d['t_pl']; r_pl = d['r_pl']
fs           = d['fs']
active_axes  = d['active_axes']
axis_data_pl = d['axis_data_pl']

# Band selection from the primary axis
f_lo, f_hi = auto_select_band(
    axis_data_pl[d['axis']], fs, cfg, float(np.abs(r_pl).max())
)
print(f"  Band: {f_lo:.0f}-{f_hi:.0f} Hz")

common_orders = np.arange(0, cfg.order_xlim, cfg.common_order_resolution)

per_axis_env: dict = {}
for ax_name in active_axes:
    result = envelope_order_spectrum(
        axis_data_pl[ax_name], ang_pl, r_pl, t_pl, fs, f_lo, f_hi, cfg
    )
    if result is not None:
        o_ax, m_ax, _ = result
        per_axis_env[ax_name] = interpolate_to_common_orders(o_ax, m_ax, common_orders)
        print(f"  {ax_name} envelope peak: {m_ax.max():.5f} m/s2")
    else:
        print(f"  {ax_name} envelope analysis failed -- skipping")

if not per_axis_env:
    raise RuntimeError("No envelope spectra available.")

combined_env   = sum(per_axis_env.values())
combined_floor = fit_noise_floor(common_orders, combined_env, cfg)
combined_snr   = snr_spectrum(combined_env, combined_floor)
bpf_orders     = bpf_reference_orders(cfg)

AXIS_COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e', 'Z': '#2ca02c'}

panel_names = list(per_axis_env.keys()) + ['Combined']
n_panels = len(panel_names)
ncols = 2
nrows = (n_panels + 1) // 2
fig, axes_grid = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes_flat = np.array(axes_grid).flatten()

for idx, pname in enumerate(panel_names):
    ax = axes_flat[idx]
    if pname == 'Combined':
        mag_p   = combined_env
        col_p   = 'black'
        lbl_p   = 'Combined (X+Y+Z)'
        floor_p = combined_floor
        cur_snr = combined_snr
    else:
        mag_p   = per_axis_env[pname]
        col_p   = AXIS_COLORS.get(pname, 'grey')
        lbl_p   = f'{pname} envelope'
        floor_p = fit_noise_floor(common_orders, mag_p, cfg)
        cur_snr = snr_spectrum(mag_p, floor_p)

    ymax_p = max(float(mag_p.max()) * 1.3, 1e-9)
    ax.plot(common_orders, mag_p, lw=0.8, color=col_p, label=lbl_p)
    ax.plot(common_orders, floor_p, lw=1.1, color='grey', ls='--', label='Noise floor')
    add_bpf_vlines(ax, cfg, ymax_p)
    for o in bpf_orders:
        if o > cfg.order_xlim:
            break
        if float(np.interp(o, common_orders, cur_snr)) >= 3.0:
            ax.plot(o, float(np.interp(o, common_orders, mag_p)), 'g^', ms=7, zorder=5)
    ax.set_xlim(0, cfg.order_xlim); ax.set_ylim(0, ymax_p)
    ax.set_xlabel('Order'); ax.set_ylabel('Envelope magnitude (m/s2)')
    ax.set_title(lbl_p,
                 fontweight='bold' if pname == 'Combined' else 'normal',
                 color='darkred' if pname == 'Combined' else 'black')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    if pname == 'Combined':
        annotate_box(ax,
            "Orders in multiple axes ADD when summed.\n"
            "Axis-specific artefacts do NOT add coherently.\n"
            "Green triangles = BPF harmonics with SNR >= 3.",
            fontsize=8)

for idx in range(n_panels, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle('Step 11 -- Multi-Axis Envelope Order Spectra',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("Step 11 done.")
