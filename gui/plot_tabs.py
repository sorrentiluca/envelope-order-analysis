from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from gui.canvas_widget import CanvasWidget
from app.marker_model import FaultOrder

_CAT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _draw_fault_vlines(ax, fault_orders: list[FaultOrder], x_max: float):
    if x_max <= 0 or not fault_orders:
        return
    for fo in fault_orders:
        labeled = False
        for n in range(1, fo.n_harmonics + 1):
            x = fo.fundamental * n
            if x > x_max:
                break
            alpha = max(0.15, 0.9 / n)
            lw = 1.4 if n == 1 else 0.8
            ls = "-" if n == 1 else "--"
            label = f"{fo.name} ({fo.fundamental}×)" if not labeled else None
            ax.axvline(x, color=fo.color, alpha=alpha, linewidth=lw, linestyle=ls, label=label)
            labeled = True


def _combine_spectra(axis_spectra: list[tuple], method: str = "sum") -> tuple:
    valid = [(f, m) for f, m in axis_spectra if len(f) > 1]
    if not valid:
        return np.array([0.0]), np.array([0.0])
    if len(valid) == 1:
        return valid[0]
    f_max = min(f[-1] for f, _ in valid)
    df_step = max(f[1] - f[0] for f, _ in valid)
    n_pts = max(2, int(f_max / df_step) + 1)
    common = np.linspace(0.0, f_max, n_pts)
    interped = []
    for freqs, mag in valid:
        fn = interp1d(freqs, mag, kind="linear", bounds_error=False, fill_value=0.0)
        interped.append(fn(common))
    stacked = np.vstack(interped)
    combined = (
        np.sqrt(np.mean(stacked ** 2, axis=0)) if method == "rms"
        else np.sum(stacked, axis=0)
    )
    return common, combined


class _SpectrumTab(QWidget):
    def __init__(self, xlabel: str, title_prefix: str, help_text: str, parent=None):
        super().__init__(parent)
        self._xlabel = xlabel
        self._title_prefix = title_prefix
        self._cat_spectra: dict[str, dict[str, tuple]] = {}
        self._fault_orders: list[FaultOrder] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        ctrl.addStretch()
        ctrl.addWidget(QLabel("Combined:"))
        self._combine_cb = QComboBox()
        self._combine_cb.addItems(["Sum", "RMS"])
        self._combine_cb.setFixedWidth(70)
        self._combine_cb.currentTextChanged.connect(self._render)
        ctrl.addWidget(self._combine_cb)
        layout.addLayout(ctrl)

        self._help_label = QLabel(help_text)
        self._help_label.setWordWrap(True)
        self._help_label.setVisible(False)
        self._help_label.setStyleSheet(
            "background:#fffbe6;border:1px solid #e6d800;"
            "border-radius:3px;padding:4px;font-size:11px;"
        )
        layout.addWidget(self._help_label)

        self._canvas = CanvasWidget()
        layout.addWidget(self._canvas)

    def set_help_visible(self, visible: bool):
        self._help_label.setVisible(visible)

    def update_plot(
        self,
        cat_spectra: dict[str, dict[str, tuple]],
        fault_orders: list[FaultOrder],
    ):
        self._cat_spectra = cat_spectra
        self._fault_orders = fault_orders
        self._render()

    def _render(self):
        cat_spectra = self._cat_spectra
        fault_orders = self._fault_orders
        if not cat_spectra:
            self._canvas.clear()
            return

        # Collect unique axis labels in insertion order
        axes: list[str] = []
        for cat_dict in cat_spectra.values():
            for lbl in cat_dict:
                if lbl not in axes:
                    axes.append(lbl)

        multi = any(len(d) > 1 for d in cat_spectra.values())
        n_plots = len(axes) + (1 if multi else 0)
        if n_plots == 0:
            self._canvas.clear()
            return

        fig = self._canvas.figure()
        fig.clear()
        cats = list(cat_spectra.keys())
        cat_color = {c: _CAT_COLORS[i % len(_CAT_COLORS)] for i, c in enumerate(cats)}
        subplots = fig.subplots(n_plots, 1, sharex=True, squeeze=False)
        subplots = [sp[0] for sp in subplots]
        x_max = 0.0

        for idx, ax_lbl in enumerate(axes):
            sp = subplots[idx]
            sp.set_ylabel("Magnitude")
            sp.set_title(f"{self._title_prefix} — {ax_lbl}", fontsize=9)
            sp.grid(True, alpha=0.3)
            for cat, cat_dict in cat_spectra.items():
                if ax_lbl not in cat_dict:
                    continue
                freqs, mag = cat_dict[ax_lbl]
                if len(freqs) > 1:
                    sp.plot(freqs, mag, lw=0.9, color=cat_color[cat], label=cat)
                    x_max = max(x_max, float(freqs[-1]))
            sp.legend(fontsize=7)

        if multi:
            method = self._combine_cb.currentText().lower()
            sp = subplots[-1]
            sp.set_ylabel("Magnitude")
            sp.set_title(f"{self._title_prefix} — Combined ({method})", fontsize=9)
            sp.grid(True, alpha=0.3)
            for cat, cat_dict in cat_spectra.items():
                per_axis = list(cat_dict.values())
                freqs_c, mag_c = _combine_spectra(per_axis, method=method)
                if len(freqs_c) > 1:
                    sp.plot(freqs_c, mag_c, lw=1.4, color=cat_color[cat], label=cat)
                    x_max = max(x_max, float(freqs_c[-1]))
            sp.legend(fontsize=7)

        # Draw fault order vlines on all subplots
        for sp in subplots:
            _draw_fault_vlines(sp, fault_orders, x_max)
            sp.legend(fontsize=7)

        subplots[-1].set_xlabel(self._xlabel)
        fig.tight_layout()
        self._canvas.refresh()


class FFTTab(_SpectrumTab):
    def __init__(self, parent=None):
        super().__init__(
            "Frequency (Hz)", "FFT Spectrum",
            "Raw frequency spectrum (Hz). Averaged across files in the same category. "
            "Fault order vlines are drawn at their order× value on the Hz axis.",
            parent,
        )


class OrderTab(_SpectrumTab):
    def __init__(self, parent=None):
        super().__init__(
            "Order (× shaft speed)", "Order Spectrum",
            "Order spectrum: X-axis is shaft-speed multiples. "
            "Signal resampled to uniform angle domain before FFT.",
            parent,
        )
