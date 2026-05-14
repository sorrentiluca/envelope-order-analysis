from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from gui.canvas_widget import CanvasWidget
from app.marker_model import FaultOrder


def _draw_fault_vlines(ax, fault_orders: list[FaultOrder], x_max: float):
    for fo in fault_orders:
        labeled = False
        for n in range(1, fo.n_harmonics + 1):
            x = fo.fundamental * n
            if x > x_max:
                break
            alpha = max(0.12, 0.85 / n)
            lw = 1.2 if n == 1 else 0.7
            ls = "-" if n == 1 else "--"
            label = f"{fo.name} ({fo.fundamental}×)" if not labeled else None
            ax.axvline(x, color=fo.color, alpha=alpha, linewidth=lw, linestyle=ls, label=label)
            labeled = True


class FFTTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = CanvasWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def update_plot(
        self,
        category_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
        fault_orders: list[FaultOrder],
    ):
        fig = self._canvas.figure()
        fig.clear()
        ax = fig.add_subplot(111)
        x_max = 0.0
        for label, (freqs, mag) in category_spectra.items():
            if len(freqs) > 1:
                ax.semilogy(freqs, mag, lw=0.8, label=label)
                x_max = max(x_max, float(freqs[-1]))
        if x_max > 0:
            _draw_fault_vlines(ax, fault_orders, x_max)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Raw FFT Spectrum — Average per Category")
        if category_spectra:
            ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._canvas.refresh()


class OrderTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = CanvasWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def update_plot(
        self,
        category_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
        fault_orders: list[FaultOrder],
    ):
        fig = self._canvas.figure()
        fig.clear()
        ax = fig.add_subplot(111)
        x_max = 0.0
        for label, (orders, mag) in category_spectra.items():
            if len(orders) > 1:
                ax.semilogy(orders, mag, lw=0.8, label=label)
                x_max = max(x_max, float(orders[-1]))
        if x_max > 0:
            _draw_fault_vlines(ax, fault_orders, x_max)
        ax.set_xlabel("Order (× shaft speed)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Order Spectrum — Average per Category")
        if category_spectra:
            ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._canvas.refresh()
