from __future__ import annotations
from collections import defaultdict

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QScrollArea, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QLabel, QMessageBox, QTabWidget,
)
from PySide6.QtCore import QThread, Signal, QObject

from gui.file_panel import FilePanel
from gui.column_panel import ColumnPanel
from gui.orders_panel import OrdersPanel
from gui.preview_panel import PreviewPanel
from gui.plot_tabs import FFTTab, OrderTab
from app.fft_engine import compute_raw_fft, average_spectra_interp
from app.order_engine import compute_order_spectrum_from_arrays, average_order_spectra
from app.segmentation_engine import SegmentWindow
from core.io import estimate_fs


class _Worker(QObject):
    finished = Signal(dict, dict)
    error = Signal(str)

    def __init__(self, entries, col_map, event_windows, samples_per_rev):
        super().__init__()
        self.entries = entries
        self.col_map = col_map
        self.event_windows = event_windows
        self.samples_per_rev = samples_per_rev

    def run(self):
        try:
            fft_by_cat: dict[str, list] = defaultdict(list)
            order_by_cat: dict[str, list] = defaultdict(list)

            for entry in self.entries:
                df = pd.read_csv(entry.file_path)
                t_col = self.col_map.get("time")
                a_col = self.col_map.get("accel")
                r_col = self.col_map.get("rpm")
                ang_col = self.col_map.get("angle")
                if not t_col or not a_col:
                    continue

                time = df[t_col].values.astype(float)
                accel = df[a_col].values.astype(float)
                rpm_arr = (
                    df[r_col].values.astype(float)
                    if r_col and r_col in df.columns else None
                )
                angle_arr = (
                    df[ang_col].values.astype(float)
                    if ang_col and ang_col in df.columns else None
                )

                windows: list[SegmentWindow] = self.event_windows.get(entry.file_path, [])
                enabled = [w for w in windows if w.enabled]
                slices = [(w.i_start, w.i_end) for w in enabled] if enabled else [(0, len(time))]

                for i0, i1 in slices:
                    sig = accel[i0:i1]
                    t_sl = time[i0:i1]
                    if len(sig) < 8:
                        continue
                    fs = estimate_fs(t_sl)
                    fft_by_cat[entry.category].append(compute_raw_fft(sig, fs))

                    if angle_arr is not None and rpm_arr is not None:
                        result = compute_order_spectrum_from_arrays(
                            angle_arr[i0:i1], sig, t_sl,
                            rpm_arr[i0:i1], fs, self.samples_per_rev,
                        )
                        if result is not None:
                            order_by_cat[entry.category].append(result)

            fft_results = {
                cat: average_spectra_interp(sp) for cat, sp in fft_by_cat.items()
            }
            order_results = {
                cat: average_order_spectra(sp) for cat, sp in order_by_cat.items()
            }
            self.finished.emit(fft_results, order_results)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration Analyser")
        self.resize(1400, 820)
        self._event_windows: dict[str, list[SegmentWindow]] = {}
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Sidebar
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        sb = QWidget()
        sb_layout = QVBoxLayout(sb)
        sb_layout.setSpacing(6)

        self._file_panel = FilePanel()
        sb_layout.addWidget(self._file_panel)

        self._col_panel = ColumnPanel()
        sb_layout.addWidget(self._col_panel)

        det_box = QGroupBox("Segmentation")
        det_form = QFormLayout(det_box)
        self._rpm_thresh_spin = QDoubleSpinBox()
        self._rpm_thresh_spin.setRange(0, 50000)
        self._rpm_thresh_spin.setValue(100.0)
        det_form.addRow("RPM threshold:", self._rpm_thresh_spin)
        self._spr_spin = QSpinBox()
        self._spr_spin.setRange(64, 8192)
        self._spr_spin.setValue(1024)
        det_form.addRow("Samples / rev:", self._spr_spin)
        sb_layout.addWidget(det_box)

        self._orders_panel = OrdersPanel()
        sb_layout.addWidget(self._orders_panel)

        preview_btn = QPushButton("Preview Signal…")
        preview_btn.clicked.connect(self._open_preview)
        sb_layout.addWidget(preview_btn)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setFixedHeight(36)
        sb_layout.addWidget(self._run_btn)
        sb_layout.addStretch()

        scroll.setWidget(sb)
        root.addWidget(scroll)

        # Right tabs
        self._tabs = QTabWidget()
        self._preview_panel = PreviewPanel()
        self._fft_tab = FFTTab()
        self._order_tab = OrderTab()
        self._tabs.addTab(self._preview_panel, "Signal Preview")
        self._tabs.addTab(self._fft_tab, "FFT Spectrum")
        self._tabs.addTab(self._order_tab, "Order Spectrum")
        root.addWidget(self._tabs, stretch=1)

        self._file_panel.files_changed.connect(self._on_files_changed)
        self._run_btn.clicked.connect(self._run_analysis)
        self._preview_panel.selection_confirmed.connect(self._on_selection_confirmed)

    def _on_files_changed(self):
        entries = self._file_panel.get_entries()
        if not entries:
            return
        try:
            df = pd.read_csv(entries[0].file_path, nrows=5)
            self._col_panel.update_columns(list(df.columns))
        except Exception:
            pass

    def _open_preview(self):
        entries = self._file_panel.get_entries()
        if not entries:
            QMessageBox.warning(self, "No files", "Add at least one data file first.")
            return
        col_map = self._col_panel.get_mapping()
        if not col_map.get("time") or not col_map.get("accel"):
            QMessageBox.warning(self, "Columns", "Please map Time and Acceleration columns.")
            return
        self._preview_panel.set_files([e.file_path for e in entries], col_map)
        self._tabs.setCurrentWidget(self._preview_panel)

    def _on_selection_confirmed(self, file_path: str, windows: list):
        self._event_windows[file_path] = windows
        self._run_analysis()

    def _run_analysis(self):
        entries = self._file_panel.get_entries()
        if not entries:
            QMessageBox.warning(self, "No files", "Add at least one data file first.")
            return
        col_map = self._col_panel.get_mapping()
        if not col_map.get("time") or not col_map.get("accel"):
            QMessageBox.warning(self, "Columns", "Please map Time and Acceleration columns.")
            return
        if self._thread and self._thread.isRunning():
            return

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")

        self._worker = _Worker(
            entries, col_map, self._event_windows, self._spr_spin.value()
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_finished(self, fft_results: dict, order_results: dict):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run Analysis")
        fault_orders = self._orders_panel.get_fault_orders()
        if fft_results:
            self._fft_tab.update_plot(fft_results, fault_orders)
            self._tabs.setCurrentWidget(self._fft_tab)
        if order_results:
            self._order_tab.update_plot(order_results, fault_orders)

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run Analysis")
        QMessageBox.critical(self, "Analysis error", msg)
