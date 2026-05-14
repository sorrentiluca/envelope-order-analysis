from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QScrollArea, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QLabel, QMessageBox, QTabWidget, QCheckBox, QComboBox,
)
from PySide6.QtCore import QThread, Signal, QObject

from gui.file_panel import FilePanel
from gui.column_panel import ColumnPanel
from gui.orders_panel import OrdersPanel
from gui.preview_panel import PreviewPanel
from gui.plot_tabs import FFTTab, OrderTab
from app.fft_engine import compute_raw_fft, average_spectra_interp
from app.order_engine import (
    compute_order_spectrum_from_arrays, average_order_spectra, rpm_to_angle,
)
from app.segmentation_engine import SegmentWindow, detect_events
from core.io import estimate_fs, detect_encoder_ppr


class _Worker(QObject):
    finished = Signal(dict, dict)
    error = Signal(str)

    def __init__(self, entries, col_map, event_windows, samples_per_rev,
                 detect_enabled, trigger_col, threshold, pre_s, post_s):
        super().__init__()
        self.entries = entries
        self.col_map = col_map
        self.event_windows = event_windows  # pre-confirmed edits; worker may override
        self.samples_per_rev = samples_per_rev
        self.detect_enabled = detect_enabled
        self.trigger_col = trigger_col
        self.threshold = threshold
        self.pre_s = pre_s
        self.post_s = post_s

    def run(self):
        try:
            fft_by_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
            order_by_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

            t_col = self.col_map.get("time")
            r_col = self.col_map.get("rpm")
            ang_col = self.col_map.get("angle")
            axes: dict[str, str] = self.col_map.get("axes", {})
            order_src = self.col_map.get("order_source", "angle")

            for entry in self.entries:
                df = pd.read_csv(entry.file_path)
                if not t_col or t_col not in df.columns:
                    continue

                time = df[t_col].values.astype(float)
                rpm_arr = (
                    df[r_col].values.astype(float)
                    if r_col and r_col in df.columns else None
                )
                angle_arr = (
                    df[ang_col].values.astype(float)
                    if ang_col and ang_col in df.columns else None
                )

                # Determine event windows for this file
                if self.detect_enabled and self.trigger_col and self.trigger_col in df.columns:
                    trig = df[self.trigger_col].values.astype(float)
                    windows = detect_events(
                        time, trig,
                        threshold=self.threshold,
                        pre_window_s=self.pre_s,
                        post_window_s=self.post_s,
                    )
                    # Merge with any user-confirmed edits
                    confirmed = self.event_windows.get(entry.file_path)
                    if confirmed:
                        windows = confirmed
                else:
                    confirmed = self.event_windows.get(entry.file_path)
                    windows = confirmed if confirmed else []

                enabled = [w for w in windows if w.enabled]
                slices = [(w.i_start, w.i_end) for w in enabled] if enabled else [(0, len(time))]

                for i0, i1 in slices:
                    t_sl = time[i0:i1]
                    if len(t_sl) < 8:
                        continue
                    fs = estimate_fs(t_sl)
                    rpm_sl = rpm_arr[i0:i1] if rpm_arr is not None else None

                    # Synthesise angle for order analysis
                    if order_src == "rpm_integrate" and rpm_sl is not None:
                        ang_sl = rpm_to_angle(t_sl, rpm_sl)
                    elif order_src == "angle" and angle_arr is not None:
                        ang_sl = angle_arr[i0:i1]
                    else:
                        ang_sl = None

                    # Fallback RPM for antialiasing when no RPM column
                    rpm_for_resample = rpm_sl if rpm_sl is not None else np.ones(len(t_sl))

                    for ax_label, ax_col in axes.items():
                        if ax_col not in df.columns:
                            continue
                        sig = df[ax_col].values.astype(float)[i0:i1]
                        if len(sig) < 8:
                            continue

                        fft_by_cat[entry.category][ax_label].append(
                            compute_raw_fft(sig, fs)
                        )

                        if ang_sl is not None:
                            result = compute_order_spectrum_from_arrays(
                                ang_sl, sig, t_sl, rpm_for_resample,
                                fs, self.samples_per_rev,
                            )
                            if result is not None:
                                order_by_cat[entry.category][ax_label].append(result)

            fft_results = {
                cat: {
                    ax: average_spectra_interp(sp_list)
                    for ax, sp_list in ax_dict.items()
                }
                for cat, ax_dict in fft_by_cat.items()
            }
            order_results = {
                cat: {
                    ax: average_order_spectra(sp_list)
                    for ax, sp_list in ax_dict.items()
                }
                for cat, ax_dict in order_by_cat.items()
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
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self._all_columns: list[str] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ── Sidebar ───────────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        sb = QWidget()
        sb_layout = QVBoxLayout(sb)
        sb_layout.setSpacing(6)

        # Help toggle
        self._help_btn = QPushButton("❓ Help")
        self._help_btn.setCheckable(True)
        self._help_btn.toggled.connect(self._toggle_help)
        sb_layout.addWidget(self._help_btn)

        self._file_panel = FilePanel()
        sb_layout.addWidget(self._file_panel)

        self._col_panel = ColumnPanel()
        sb_layout.addWidget(self._col_panel)

        # Segmentation group
        det_box = QGroupBox("Segmentation")
        det_layout = QVBoxLayout(det_box)
        det_form = QFormLayout()

        # Samples/rev + Auto
        spr_row = QWidget()
        spr_h = QHBoxLayout(spr_row)
        spr_h.setContentsMargins(0, 0, 0, 0)
        self._spr_spin = QSpinBox()
        self._spr_spin.setRange(64, 8192)
        self._spr_spin.setValue(1024)
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setFixedWidth(46)
        self._auto_btn.clicked.connect(self._auto_detect_spr)
        spr_h.addWidget(self._spr_spin, stretch=1)
        spr_h.addWidget(self._auto_btn)
        det_form.addRow("Samples / rev:", spr_row)
        det_layout.addLayout(det_form)

        # Detect Events checkbox
        self._detect_chk = QCheckBox("Detect Events")
        self._detect_chk.toggled.connect(self._on_detect_toggled)
        det_layout.addWidget(self._detect_chk)

        # Collapsible options
        self._det_opts = QWidget()
        opts_form = QFormLayout(self._det_opts)
        opts_form.setContentsMargins(12, 0, 0, 0)

        self._trigger_cb = QComboBox()
        opts_form.addRow("Trigger column:", self._trigger_cb)

        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0, 1e9)
        self._thresh_spin.setValue(100.0)
        self._thresh_spin.setSingleStep(10)
        opts_form.addRow("Threshold:", self._thresh_spin)

        self._pre_spin = QDoubleSpinBox()
        self._pre_spin.setRange(0, 3600)
        self._pre_spin.setSingleStep(0.05)
        self._pre_spin.setDecimals(3)
        opts_form.addRow("Time before (s):", self._pre_spin)

        self._post_spin = QDoubleSpinBox()
        self._post_spin.setRange(0, 3600)
        self._post_spin.setSingleStep(0.05)
        self._post_spin.setDecimals(3)
        opts_form.addRow("Time after (s):", self._post_spin)

        self._det_opts.setVisible(False)
        det_layout.addWidget(self._det_opts)

        self._seg_help = QLabel(
            "Samples/rev controls order resolution. \"Detect Events\" slices each file "
            "to spans where the trigger channel exceeds the threshold, extended by "
            "Time before/after. When unchecked the whole file is used."
        )
        self._seg_help.setWordWrap(True)
        self._seg_help.setVisible(False)
        self._seg_help.setStyleSheet(
            "background:#fffbe6;border:1px solid #e6d800;"
            "border-radius:3px;padding:4px;font-size:11px;"
        )
        sb_layout.addWidget(det_box)
        sb_layout.addWidget(self._seg_help)

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

        # ── Right tabs ───────────────────────────────────────────────────────
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

    # ── Help ──────────────────────────────────────────────────────────────────
    def _toggle_help(self, checked: bool):
        self._file_panel.set_help_visible(checked)
        self._col_panel.set_help_visible(checked)
        self._seg_help.setVisible(checked)
        self._orders_panel.set_help_visible(checked)
        self._preview_panel.set_help_visible(checked)
        self._fft_tab.set_help_visible(checked)
        self._order_tab.set_help_visible(checked)

    # ── Detect Events toggle ──────────────────────────────────────────────────
    def _on_detect_toggled(self, checked: bool):
        self._det_opts.setVisible(checked)

    # ── Auto PPR ──────────────────────────────────────────────────────────────
    def _auto_detect_spr(self):
        entries = self._file_panel.get_entries()
        if not entries:
            return
        col_map = self._col_panel.get_mapping()
        ang_col = col_map.get("angle")
        if not ang_col:
            QMessageBox.warning(
                self, "No angle column",
                "Auto-detect PPR requires an Angle/Encoder column to be mapped."
            )
            return
        try:
            df = pd.read_csv(entries[0].file_path)
            if ang_col not in df.columns:
                return
            ppr = detect_encoder_ppr(df[ang_col].values.astype(float))
            ppr_pow2 = 2 ** round(math.log2(max(ppr, 1)))
            self._spr_spin.setValue(max(64, min(8192, ppr_pow2)))
        except Exception as exc:
            QMessageBox.warning(self, "Auto-detect failed", str(exc))

    # ── Files changed ─────────────────────────────────────────────────────────
    def _on_files_changed(self):
        entries = self._file_panel.get_entries()
        if not entries:
            return
        try:
            df = pd.read_csv(entries[0].file_path, nrows=5)
            cols = list(df.columns)
            self._all_columns = cols
            self._col_panel.update_columns(cols)
            # Repopulate trigger combobox
            prev = self._trigger_cb.currentText()
            self._trigger_cb.blockSignals(True)
            self._trigger_cb.clear()
            self._trigger_cb.addItems(cols)
            if prev in cols:
                self._trigger_cb.setCurrentText(prev)
            self._trigger_cb.blockSignals(False)
        except Exception:
            pass

    # ── Preview ───────────────────────────────────────────────────────────────
    def _open_preview(self):
        entries = self._file_panel.get_entries()
        if not entries:
            QMessageBox.warning(self, "No files", "Add at least one data file first.")
            return
        col_map = self._col_panel.get_mapping()
        if not col_map.get("time"):
            QMessageBox.warning(self, "Columns", "Please map the Time column.")
            return

        # Run detection on all files if enabled
        events_by_file: dict[str, list[SegmentWindow]] = {}
        detect_enabled = self._detect_chk.isChecked()
        trigger_col = self._trigger_cb.currentText() if detect_enabled else None
        threshold = self._thresh_spin.value()
        pre_s = self._pre_spin.value()
        post_s = self._post_spin.value()

        for entry in entries:
            # Use confirmed edits if available, else re-detect
            if entry.file_path in self._event_windows:
                events_by_file[entry.file_path] = self._event_windows[entry.file_path]
            elif detect_enabled and trigger_col:
                try:
                    df = pd.read_csv(entry.file_path)
                    t_col = col_map.get("time")
                    if t_col and t_col in df.columns and trigger_col in df.columns:
                        time = df[t_col].values.astype(float)
                        trig = df[trigger_col].values.astype(float)
                        events_by_file[entry.file_path] = detect_events(
                            time, trig, threshold=threshold,
                            pre_window_s=pre_s, post_window_s=post_s,
                        )
                except Exception:
                    events_by_file[entry.file_path] = []
            else:
                events_by_file[entry.file_path] = []

        self._preview_panel.set_files(
            [e.file_path for e in entries],
            col_map,
            events_by_file,
            trigger_col=trigger_col,
            threshold=threshold,
        )
        self._tabs.setCurrentWidget(self._preview_panel)

    def _on_selection_confirmed(self, events_by_file: dict):
        self._event_windows.update(events_by_file)

    # ── Analysis ──────────────────────────────────────────────────────────────
    def _run_analysis(self):
        entries = self._file_panel.get_entries()
        if not entries:
            QMessageBox.warning(self, "No files", "Add at least one data file first.")
            return
        col_map = self._col_panel.get_mapping()
        if not col_map.get("time"):
            QMessageBox.warning(self, "Columns", "Please map the Time column.")
            return
        if not col_map.get("axes"):
            QMessageBox.warning(self, "Columns", "Please enable at least one Axis column.")
            return
        if self._thread and self._thread.isRunning():
            return

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")

        self._worker = _Worker(
            entries=entries,
            col_map=col_map,
            event_windows=dict(self._event_windows),
            samples_per_rev=self._spr_spin.value(),
            detect_enabled=self._detect_chk.isChecked(),
            trigger_col=self._trigger_cb.currentText() if self._detect_chk.isChecked() else None,
            threshold=self._thresh_spin.value(),
            pre_s=self._pre_spin.value(),
            post_s=self._post_spin.value(),
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
