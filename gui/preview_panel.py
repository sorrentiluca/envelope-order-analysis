from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QDoubleSpinBox, QGroupBox,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal

from gui.canvas_widget import CanvasWidget
from app.segmentation_engine import SegmentWindow, detect_events


class PreviewPanel(QWidget):
    selection_confirmed = Signal(str, list)  # (file_path, list[SegmentWindow])

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._canvas = CanvasWidget(figsize=(10, 5))
        layout.addWidget(self._canvas, stretch=3)

        # Detection controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("File:"))
        self._file_combo = QComboBox()
        self._file_combo.currentIndexChanged.connect(self._load_selected_file)
        ctrl.addWidget(self._file_combo, stretch=1)

        ctrl.addWidget(QLabel("RPM thresh:"))
        self._rpm_thresh = QDoubleSpinBox()
        self._rpm_thresh.setRange(0, 50000)
        self._rpm_thresh.setValue(100.0)
        self._rpm_thresh.setFixedWidth(70)
        ctrl.addWidget(self._rpm_thresh)

        ctrl.addWidget(QLabel("Min dur (s):"))
        self._min_dur = QDoubleSpinBox()
        self._min_dur.setRange(0, 3600)
        self._min_dur.setSingleStep(0.1)
        self._min_dur.setValue(0.1)
        self._min_dur.setFixedWidth(60)
        ctrl.addWidget(self._min_dur)

        ctrl.addWidget(QLabel("Speed tol (%):"))
        self._speed_tol = QDoubleSpinBox()
        self._speed_tol.setRange(0, 50)
        self._speed_tol.setSingleStep(0.5)
        self._speed_tol.setValue(5.0)
        self._speed_tol.setFixedWidth(60)
        ctrl.addWidget(self._speed_tol)

        detect_btn = QPushButton("Detect Events")
        detect_btn.clicked.connect(self._detect)
        ctrl.addWidget(detect_btn)
        layout.addLayout(ctrl)

        # Help label
        self._help_label = QLabel(
            "Load a file and click ‘Detect Events’ to find constant-speed regions. "
            "Green shading = event window, darker green = stable plateau. "
            "Uncheck events to exclude them. Edit t start/end to trim bounds. "
            "The bottom subplot shows the angle-resampled signal for a gut-check."
        )
        self._help_label.setWordWrap(True)
        self._help_label.setVisible(False)
        self._help_label.setStyleSheet(
            "background:#fffbe6;border:1px solid #e6d800;"
            "border-radius:3px;padding:4px;font-size:11px;"
        )
        layout.addWidget(self._help_label)

        # Event table
        box = QGroupBox("Detected Events — uncheck to exclude; edit t start/end to adjust")
        inner = QVBoxLayout()
        self._event_table = QTableWidget(0, 5)
        self._event_table.setHorizontalHeaderLabels(
            ["Include", "t start (s)", "t end (s)", "Mean RPM", "Plateau range"]
        )
        hdr = self._event_table.horizontalHeader()
        for col in range(4):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._event_table.verticalHeader().setVisible(False)
        self._event_table.cellChanged.connect(self._on_cell_changed)
        inner.addWidget(self._event_table)
        confirm_btn = QPushButton("Confirm Selection  →  Run Analysis")
        confirm_btn.clicked.connect(self._confirm)
        inner.addWidget(confirm_btn)
        box.setLayout(inner)
        layout.addWidget(box, stretch=2)

        self._col_map: dict = {}
        self._df: Optional[pd.DataFrame] = None
        self._time: Optional[np.ndarray] = None
        self._rpm: Optional[np.ndarray] = None
        self._accel: Optional[np.ndarray] = None
        self._angle: Optional[np.ndarray] = None
        self._events: list[SegmentWindow] = []
        self._file_paths: list[str] = []

    def set_help_visible(self, visible: bool):
        self._help_label.setVisible(visible)

    def set_files(self, file_paths: list[str], col_map: dict):
        self._col_map = col_map
        self._file_paths = list(file_paths)
        self._file_combo.blockSignals(True)
        self._file_combo.clear()
        for p in file_paths:
            self._file_combo.addItem(os.path.basename(p), userData=p)
        self._file_combo.blockSignals(False)
        if file_paths:
            self._load_selected_file(0)

    def _load_selected_file(self, index: int):
        if index < 0 or index >= self._file_combo.count():
            return
        path = self._file_combo.itemData(index)
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self._df = df
            t_col = self._col_map.get("time")
            r_col = self._col_map.get("rpm")
            ang_col = self._col_map.get("angle")
            axes = self._col_map.get("axes", {})
            a_col = next(iter(axes.values()), None) if axes else None
            if not t_col or not a_col:
                return
            self._time = df[t_col].values.astype(float)
            self._accel = df[a_col].values.astype(float)
            self._rpm = (
                df[r_col].values.astype(float)
                if r_col and r_col in df.columns
                else np.zeros_like(self._time)
            )
            self._angle = (
                df[ang_col].values.astype(float)
                if ang_col and ang_col in df.columns else None
            )
            self._events = []
            self._event_table.setRowCount(0)
            self._draw_traces()
        except Exception as exc:
            print(f"PreviewPanel: could not load {path}: {exc}")

    def _detect(self):
        if self._time is None:
            return
        self._events = detect_events(
            self._time, self._rpm,
            rpm_threshold=self._rpm_thresh.value(),
            min_duration=self._min_dur.value(),
            speed_tol=self._speed_tol.value() / 100.0,
        )
        self._populate_table()
        self._draw_traces()

    def _populate_table(self):
        self._event_table.blockSignals(True)
        self._event_table.setRowCount(0)
        for ev in self._events:
            row = self._event_table.rowCount()
            self._event_table.insertRow(row)
            chk = QTableWidgetItem()
            chk.setCheckState(
                Qt.CheckState.Checked if ev.enabled else Qt.CheckState.Unchecked
            )
            self._event_table.setItem(row, 0, chk)
            self._event_table.setItem(row, 1, QTableWidgetItem(f"{ev.t_start:.4f}"))
            self._event_table.setItem(row, 2, QTableWidgetItem(f"{ev.t_end:.4f}"))
            self._event_table.setItem(row, 3, QTableWidgetItem(f"{ev.mean_rpm:.1f}"))
            plateau_str = "—"
            if ev.plateau_t_start is not None:
                plateau_str = f"{ev.plateau_t_start:.4f} – {ev.plateau_t_end:.4f} s"
            p_item = QTableWidgetItem(plateau_str)
            p_item.setFlags(p_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._event_table.setItem(row, 4, p_item)
        self._event_table.blockSignals(False)

    def _on_cell_changed(self, row: int, col: int):
        if row >= len(self._events):
            return
        ev = self._events[row]
        if col == 0:
            ev.enabled = (
                self._event_table.item(row, 0).checkState() == Qt.CheckState.Checked
            )
            self._draw_traces()
        elif col in (1, 2) and self._time is not None:
            try:
                val = float(self._event_table.item(row, col).text())
                if col == 1:
                    ev.t_start = val
                    ev.i_start = int(np.searchsorted(self._time, val))
                else:
                    ev.t_end = val
                    ev.i_end = int(np.searchsorted(self._time, val))
                self._draw_traces()
            except ValueError:
                pass

    def _try_resample_preview(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self._time is None or self._accel is None:
            return None
        enabled = [ev for ev in self._events if ev.enabled]
        if enabled:
            i0, i1 = enabled[0].i_start, enabled[0].i_end
        else:
            i0, i1 = 0, min(2000, len(self._time))
        t_sl = self._time[i0:i1]
        sig_sl = self._accel[i0:i1]
        if len(t_sl) < 10:
            return None
        rpm_sl = self._rpm[i0:i1] if self._rpm is not None else None
        order_src = self._col_map.get("order_source", "angle")
        try:
            from app.order_engine import rpm_to_angle
            from core.config import Config
            from core.resampling import resample_to_uniform_angle
            from core.io import estimate_fs
            if order_src == "rpm_integrate" and rpm_sl is not None:
                angle = rpm_to_angle(t_sl, rpm_sl)
            elif order_src == "angle" and self._angle is not None:
                angle = self._angle[i0:i1]
            else:
                return None
            rpm_for_rs = rpm_sl if rpm_sl is not None else np.ones_like(sig_sl) * 1000.0
            fs = estimate_fs(t_sl)
            result = resample_to_uniform_angle(angle, sig_sl, t_sl, rpm_for_rs, fs, Config())
            if result is None:
                return None
            unif_angle, accel_res, *_ = result
            return unif_angle, accel_res
        except Exception:
            return None

    def _draw_traces(self):
        fig = self._canvas.figure()
        fig.clear()
        if self._time is None:
            self._canvas.refresh()
            return

        resample_data = self._try_resample_preview()
        n = 3 if resample_data is not None else 2
        axs = fig.subplots(n, 1)
        if n == 2:
            ax1, ax2 = axs
        else:
            ax1, ax2, ax3 = axs

        ax1.plot(self._time, self._accel, lw=0.5, color="steelblue")
        ax1.set_ylabel("Acceleration")
        ax1.set_title("Signal Preview", fontsize=9)

        ax2.plot(self._time, np.abs(self._rpm), lw=0.7, color="darkorange")
        ax2.axhline(
            self._rpm_thresh.value(), color="red", ls="--", lw=0.8, label="Threshold"
        )
        ax2.set_ylabel("|RPM|")
        ax2.legend(fontsize=7)
        if resample_data is None:
            ax2.set_xlabel("Time (s)")

        for ev in self._events:
            color = "green" if ev.enabled else "gray"
            alpha = 0.15 if ev.enabled else 0.06
            for sp in (ax1, ax2):
                sp.axvspan(ev.t_start, ev.t_end, alpha=alpha, color=color)
            if ev.plateau_t_start is not None and ev.enabled:
                for sp in (ax1, ax2):
                    sp.axvspan(ev.plateau_t_start, ev.plateau_t_end, alpha=0.3, color="green")

        if resample_data is not None:
            unif_angle, accel_res = resample_data
            ax3.plot(unif_angle, accel_res, lw=0.5, color="purple")
            ax3.set_ylabel("Accel (resampled)")
            ax3.set_xlabel("Cumulative angle (°)")
            ax3.set_title("Angle-resampled signal (gut check)", fontsize=9)
            ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        self._canvas.refresh()

    def _confirm(self):
        path = self._file_combo.currentData()
        selected = [ev for ev in self._events if ev.enabled]
        if path is not None:
            self.selection_confirmed.emit(path, selected)
