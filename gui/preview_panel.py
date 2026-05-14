from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QGroupBox, QComboBox,
)
from PySide6.QtCore import Qt, Signal

from gui.canvas_widget import CanvasWidget
from app.segmentation_engine import SegmentWindow


class PreviewPanel(QWidget):
    """Shows signal traces and detected event windows. Detection settings live in the sidebar."""

    selection_confirmed = Signal(dict)  # {file_path: list[SegmentWindow]}

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._canvas = CanvasWidget(figsize=(10, 4))
        layout.addWidget(self._canvas, stretch=3)

        # File selector
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._file_combo = QComboBox()
        self._file_combo.currentIndexChanged.connect(self._on_file_changed)
        file_row.addWidget(self._file_combo, stretch=1)
        layout.addLayout(file_row)

        # Help label
        self._help_label = QLabel(
            "Events are detected automatically using the trigger settings in the sidebar. "
            "Green shading = event window. Uncheck or edit bounds, then click Confirm."
        )
        self._help_label.setWordWrap(True)
        self._help_label.setVisible(False)
        self._help_label.setStyleSheet(
            "background:#fffbe6;border:1px solid #e6d800;"
            "border-radius:3px;padding:4px;font-size:11px;"
        )
        layout.addWidget(self._help_label)

        # Event table
        box = QGroupBox("Detected Events — uncheck to exclude; edit t start/end to adjust bounds")
        inner = QVBoxLayout()
        self._event_table = QTableWidget(0, 4)
        self._event_table.setHorizontalHeaderLabels(
            ["Include", "t start (s)", "t end (s)", "Mean trigger"]
        )
        hdr = self._event_table.horizontalHeader()
        for col in range(4):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._event_table.verticalHeader().setVisible(False)
        self._event_table.cellChanged.connect(self._on_cell_changed)
        inner.addWidget(self._event_table)
        confirm_btn = QPushButton("Confirm Selection  →  Use in Analysis")
        confirm_btn.clicked.connect(self._confirm)
        inner.addWidget(confirm_btn)
        box.setLayout(inner)
        layout.addWidget(box, stretch=2)

        self._col_map: dict = {}
        self._events_by_file: dict[str, list[SegmentWindow]] = {}
        self._file_paths: list[str] = []
        self._trigger_col: Optional[str] = None
        self._threshold: float = 100.0
        self._df_cache: dict[str, pd.DataFrame] = {}

    def set_help_visible(self, visible: bool):
        self._help_label.setVisible(visible)

    def set_files(
        self,
        file_paths: list[str],
        col_map: dict,
        events_by_file: dict[str, list[SegmentWindow]],
        trigger_col: Optional[str] = None,
        threshold: float = 100.0,
    ):
        self._col_map = col_map
        self._file_paths = list(file_paths)
        self._events_by_file = {p: list(evs) for p, evs in events_by_file.items()}
        self._trigger_col = trigger_col
        self._threshold = threshold
        self._df_cache.clear()

        self._file_combo.blockSignals(True)
        self._file_combo.clear()
        for p in file_paths:
            self._file_combo.addItem(os.path.basename(p), userData=p)
        self._file_combo.blockSignals(False)

        if file_paths:
            self._on_file_changed(0)

    def _current_path(self) -> Optional[str]:
        return self._file_combo.currentData()

    def _load_df(self, path: str) -> Optional[pd.DataFrame]:
        if path not in self._df_cache:
            try:
                self._df_cache[path] = pd.read_csv(path)
            except Exception:
                return None
        return self._df_cache[path]

    def _on_file_changed(self, index: int):
        self._populate_table()
        self._draw_traces()

    def _populate_table(self):
        path = self._current_path()
        events = self._events_by_file.get(path, [])
        self._event_table.blockSignals(True)
        self._event_table.setRowCount(0)
        for ev in events:
            row = self._event_table.rowCount()
            self._event_table.insertRow(row)
            chk = QTableWidgetItem()
            chk.setCheckState(
                Qt.CheckState.Checked if ev.enabled else Qt.CheckState.Unchecked
            )
            self._event_table.setItem(row, 0, chk)
            self._event_table.setItem(row, 1, QTableWidgetItem(f"{ev.t_start:.4f}"))
            self._event_table.setItem(row, 2, QTableWidgetItem(f"{ev.t_end:.4f}"))
            trig_item = QTableWidgetItem(f"{ev.mean_trigger:.2f}")
            trig_item.setFlags(trig_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._event_table.setItem(row, 3, trig_item)
        self._event_table.blockSignals(False)

    def _on_cell_changed(self, row: int, col: int):
        path = self._current_path()
        events = self._events_by_file.get(path, [])
        if row >= len(events):
            return
        ev = events[row]
        if col == 0:
            ev.enabled = (
                self._event_table.item(row, 0).checkState() == Qt.CheckState.Checked
            )
            self._draw_traces()
        elif col in (1, 2):
            df = self._load_df(path)
            if df is None:
                return
            t_col = self._col_map.get("time")
            if not t_col:
                return
            time = df[t_col].values.astype(float)
            try:
                val = float(self._event_table.item(row, col).text())
                if col == 1:
                    ev.t_start = val
                    ev.i_start = int(np.searchsorted(time, val))
                else:
                    ev.t_end = val
                    ev.i_end = int(np.searchsorted(time, val))
                self._draw_traces()
            except ValueError:
                pass

    def _draw_traces(self):
        fig = self._canvas.figure()
        fig.clear()
        path = self._current_path()
        if not path:
            self._canvas.refresh()
            return

        df = self._load_df(path)
        if df is None:
            self._canvas.refresh()
            return

        t_col = self._col_map.get("time")
        axes = self._col_map.get("axes", {})
        a_col = next(iter(axes.values()), None) if axes else None

        if not t_col:
            self._canvas.refresh()
            return

        time = df[t_col].values.astype(float)
        events = self._events_by_file.get(path, [])

        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        # Top: acceleration
        if a_col and a_col in df.columns:
            accel = df[a_col].values.astype(float)
            ax1.plot(time, accel, lw=0.5, color="steelblue")
            ax1.set_ylabel(a_col)
        ax1.set_title("Signal Preview", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Bottom: trigger channel
        trig_data = None
        if self._trigger_col and self._trigger_col in df.columns:
            trig_data = df[self._trigger_col].values.astype(float)
            ax2.plot(time, np.abs(trig_data), lw=0.7, color="darkorange")
            ax2.axhline(
                self._threshold, color="red", ls="--", lw=0.8, label=f"Threshold {self._threshold}"
            )
            ax2.set_ylabel(f"|{self._trigger_col}|")
            ax2.legend(fontsize=7)
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.3)

        # Event shading
        for ev in events:
            color = "green" if ev.enabled else "gray"
            alpha = 0.18 if ev.enabled else 0.06
            ax1.axvspan(ev.t_start, ev.t_end, alpha=alpha, color=color)
            ax2.axvspan(ev.t_start, ev.t_end, alpha=alpha, color=color)

        fig.tight_layout()
        self._canvas.refresh()

    def _confirm(self):
        self.selection_confirmed.emit(dict(self._events_by_file))
