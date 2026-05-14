from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QGroupBox, QCheckBox, QLabel,
)

NONE_OPTION = "(None)"

_KW_FIXED: dict[str, list[str]] = {
    "time":  ["time", "t_", "_t", "sec", "s)"],
    "rpm":   ["rpm", "speed", "rot", "rev", "freq"],
    "angle": ["angle", "enc", "tach", "pulse", "deg"],
}
_KW_AXES: dict[str, list[str]] = {
    "X": ["x_accel", "x accel", "accel_x", "ch1", "_x)", "x_", " x)", "(x"],
    "Y": ["y_accel", "y accel", "accel_y", "ch2", "_y)", "y_", " y)", "(y"],
    "Z": ["z_accel", "z accel", "accel_z", "ch3", "_z)", "z_", " z)", "(z"],
}


def auto_detect_columns(columns: list[str]) -> dict[str, str | None]:
    cl = [c.lower() for c in columns]
    result: dict[str, str | None] = {}
    for role, kws in _KW_FIXED.items():
        result[role] = next(
            (c for c, low in zip(columns, cl) if any(kw in low for kw in kws)), None
        )
    for ax, kws in _KW_AXES.items():
        result[ax] = next(
            (c for c, low in zip(columns, cl) if any(kw in low for kw in kws)), None
        )
    return result


class ColumnPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        box = QGroupBox("Column Mapping")
        form = QFormLayout()

        self._fixed_combos: dict[str, QComboBox] = {}
        for role, lbl in [("time", "Time"), ("rpm", "RPM")]:
            cb = QComboBox()
            self._fixed_combos[role] = cb
            form.addRow(lbl + ":", cb)

        self._angle_cb = QComboBox()
        self._fixed_combos["angle"] = self._angle_cb
        form.addRow("Angle (optional):", self._angle_cb)

        self._order_src_cb = QComboBox()
        self._order_src_cb.addItems(["Angle column", "Integrate from RPM"])
        self._order_src_cb.currentIndexChanged.connect(self._on_src_change)
        form.addRow("Order source:", self._order_src_cb)

        self._axis_checks: dict[str, QCheckBox] = {}
        self._axis_combos: dict[str, QComboBox] = {}
        for ax in ("X", "Y", "Z"):
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            chk = QCheckBox()
            chk.setChecked(False)
            acb = QComboBox()
            acb.addItem(NONE_OPTION)
            row.addWidget(chk)
            row.addWidget(acb, stretch=1)
            self._axis_checks[ax] = chk
            self._axis_combos[ax] = acb
            form.addRow(f"Axis {ax}:", row_w)

        box.setLayout(form)

        self._help_label = QLabel(
            "Map your CSV columns. Angle or RPM required for order analysis. "
            "Select ‘Integrate from RPM’ if you have no encoder. "
            "Tick Axis X/Y/Z for each acceleration channel to include."
        )
        self._help_label.setWordWrap(True)
        self._help_label.setVisible(False)
        self._help_label.setStyleSheet(
            "background:#fffbe6;border:1px solid #e6d800;"
            "border-radius:3px;padding:4px;font-size:11px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        layout.addWidget(self._help_label)

    def _on_src_change(self, idx: int):
        self._angle_cb.setEnabled(idx == 0)

    def set_help_visible(self, visible: bool):
        self._help_label.setVisible(visible)

    def update_columns(self, columns: list[str]):
        detected = auto_detect_columns(columns)
        for role, cb in self._fixed_combos.items():
            cb.blockSignals(True)
            cb.clear()
            if role == "angle":
                cb.addItem(NONE_OPTION)
            cb.addItems(columns)
            det = detected.get(role)
            if det:
                cb.setCurrentText(det)
            elif role == "angle":
                cb.setCurrentText(NONE_OPTION)
            cb.blockSignals(False)

        for ax in ("X", "Y", "Z"):
            cb = self._axis_combos[ax]
            chk = self._axis_checks[ax]
            cb.blockSignals(True)
            cb.clear()
            cb.addItem(NONE_OPTION)
            cb.addItems(columns)
            det = detected.get(ax)
            if det:
                cb.setCurrentText(det)
                chk.setChecked(True)
            else:
                cb.setCurrentText(NONE_OPTION)
                chk.setChecked(False)
            cb.blockSignals(False)

    def get_mapping(self) -> dict:
        m: dict = {}
        for role, cb in self._fixed_combos.items():
            val = cb.currentText()
            m[role] = None if val == NONE_OPTION else val
        m["order_source"] = (
            "rpm_integrate" if self._order_src_cb.currentIndex() == 1 else "angle"
        )
        axes: dict[str, str] = {}
        for ax in ("X", "Y", "Z"):
            if self._axis_checks[ax].isChecked():
                val = self._axis_combos[ax].currentText()
                if val != NONE_OPTION:
                    axes[ax] = val
        m["axes"] = axes
        return m
