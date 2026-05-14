from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QComboBox, QGroupBox

NONE_OPTION = "(None)"

_KEYWORDS: dict[str, list[str]] = {
    "time":  ["time", "t_", "_t", "sec", "s)"],
    "rpm":   ["rpm", "speed", "rot", "rev", "freq"],
    "angle": ["angle", "enc", "tach", "pulse", "deg"],
    "accel": ["accel", "vib", "m/s2", "ax_", "ay_", "az_",
               "x_accel", "y_accel", "z_accel", "ch1"],
}


def auto_detect_columns(columns: list[str]) -> dict[str, str | None]:
    col_lower = [c.lower() for c in columns]
    result: dict[str, str | None] = {k: None for k in _KEYWORDS}
    for role, kws in _KEYWORDS.items():
        for c, cl in zip(columns, col_lower):
            if any(kw in cl for kw in kws):
                result[role] = c
                break
    return result


class ColumnPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        box = QGroupBox("Column Mapping")
        form = QFormLayout()
        self._combos: dict[str, QComboBox] = {}
        for role, label in [
            ("time", "Time"),
            ("rpm", "RPM"),
            ("angle", "Angle (optional)"),
            ("accel", "Acceleration"),
        ]:
            cb = QComboBox()
            self._combos[role] = cb
            form.addRow(label + ":", cb)
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

    def update_columns(self, columns: list[str]):
        detected = auto_detect_columns(columns)
        for role, cb in self._combos.items():
            cb.blockSignals(True)
            cb.clear()
            if role == "angle":
                cb.addItem(NONE_OPTION)
            cb.addItems(columns)
            det = detected.get(role)
            if det and det in columns:
                cb.setCurrentText(det)
            elif role == "angle":
                cb.setCurrentText(NONE_OPTION)
            cb.blockSignals(False)

    def get_mapping(self) -> dict[str, str | None]:
        m = {}
        for role, cb in self._combos.items():
            val = cb.currentText()
            m[role] = None if val == NONE_OPTION else val
        return m
