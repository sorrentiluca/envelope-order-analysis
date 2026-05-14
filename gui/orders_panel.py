from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QColorDialog, QLabel,
)
from PySide6.QtGui import QColor
from app.marker_model import FaultOrder

_PRESET_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
]


class OrdersPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        box = QGroupBox("Fault Orders")
        inner = QVBoxLayout()

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Order ×", "Harmonics", "Colour", ""]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, 36)
        self._table.setColumnWidth(4, 28)
        self._table.verticalHeader().setVisible(False)
        inner.addWidget(self._table)

        add_btn = QPushButton("+ Add Order")
        add_btn.clicked.connect(lambda: self._append_row())
        inner.addWidget(add_btn)
        box.setLayout(inner)

        self._help_label = QLabel(
            "Add fundamental shaft orders (e.g. 5.35× for ball-pass, 20× for gear mesh). "
            "Vertical lines appear at each harmonic multiple. Each order gets its own colour."
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
        self._color_idx = 0

    def set_help_visible(self, visible: bool):
        self._help_label.setVisible(visible)

    def _append_row(self, fo: FaultOrder | None = None):
        if fo is None:
            color = _PRESET_COLORS[self._color_idx % len(_PRESET_COLORS)]
            self._color_idx += 1
            fo = FaultOrder(color=color)

        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(fo.name))
        self._table.setItem(row, 1, QTableWidgetItem(str(fo.fundamental)))
        self._table.setItem(row, 2, QTableWidgetItem(str(fo.n_harmonics)))

        color_btn = QPushButton()
        color_btn.setStyleSheet(
            f"background-color: {fo.color}; border: 1px solid #888;"
        )
        color_btn.setFixedWidth(32)
        color_btn.setProperty("hex_color", fo.color)
        color_btn.clicked.connect(lambda _=False, b=color_btn: self._pick_color(b))
        self._table.setCellWidget(row, 3, color_btn)

        del_btn = QPushButton("×")
        del_btn.setFixedWidth(24)
        del_btn.clicked.connect(self._delete_row)
        self._table.setCellWidget(row, 4, del_btn)

    def _pick_color(self, btn: QPushButton):
        current = QColor(btn.property("hex_color") or "#ff0000")
        color = QColorDialog.getColor(current, self, "Choose colour")
        if color.isValid():
            h = color.name()
            btn.setProperty("hex_color", h)
            btn.setStyleSheet(f"background-color: {h}; border: 1px solid #888;")

    def _delete_row(self):
        btn = self.sender()
        for r in range(self._table.rowCount()):
            if self._table.cellWidget(r, 4) is btn:
                self._table.removeRow(r)
                break

    def get_fault_orders(self) -> list[FaultOrder]:
        orders = []
        for row in range(self._table.rowCount()):
            try:
                name = self._table.item(row, 0).text()
                fundamental = float(self._table.item(row, 1).text())
                n_harmonics = int(self._table.item(row, 2).text())
                color_btn = self._table.cellWidget(row, 3)
                color = color_btn.property("hex_color") if color_btn else "#ff0000"
                orders.append(FaultOrder(
                    name=name, fundamental=fundamental,
                    n_harmonics=n_harmonics, color=color,
                ))
            except (ValueError, AttributeError):
                pass
        return orders
