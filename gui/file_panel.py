from __future__ import annotations
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QFileDialog, QLabel, QHeaderView,
)
from PySide6.QtCore import Qt, Signal
from app.category_model import FileEntry


class FilePanel(QWidget):
    files_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Data Files"))

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Files…")
        self._clear_btn = QPushButton("Clear All")
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["File", "Category", ""])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(2, 28)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        self._help_label = QLabel(
            "Load one or more CSV files. Assign each file a category label — "
            "files with the same label are averaged together on the spectrum plots."
        )
        self._help_label.setWordWrap(True)
        self._help_label.setVisible(False)
        self._help_label.setStyleSheet(
            "background:#fffbe6;border:1px solid #e6d800;"
            "border-radius:3px;padding:4px;font-size:11px;"
        )
        layout.addWidget(self._help_label)

        self._add_btn.clicked.connect(self._add_files)
        self._clear_btn.clicked.connect(self._clear)

    def set_help_visible(self, visible: bool):
        self._help_label.setVisible(visible)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select data files", "",
            "Data files (*.csv *.tsv *.txt *.dxd);;All files (*)"
        )
        for p in paths:
            self._append_row(p)
        if paths:
            self.files_changed.emit()

    def _append_row(self, path: str):
        row = self._table.rowCount()
        self._table.insertRow(row)
        name_item = QTableWidgetItem(os.path.basename(path))
        name_item.setData(Qt.ItemDataRole.UserRole, path)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, name_item)
        self._table.setItem(row, 1, QTableWidgetItem("Default"))
        del_btn = QPushButton("×")
        del_btn.setFixedWidth(24)
        del_btn.clicked.connect(self._delete_row)
        self._table.setCellWidget(row, 2, del_btn)

    def _delete_row(self):
        btn = self.sender()
        for r in range(self._table.rowCount()):
            if self._table.cellWidget(r, 2) is btn:
                self._table.removeRow(r)
                self.files_changed.emit()
                break

    def _clear(self):
        self._table.setRowCount(0)
        self.files_changed.emit()

    def get_entries(self) -> list[FileEntry]:
        entries = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            cat_item = self._table.item(row, 1)
            if item:
                entries.append(FileEntry(
                    file_path=item.data(Qt.ItemDataRole.UserRole),
                    category=cat_item.text() if cat_item else "Default",
                ))
        return entries
