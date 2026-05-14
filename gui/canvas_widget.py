from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class CanvasWidget(QWidget):
    def __init__(self, figsize=(8, 4), parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=figsize, tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)
        layout.setContentsMargins(0, 0, 0, 0)

    def figure(self) -> Figure:
        return self._fig

    def refresh(self):
        self._canvas.draw_idle()

    def clear(self):
        self._fig.clear()
        self._canvas.draw_idle()
