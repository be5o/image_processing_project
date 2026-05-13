"""Clickable FFT magnitude-spectrum viewer for interactive notch filtering."""

import numpy as np
from PyQt6.QtWidgets import QWidget, QScrollArea, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap
from utils.image_convert import array_to_qimage


class FftViewer(QWidget):
    """Displays the log-magnitude FFT spectrum and lets the user mark notch
    positions by clicking.

    Design
    ------
    * Each left-click records (row, col) in *image* coordinates.
    * The viewer automatically draws a marker at the clicked point **and** at
      its conjugate-symmetric counterpart (rows-u, cols-v mod shape), ensuring
      that the resulting inverse-FFT is real-valued.
    * ``get_notch_points()`` returns the raw click list; the symmetric partners
      are computed by ``frequency_filters.build_notch_filter``.

    Signals
    -------
    notch_clicked(row, col)  – emitted with image-space coordinates on every click.
    """

    notch_clicked = pyqtSignal(int, int)  # (row, col) in image/pixmap coordinates of the click

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pixmap: QPixmap | None = None # the original spectrum image without any notch markers
        self._img_h = 0
        self._img_w = 0
        self._notch_points: list[tuple[int, int]] = []    # list of (row, col) points where the user clicked to place notches

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)     # the pixmap is smaller than the label, so it will be centered and we can compute click offsets

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll)

        self.label.mousePressEvent = self._on_label_click

    # ── public API ────────────────────────────────────────────────────────────

    def display_spectrum(self, spectrum_array: np.ndarray) -> None:
        """Show a new spectrum image and reset all notch markers."""
        self._img_h, self._img_w = spectrum_array.shape[:2]
        self._notch_points = []
        qimg = array_to_qimage(spectrum_array)
        self._base_pixmap = QPixmap.fromImage(qimg)
        self._redraw()

    def clear_notches(self) -> None:
        """Remove all notch markers from the display and the internal list."""
        self._notch_points = []
        self._redraw()

    def get_notch_points(self) -> list[tuple[int, int]]:
        """Return a copy of the recorded (row, col) notch positions."""
        return list(self._notch_points)

    # ── private helpers ───────────────────────────────────────────────────────

    def _on_label_click(self, event) -> None:
        if self._base_pixmap is None:
            return
        img_pt = self._label_to_image(event.pos())
        if img_pt is None:
            return
        row, col = img_pt
        self._notch_points.append((row, col))
        self._redraw()
        self.notch_clicked.emit(row, col)

    def _label_to_image(self, pos: QPoint):
        """Map a QPoint in label space to (row, col) in image/pixmap space.

        The pixmap is displayed at its native resolution inside a larger label
        (AlignCenter).  We subtract the centering offset then clamp to bounds.
        """
        lw = self.label.width()
        lh = self.label.height()
        pw = self._base_pixmap.width()
        ph = self._base_pixmap.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        rx = pos.x() - ox
        ry = pos.y() - oy
        if 0 <= rx < pw and 0 <= ry < ph:
            col = max(0, min(rx, self._img_w - 1))
            row = max(0, min(ry, self._img_h - 1))
            return row, col
        return None

    def _redraw(self) -> None:
        """Recompose base pixmap + all notch markers and push to the label."""
        if self._base_pixmap is None:
            return
        pixmap = QPixmap(self._base_pixmap)     # shallow copy

        if self._notch_points:
            rows, cols = self._img_h, self._img_w
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            for (row, col) in self._notch_points:
                sym_row = int((rows - row) % rows)
                sym_col = int((cols - col) % cols)
                for (cy, cx) in [(row, col), (sym_row, sym_col)]:
                    r = 8
                    # Red circle
                    painter.setPen(QPen(QColor(255, 60, 60), 2))
                    painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
                    # Yellow crosshair
                    painter.setPen(QPen(QColor(255, 220, 0), 1))
                    painter.drawLine(cx - r - 5, cy, cx + r + 5, cy)
                    painter.drawLine(cx, cy - r - 5, cx, cy + r + 5)

            painter.end()

        self.label.setPixmap(pixmap)
        self.label.adjustSize()
