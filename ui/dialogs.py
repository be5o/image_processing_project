"""Reusable dialog windows for metadata display, error reporting, and ROI stats."""

import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QMessageBox, QWidget)
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics
from PyQt6.QtCore import Qt
from typing import Dict, Any


class MetadataDialog(QDialog):
    """Modal dialog to display image metadata in a readable table."""
    def __init__(self, metadata: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Metadata")
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        title = QLabel("Extracted Metadata")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Property", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setRowCount(len(metadata))
        for row, (key, value) in enumerate(metadata.items()):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class ErrorDialog:
    """Static helper for consistent error messaging."""
    @staticmethod
    def show(parent, title: str, message: str, critical: bool = False):
        box = QMessageBox(parent)
        box.setWindowTitle(title)
        box.setText(message)
        box.setIcon(QMessageBox.Icon.Critical if critical else QMessageBox.Icon.Warning)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()


class ProcessingProgressDialog(QDialog):
    """Optional: Show progress for long operations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing...")
        self.setModal(True)
        self.setFixedSize(300, 100)
        layout = QVBoxLayout(self)
        self.label = QLabel("Applying filter...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)


# ── Phase 2: ROI Statistics ───────────────────────────────────────────────────

class HistogramWidget(QWidget):
    """Custom bar-chart histogram drawn entirely with QPainter (no matplotlib).

    Renders 256 vertical bars with:
    - Y-axis : count scale with 5 tick marks and numeric labels
    - X-axis : intensity labels at 0, 64, 128, 192, 255
    - Faint horizontal grid lines for readability
    - Dark background to match the application theme

    Call ``update_histogram(hist)`` to refresh in-place without recreating.
    """

    # Layout constants
    _PAD_LEFT   = 54   # room for Y-axis labels
    _PAD_BOTTOM = 28   # room for X-axis labels
    _PAD_TOP    = 10
    _PAD_RIGHT  = 8

    def __init__(self, hist: np.ndarray, parent=None):
        super().__init__(parent)
        self.hist = np.asarray(hist, dtype=np.int64)
        self.setMinimumSize(320, 200)

    # ── public ────────────────────────────────────────────────────────────────

    def update_histogram(self, hist: np.ndarray) -> None:
        """Replace the displayed histogram data and trigger a repaint."""
        self.hist = np.asarray(hist, dtype=np.int64)
        self.update()   # schedule a repaint

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        if self.hist is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        W, H = self.width(), self.height()

        # ── background ────────────────────────────────────────────────────────
        painter.fillRect(self.rect(), QColor(18, 18, 28))

        # ── drawing area (inside the axes) ────────────────────────────────────
        left   = self._PAD_LEFT
        top    = self._PAD_TOP
        right  = W - self._PAD_RIGHT
        bottom = H - self._PAD_BOTTOM
        draw_w = right - left
        draw_h = bottom - top

        if draw_w <= 0 or draw_h <= 0:
            painter.end()
            return

        max_val = float(self.hist.max()) if self.hist.max() > 0 else 1.0

        # ── axis fonts ────────────────────────────────────────────────────────
        font = QFont("Arial", 8)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # ── Y-axis grid lines + tick labels (5 intervals) ─────────────────────
        n_y_ticks = 5
        for i in range(n_y_ticks + 1):
            frac  = i / n_y_ticks
            y_px  = bottom - int(frac * draw_h)
            count = int(frac * max_val)

            # grid line
            painter.setPen(QPen(QColor(45, 45, 65), 1, Qt.PenStyle.DotLine))
            painter.drawLine(left, y_px, right, y_px)

            # tick mark
            painter.setPen(QPen(QColor(140, 140, 160), 1))
            painter.drawLine(left - 4, y_px, left, y_px)

            # label (right-aligned into the left padding)
            label = _fmt_count(count)
            lw    = fm.horizontalAdvance(label)
            painter.drawText(left - 6 - lw, y_px + fm.ascent() // 2, label)

        # ── X-axis tick labels ─────────────────────────────────────────────────
        x_ticks = [0, 64, 128, 192, 255]
        painter.setPen(QPen(QColor(140, 140, 160), 1))
        for val in x_ticks:
            x_px = left + int(val * draw_w / 255)
            # tick mark
            painter.drawLine(x_px, bottom, x_px, bottom + 4)
            # label (centred)
            label = str(val)
            lw    = fm.horizontalAdvance(label)
            painter.drawText(x_px - lw // 2, bottom + 4 + fm.ascent(), label)

        # ── axes border ───────────────────────────────────────────────────────
        painter.setPen(QPen(QColor(100, 100, 130), 1))
        painter.drawRect(left, top, draw_w, draw_h)

        # ── histogram bars ────────────────────────────────────────────────────
        bar_w = max(1.0, draw_w / 256.0)
        painter.setPen(Qt.PenStyle.NoPen)
        bar_color = QColor(60, 140, 255)
        for i in range(256):
            if self.hist[i] == 0:
                continue
            bar_h = int((self.hist[i] / max_val) * draw_h)
            x = left + int(i * draw_w / 256)
            y = bottom - bar_h
            # Gradient tint: brighter near peak
            bright = int(80 + 175 * (self.hist[i] / max_val))
            painter.setBrush(QColor(max(0, bright - 80), max(0, bright - 30), bright))
            painter.drawRect(x, y, max(1, int(bar_w)), bar_h)

        # ── axis titles ───────────────────────────────────────────────────────
        title_font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(180, 180, 200), 1))

        # X title
        x_title = "Intensity (0–255)"
        xtw = QFontMetrics(title_font).horizontalAdvance(x_title)
        painter.drawText(left + (draw_w - xtw) // 2, H - 2, x_title)

        # Y title (rotated)
        painter.save()
        painter.translate(12, top + draw_h // 2)
        painter.rotate(-90)
        y_title = "Count"
        ytw = QFontMetrics(title_font).horizontalAdvance(y_title)
        painter.drawText(-ytw // 2, 0, y_title)
        painter.restore()

        painter.end()


def _fmt_count(n: int) -> str:
    """Format a count value compactly (e.g. 12500 → '12.5k')."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


class RoiStatsDialog(QDialog):
    """Modal dialog that shows ROI *numerical* statistics only.

    The histogram is now displayed in the dedicated 'ROI Histogram' tab inside
    the main viewer area (see MainWindow.compute_roi_stats).  This dialog is
    kept for a quick textual summary that can be invoked programmatically.

    Accepts the dict returned by ``roi_stats.compute_roi_stats``.
    """

    def __init__(self, stats: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Statistics")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        title = QLabel("Region of Interest Analysis")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 4px;")
        layout.addWidget(title)

        info = (
            f"Pixel Count :  {stats['pixel_count']:,}\n"
            f"ROI Size    :  {stats['roi_shape'][1]} × {stats['roi_shape'][0]} px\n"
            f"Mean        :  {stats['mean']:.3f}\n"
            f"Variance    :  {stats['variance']:.3f}\n"
            f"Std Dev     :  {stats['std']:.3f}"
        )
        info_label = QLabel(info)
        info_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(info_label)

        note = QLabel("📊 Full histogram available in the 'ROI Histogram' tab.")
        note.setStyleSheet("color: #6ad; font-size: 10px; margin-top: 6px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
