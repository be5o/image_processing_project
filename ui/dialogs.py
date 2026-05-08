"""Reusable dialog windows for metadata display, error reporting, and ROI stats."""

import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QMessageBox, QWidget)
from PyQt6.QtGui import QPainter, QPen, QColor
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

    Renders 256 vertical bars whose heights are proportional to each bin's
    frequency.  The background is dark to match the application theme.
    """

    def __init__(self, hist: np.ndarray, parent=None):
        super().__init__(parent)
        self.hist = hist
        self.setMinimumSize(256, 130)
        self.setMaximumHeight(160)

    def paintEvent(self, _event) -> None:
        if self.hist is None:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(22, 22, 32))

        w, h = self.width(), self.height()
        pad = 4
        draw_h = h - 2 * pad
        draw_w = w - 2 * pad
        max_val = float(self.hist.max()) if self.hist.max() > 0 else 1.0
        bar_w = max(1.0, draw_w / 256.0)

        painter.setPen(QPen(QColor(70, 150, 255), 1))
        for i in range(256):
            if self.hist[i] == 0:
                continue
            bar_h = int((self.hist[i] / max_val) * draw_h)
            x = pad + int(i * draw_w / 256)
            y = h - pad - bar_h
            painter.drawRect(x, y, max(1, int(bar_w)), bar_h)

        painter.end()


class RoiStatsDialog(QDialog):
    """Modal dialog that shows ROI numerical statistics and a histogram.

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

        layout.addWidget(QLabel("Intensity Histogram:"))
        layout.addWidget(HistogramWidget(stats['histogram']))

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
