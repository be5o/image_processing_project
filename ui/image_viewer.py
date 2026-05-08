"""Scrollable image viewer with optional ROI rubber-band selection.

Phase 1 public API (display_image) is preserved unchanged.
Phase 2 adds ROI drawing mode and a persistent overlay rectangle.
"""

import numpy as np
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QScrollArea
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from utils.image_convert import array_to_qimage


class ImageViewer(QWidget):
    """Scrollable image display widget.

    ROI Mode
    --------
    Call ``set_roi_mode(True)`` to enter rubber-band selection.  The user
    clicks and drags to draw a rectangle; on mouse-release the ``roi_selected``
    signal fires with a QRect in **pixmap coordinates** (= display-scale coords).

    The caller (MainWindow) is responsible for converting those coords to image
    coordinates using the current zoom scale, and for calling
    ``set_roi_overlay(rect)`` to re-draw the confirmed rectangle after every
    subsequent display refresh.
    """

    roi_selected = pyqtSignal(object)   # emits QRect in pixmap coordinates

    def __init__(self):
        super().__init__()
        self._base_pixmap: QPixmap | None = None
        self._roi_mode = False
        self._press_pt: QPoint | None = None
        self._drag_pt: QPoint | None = None
        self._roi_rect: QRect | None = None     # confirmed overlay rect

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMouseTracking(True)
        self.scroll.setWidget(self.label)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll)

        self.label.mousePressEvent   = self._on_press
        self.label.mouseMoveEvent    = self._on_move
        self.label.mouseReleaseEvent = self._on_release

    # ── public API ────────────────────────────────────────────────────────────

    def display_image(self, img_array: np.ndarray) -> None:
        """Convert numpy array to pixmap and render (with any active overlay)."""
        q_img = array_to_qimage(img_array)
        self._base_pixmap = QPixmap.fromImage(q_img)
        self._render()
        self.label.adjustSize()

    def set_roi_mode(self, enabled: bool) -> None:
        """Toggle rubber-band ROI drawing.  Changes cursor to crosshair."""
        self._roi_mode = enabled
        cur = Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        self.label.setCursor(cur)

    def set_roi_overlay(self, rect: QRect | None) -> None:
        """Draw (or clear) a persistent green ROI rectangle in pixmap coords."""
        self._roi_rect = rect
        self._render()

    def clear_roi(self) -> None:
        """Remove confirmed overlay rect and cancel any in-progress drag."""
        self._roi_rect = None
        self._press_pt = None
        self._drag_pt = None
        self._render()

    # ── private helpers ───────────────────────────────────────────────────────

    def _pixmap_offset(self) -> tuple[int, int]:
        """Return (ox, oy): top-left offset of pixmap inside the label."""
        if self._base_pixmap is None:
            return 0, 0
        ox = max(0, (self.label.width()  - self._base_pixmap.width())  // 2)
        oy = max(0, (self.label.height() - self._base_pixmap.height()) // 2)
        return ox, oy

    def _label_to_pixmap(self, pos: QPoint) -> QPoint | None:
        """Convert label-space QPoint to pixmap-space QPoint (clamped)."""
        if self._base_pixmap is None:
            return None
        ox, oy = self._pixmap_offset()
        rx = pos.x() - ox
        ry = pos.y() - oy
        pw = self._base_pixmap.width()
        ph = self._base_pixmap.height()
        if 0 <= rx <= pw and 0 <= ry <= ph:
            return QPoint(max(0, min(rx, pw - 1)),
                          max(0, min(ry, ph - 1)))
        return None

    def _on_press(self, event) -> None:
        if not self._roi_mode:
            return
        pt = self._label_to_pixmap(event.pos())
        if pt:
            self._press_pt = pt
            self._drag_pt  = pt
            self._roi_rect = None   # clear confirmed rect when new drag starts

    def _on_move(self, event) -> None:
        if not self._roi_mode or self._press_pt is None:
            return
        pt = self._label_to_pixmap(event.pos())
        if pt:
            self._drag_pt = pt
            self._render()

    def _on_release(self, event) -> None:
        if not self._roi_mode or self._press_pt is None:
            return
        pt = self._label_to_pixmap(event.pos())
        if pt:
            self._drag_pt = pt

        x1 = min(self._press_pt.x(), self._drag_pt.x())
        y1 = min(self._press_pt.y(), self._drag_pt.y())
        x2 = max(self._press_pt.x(), self._drag_pt.x())
        y2 = max(self._press_pt.y(), self._drag_pt.y())
        self._press_pt = None
        self._drag_pt  = None

        if x2 - x1 > 2 and y2 - y1 > 2:
            self._roi_rect = QRect(x1, y1, x2 - x1, y2 - y1)
            self._render()
            self.roi_selected.emit(self._roi_rect)

    def _render(self) -> None:
        """Compose base pixmap + active overlay, then push to label."""
        if self._base_pixmap is None:
            return

        # Determine rectangle to draw (drag preview > confirmed roi)
        draw_rect: QRect | None = None
        is_drag = False

        if self._press_pt and self._drag_pt:
            x1 = min(self._press_pt.x(), self._drag_pt.x())
            y1 = min(self._press_pt.y(), self._drag_pt.y())
            draw_rect = QRect(x1, y1,
                              abs(self._drag_pt.x() - self._press_pt.x()),
                              abs(self._drag_pt.y() - self._press_pt.y()))
            is_drag = True
        elif self._roi_rect:
            draw_rect = self._roi_rect

        if draw_rect and draw_rect.width() > 0 and draw_rect.height() > 0:
            pixmap = QPixmap(self._base_pixmap)
            painter = QPainter(pixmap)
            color = QColor(0, 160, 255) if is_drag else QColor(0, 230, 80)
            pen = QPen(color, 2)
            if is_drag:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(draw_rect)
            painter.end()
            self.label.setPixmap(pixmap)
        else:
            self.label.setPixmap(self._base_pixmap)
