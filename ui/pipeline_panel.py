"""Pipeline visualization panel — horizontal scrollable node strip.

Placed below the main image viewer.  Shows one clickable node per
HistoryEntry in the active branch, with forward/backward navigation and
a branch-selector combo box.

Signals
-------
step_clicked(branch_idx, step_idx)  user clicked a history node
branch_changed_sig(branch_idx)      user selected a different branch
nav_prev()                          ◀ button clicked
nav_next()                          ▶ button clicked
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


# ── Single step node ──────────────────────────────────────────────────────────

class _StepNode(QPushButton):
    """One clickable node in the strip."""

    _STYLE_ACTIVE = """
        QPushButton {
            background: #1a6fbf;
            color: white;
            border: 2px solid #4da6ff;
            border-radius: 5px;
            font-weight: bold;
            padding: 2px 8px;
            min-width: 80px;
        }
    """
    _STYLE_NORMAL = """
        QPushButton {
            background: #2d2d3a;
            color: #cccccc;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 2px 8px;
            min-width: 80px;
        }
        QPushButton:hover {
            background: #3a3a50;
            border: 1px solid #7a7aaa;
        }
    """

    def __init__(self, step_idx: int, label: str,
                 is_current: bool = False, parent=None):
        super().__init__(parent)
        # Truncate long labels so nodes stay compact
        short = label if len(label) <= 16 else label[:14] + "…"
        self.setText(f"[{step_idx}] {short}")
        self.setToolTip(f"Step {step_idx}: {label}")
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(self._STYLE_ACTIVE if is_current else self._STYLE_NORMAL)


# ── Panel ─────────────────────────────────────────────────────────────────────

class PipelinePanel(QWidget):
    """Compact panel that visualises the processing pipeline."""

    step_clicked       = pyqtSignal(int, int)   # (branch_idx, step_idx)
    branch_changed_sig = pyqtSignal(int)         # branch_idx
    nav_prev           = pyqtSignal()
    nav_next           = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(78)
        self._current_branch_idx = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.setSpacing(2)

        # ── Top row: branch selector + nav controls ───────────────────────
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(6)

        branch_lbl = QLabel("Branch:")
        branch_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        top_lay.addWidget(branch_lbl)

        self.branch_combo = QComboBox()
        self.branch_combo.setFixedWidth(160)
        self.branch_combo.setFixedHeight(22)
        self.branch_combo.setStyleSheet("font-size: 11px;")
        self.branch_combo.currentIndexChanged.connect(self._on_combo_changed)
        top_lay.addWidget(self.branch_combo)

        top_lay.addStretch()

        nav_lbl = QLabel("Navigate:")
        nav_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        top_lay.addWidget(nav_lbl)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedSize(30, 22)
        self.btn_prev.setToolTip("Previous step")
        self.btn_prev.clicked.connect(self.nav_prev)
        top_lay.addWidget(self.btn_prev)

        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedSize(30, 22)
        self.btn_next.setToolTip("Next step")
        self.btn_next.clicked.connect(self.nav_next)
        top_lay.addWidget(self.btn_next)

        outer.addWidget(top)

        # ── Bottom row: scrollable node strip ────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setFixedHeight(40)

        self._strip = QWidget()
        self._strip_lay = QHBoxLayout(self._strip)
        self._strip_lay.setContentsMargins(2, 2, 2, 2)
        self._strip_lay.setSpacing(0)
        self._strip_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._scroll.setWidget(self._strip)

        outer.addWidget(self._scroll)

    # ── public ────────────────────────────────────────────────────────────

    def refresh(self, pipeline_manager) -> None:
        """Rebuild the entire panel from the PipelineManager's current state."""
        self._rebuild_combo(pipeline_manager)
        self._rebuild_strip(pipeline_manager)

    # ── private ───────────────────────────────────────────────────────────

    def _rebuild_combo(self, pm) -> None:
        self.branch_combo.blockSignals(True)
        self.branch_combo.clear()
        for b in pm.branches:
            if b.parent_branch_id is not None:
                parent_name = next(
                    (pb.name for pb in pm.branches
                     if pb.branch_id == b.parent_branch_id), "?"
                )
                label = f"{b.name} ← {parent_name}@{b.fork_step}"
            else:
                label = b.name
            self.branch_combo.addItem(label)
        self._current_branch_idx = pm.current_branch_idx
        self.branch_combo.setCurrentIndex(pm.current_branch_idx)
        self.branch_combo.blockSignals(False)

    def _rebuild_strip(self, pm) -> None:
        # Clear existing nodes
        while self._strip_lay.count():
            item = self._strip_lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        branch = pm.current_branch
        if branch is None:
            return

        bidx = pm.current_branch_idx
        current_step = branch.current_idx

        for i, entry in enumerate(branch.entries):
            node = _StepNode(i, entry.label, is_current=(i == current_step))
            node.clicked.connect(
                lambda _checked, b=bidx, s=i: self.step_clicked.emit(b, s)
            )
            self._strip_lay.addWidget(node)

            if i < len(branch.entries) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #666; padding: 0 3px; font-size: 11px;")
                arrow.setFixedHeight(32)
                self._strip_lay.addWidget(arrow)

        # Scroll so the active node is visible (~116 px per node+arrow pair)
        approx_x = max(0, current_step * 116 - 60)
        self._scroll.horizontalScrollBar().setValue(approx_x)

    def _on_combo_changed(self, idx: int) -> None:
        if idx >= 0 and idx != self._current_branch_idx:
            self._current_branch_idx = idx
            self.branch_changed_sig.emit(idx)
