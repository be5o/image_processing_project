"""Clinical Image Analysis Workbench — Main Window.

Phase 1  : I/O, zoom, spatial filters, local histogram equalisation, pipeline.
Phase 2  : FFT / notch filtering, noise injection, ROI statistics.
Bonus    : Morphological engine.
Pipeline : Branch-aware undo/redo, pipeline visualisation panel, log export,
           native DICOM save.
"""

from __future__ import annotations

import os
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QSlider,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, QRect
from PyQt6.QtGui import QAction

from core.image_state import ImageState
from core.pipline_manager import PipelineManager
from ui.image_viewer import ImageViewer
from ui.fft_viewer import FftViewer
from ui.pipeline_panel import PipelinePanel
from ui.dialogs import ErrorDialog, RoiStatsDialog
from utils.worker import Worker
from utils.validators import validate_kernel_size, validate_sigma
from processing import io_handler, interpolation, spatial_filters, histogram_eq
from processing import frequency_filters, noise_injection, roi_stats, morphology


class MainWindow(QMainWindow):
    """Clinical Image Analysis Workbench — Phase 1 + Phase 2 + Bonus."""

    _instances: list = []

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Image Analysis Workbench")
        self.resize(1280, 860)

        self.state    = ImageState()
        self.pipeline = PipelineManager(self.state)
        self.pipeline.image_updated.connect(self.update_viewer)
        self.pipeline.error_occurred.connect(self.show_error)
        self.pipeline.pipeline_changed.connect(self._refresh_pipeline_panel)

        self._thread = None
        self._worker = None

        # Pending label/params set before every _start_operation call so that
        # _on_op_done can forward them to pipeline.apply_result.
        self._pending_label:  str  = "Operation"
        self._pending_params: dict = {}

        # Phase 2 state
        self._roi_image_coords: tuple | None = None

        MainWindow._instances.append(self)
        self.init_ui()
        self._build_menu()

    # ═══════════════════════════════════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════════════════════════════════

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ── Left control panel (scrollable) ──────────────────────────────
        inner_panel = QWidget()
        inner_layout = QVBoxLayout(inner_panel)
        inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        inner_layout.setSpacing(6)

        inner_layout.addWidget(self._make_io_group())
        inner_layout.addWidget(self._make_zoom_group())
        inner_layout.addWidget(self._make_filter_group())
        inner_layout.addWidget(self._make_he_group())
        inner_layout.addWidget(self._make_pipeline_group())
        inner_layout.addWidget(self._make_freq_group())
        inner_layout.addWidget(self._make_noise_group())
        inner_layout.addWidget(self._make_roi_group())
        inner_layout.addWidget(self._make_morph_group())

        scroll_ctrl = QScrollArea()
        scroll_ctrl.setWidgetResizable(True)
        scroll_ctrl.setWidget(inner_panel)
        scroll_ctrl.setFixedWidth(310)
        scroll_ctrl.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll_ctrl)

        # ── Right: viewer tabs + pipeline panel ──────────────────────────
        self.viewer_tabs = QTabWidget()

        viewer_tab = QWidget()
        v_layout = QVBoxLayout(viewer_tab)
        self.viewer = ImageViewer()
        v_layout.addWidget(self.viewer)
        self.metadata_label = QLabel("Metadata will appear here.")
        self.metadata_label.setWordWrap(True)
        v_layout.addWidget(self.metadata_label)
        self.viewer_tabs.addTab(viewer_tab, "Image Viewer")

        # Edge output tabs (lazy)
        self._edge_gx_viewer  = ImageViewer()
        self._edge_gy_viewer  = ImageViewer()
        self._edge_mag_viewer = ImageViewer()
        self._tab_gx  = QWidget(); QVBoxLayout(self._tab_gx).addWidget(self._edge_gx_viewer)
        self._tab_gy  = QWidget(); QVBoxLayout(self._tab_gy).addWidget(self._edge_gy_viewer)
        self._tab_mag = QWidget(); QVBoxLayout(self._tab_mag).addWidget(self._edge_mag_viewer)
        self._edge_tabs_visible = False

        # FFT Spectrum tab (lazy)
        self.fft_viewer = FftViewer()
        self._fft_tab = QWidget()
        fft_layout = QVBoxLayout(self._fft_tab)
        fft_layout.addWidget(QLabel(
            "Click on bright spikes to place a notch filter. "
            "Each click also marks its conjugate point automatically."
        ))
        fft_layout.addWidget(self.fft_viewer)
        self._fft_tab_visible = False

        outer_tabs = QTabWidget()
        outer_tabs.addTab(self.viewer_tabs, "Processing")

        # Pipeline panel fixed below viewer area
        self.pipeline_panel = PipelinePanel()
        self.pipeline_panel.step_clicked.connect(
            lambda bi, si: self.pipeline.jump_to(bi, si))
        self.pipeline_panel.branch_changed_sig.connect(
            self.pipeline.switch_branch)
        self.pipeline_panel.nav_prev.connect(self.pipeline.undo)
        self.pipeline_panel.nav_next.connect(self.pipeline.redo)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(outer_tabs)
        right_layout.addWidget(self.pipeline_panel)
        main_layout.addWidget(right_widget)

        # Collect all operation buttons for _set_busy
        self._all_op_buttons = [
            self.apply_filter_btn, self.apply_he_btn,
            self.btn_undo, self.btn_redo, self.btn_reset,
            self.btn_load, self.btn_save,
            self.btn_compute_fft, self.btn_clear_notches, self.btn_apply_notch,
            self.btn_inject_noise,
            self.btn_compute_stats, self.btn_clear_roi,
            self.btn_apply_thresh,
            self.btn_erode, self.btn_dilate,
            self.btn_open, self.btn_close, self.btn_boundary,
        ]

        self.connect_signals()

    # ── Group builders ────────────────────────────────────────────────────

    def _make_io_group(self) -> QGroupBox:
        g = QGroupBox("Image I/O")
        lay = QVBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_save = QPushButton("Save Current")
        lay.addWidget(self.btn_load)
        lay.addWidget(self.btn_save)
        g.setLayout(lay)
        return g

    def _make_zoom_group(self) -> QGroupBox:
        g = QGroupBox("Custom Zoom (From Scratch)")
        lay = QVBoxLayout()
        self.zoom_method = QComboBox()
        self.zoom_method.addItems(["Nearest Neighbor", "Bilinear"])
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)
        self.zoom_slider.setValue(100)
        self.zoom_label = QLabel("Scale: 100%")
        lay.addWidget(QLabel("Interpolation Method:"))
        lay.addWidget(self.zoom_method)
        lay.addWidget(self.zoom_label)
        lay.addWidget(self.zoom_slider)
        g.setLayout(lay)
        return g

    def _make_filter_group(self) -> QGroupBox:
        g = QGroupBox("Spatial Filters")
        lay = QFormLayout()
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Average", "Gaussian", "Sobel", "Prewitt", "Median"])
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(3, 11)
        self.kernel_size.setSingleStep(2)
        self.kernel_size.setValue(3)
        self.gauss_sigma = QDoubleSpinBox()
        self.gauss_sigma.setRange(0.1, 10.0)
        self.gauss_sigma.setValue(1.0)
        self.apply_filter_btn = QPushButton("Apply Filter")
        lay.addRow("Filter Type:", self.filter_type)
        lay.addRow("Kernel Size (Odd):", self.kernel_size)
        self._sigma_label = QLabel("Gaussian σ:")
        lay.addRow(self._sigma_label, self.gauss_sigma)
        lay.addRow(self.apply_filter_btn)
        g.setLayout(lay)
        self._update_sigma_visibility(self.filter_type.currentText())
        return g

    def _make_he_group(self) -> QGroupBox:
        g = QGroupBox("Local Histogram Eq")
        lay = QVBoxLayout()
        self.block_size = QSpinBox()
        self.block_size.setRange(4, 64)
        self.block_size.setValue(8)
        self.apply_he_btn = QPushButton("Apply Local HE")
        lay.addWidget(QLabel("Block Size:"))
        lay.addWidget(self.block_size)
        lay.addWidget(self.apply_he_btn)
        g.setLayout(lay)
        return g

    def _make_pipeline_group(self) -> QGroupBox:
        g = QGroupBox("Pipeline Controls")
        lay = QHBoxLayout()
        self.btn_undo  = QPushButton("Undo")
        self.btn_redo  = QPushButton("Redo")
        self.btn_reset = QPushButton("Reset")
        lay.addWidget(self.btn_undo)
        lay.addWidget(self.btn_redo)
        lay.addWidget(self.btn_reset)
        g.setLayout(lay)
        return g

    def _make_freq_group(self) -> QGroupBox:
        g = QGroupBox("Frequency Domain")
        lay = QFormLayout()
        lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self.btn_compute_fft = QPushButton("Compute FFT Spectrum")
        lay.addRow(self.btn_compute_fft)

        hint = QLabel("→ Click on bright spikes in the FFT tab to add notches")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 10px;")
        lay.addRow(hint)

        self.notch_filter_type = QComboBox()
        self.notch_filter_type.addItems(["Ideal", "Butterworth", "Gaussian"])
        lay.addRow("Filter Type:", self.notch_filter_type)

        self.notch_radius = QSpinBox()
        self.notch_radius.setRange(1, 200)
        self.notch_radius.setValue(10)
        lay.addRow("Notch Radius:", self.notch_radius)

        self._order_label = QLabel("BW Order:")
        self.notch_order = QSpinBox()
        self.notch_order.setRange(1, 10)
        self.notch_order.setValue(2)
        lay.addRow(self._order_label, self.notch_order)

        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_clear_notches = QPushButton("Clear Notches")
        self.btn_apply_notch   = QPushButton("Apply Notch")
        btn_lay.addWidget(self.btn_clear_notches)
        btn_lay.addWidget(self.btn_apply_notch)
        lay.addRow(btn_row)

        g.setLayout(lay)
        self._update_notch_order_visibility(self.notch_filter_type.currentText())
        return g

    def _make_noise_group(self) -> QGroupBox:
        g = QGroupBox("Noise Injection")
        lay = QFormLayout()
        lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self.noise_type = QComboBox()
        self.noise_type.addItems(["Gaussian", "Uniform"])
        lay.addRow("Noise Type:", self.noise_type)

        self._noise_mean_lbl = QLabel("Mean:")
        self.noise_mean = QDoubleSpinBox()
        self.noise_mean.setRange(-100.0, 100.0)
        self.noise_mean.setValue(0.0)
        lay.addRow(self._noise_mean_lbl, self.noise_mean)

        self._noise_std_lbl = QLabel("Std Dev:")
        self.noise_std = QDoubleSpinBox()
        self.noise_std.setRange(0.1, 150.0)
        self.noise_std.setValue(25.0)
        lay.addRow(self._noise_std_lbl, self.noise_std)

        self._noise_low_lbl = QLabel("Low:")
        self.noise_low = QDoubleSpinBox()
        self.noise_low.setRange(-255.0, 0.0)
        self.noise_low.setValue(-30.0)
        lay.addRow(self._noise_low_lbl, self.noise_low)

        self._noise_high_lbl = QLabel("High:")
        self.noise_high = QDoubleSpinBox()
        self.noise_high.setRange(0.0, 255.0)
        self.noise_high.setValue(30.0)
        lay.addRow(self._noise_high_lbl, self.noise_high)

        self.btn_inject_noise = QPushButton("Inject Noise")
        lay.addRow(self.btn_inject_noise)

        g.setLayout(lay)
        self._update_noise_params_visibility(self.noise_type.currentText())
        return g

    def _make_roi_group(self) -> QGroupBox:
        g = QGroupBox("ROI Statistics")
        lay = QVBoxLayout()

        self.btn_draw_roi = QPushButton("Draw ROI")
        self.btn_draw_roi.setCheckable(True)
        self.btn_draw_roi.setToolTip("Click and drag on the image to select a region")
        lay.addWidget(self.btn_draw_roi)

        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_clear_roi     = QPushButton("Clear ROI")
        self.btn_compute_stats = QPushButton("Show Statistics")
        btn_lay.addWidget(self.btn_clear_roi)
        btn_lay.addWidget(self.btn_compute_stats)
        lay.addWidget(btn_row)

        g.setLayout(lay)
        return g

    def _make_morph_group(self) -> QGroupBox:
        g = QGroupBox("Morphological Engine (Bonus)")
        lay = QFormLayout()
        lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        thresh_lbl = QLabel("Threshold:")
        thresh_lbl.setStyleSheet("font-weight: bold;")
        lay.addRow(thresh_lbl)

        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(128)
        self.thresh_val_label = QLabel("128")
        lay.addRow("Value:", self.thresh_val_label)
        lay.addRow(self.thresh_slider)
        self.btn_apply_thresh = QPushButton("Apply Threshold")
        lay.addRow(self.btn_apply_thresh)

        se_lbl = QLabel("Structuring Element:")
        se_lbl.setStyleSheet("font-weight: bold;")
        lay.addRow(se_lbl)

        self.se_shape = QComboBox()
        self.se_shape.addItems(["Square", "Cross"])
        lay.addRow("SE Shape:", self.se_shape)

        self.se_size = QSpinBox()
        self.se_size.setRange(3, 11)
        self.se_size.setSingleStep(2)
        self.se_size.setValue(3)
        lay.addRow("SE Size:", self.se_size)

        row1 = QWidget(); r1lay = QHBoxLayout(row1); r1lay.setContentsMargins(0, 0, 0, 0)
        self.btn_erode  = QPushButton("Erode")
        self.btn_dilate = QPushButton("Dilate")
        r1lay.addWidget(self.btn_erode); r1lay.addWidget(self.btn_dilate)
        lay.addRow(row1)

        row2 = QWidget(); r2lay = QHBoxLayout(row2); r2lay.setContentsMargins(0, 0, 0, 0)
        self.btn_open  = QPushButton("Open")
        self.btn_close = QPushButton("Close")
        r2lay.addWidget(self.btn_open); r2lay.addWidget(self.btn_close)
        lay.addRow(row2)

        self.btn_boundary = QPushButton("Boundary Extract")
        lay.addRow(self.btn_boundary)

        g.setLayout(lay)
        return g

    # ═══════════════════════════════════════════════════════════════════════
    # Menu
    # ═══════════════════════════════════════════════════════════════════════

    def _build_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        new_win = QAction("New Window", self)
        new_win.setShortcut("Ctrl+N")
        new_win.triggered.connect(MainWindow.open_new_window)
        file_menu.addAction(new_win)
        file_menu.addSeparator()

        close_act = QAction("Close Window", self)
        close_act.setShortcut("Ctrl+W")
        close_act.triggered.connect(self.close)
        file_menu.addAction(close_act)

    @staticmethod
    def open_new_window():
        w = MainWindow()
        w.show()

    def closeEvent(self, event):
        try:
            MainWindow._instances.remove(self)
        except ValueError:
            pass
        super().closeEvent(event)

    # ═══════════════════════════════════════════════════════════════════════
    # Signal connections
    # ═══════════════════════════════════════════════════════════════════════

    def connect_signals(self):
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_image)
        self.zoom_slider.valueChanged.connect(self.apply_zoom)
        self.zoom_method.currentIndexChanged.connect(self.apply_zoom)
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.apply_he_btn.clicked.connect(self.apply_local_he)
        self.btn_undo.clicked.connect(self.pipeline.undo)
        self.btn_redo.clicked.connect(self.pipeline.redo)
        self.btn_reset.clicked.connect(self.pipeline.reset)
        self.filter_type.currentIndexChanged.connect(
            lambda: self._update_sigma_visibility(self.filter_type.currentText()))

        self.btn_compute_fft.clicked.connect(self.compute_fft_spectrum)
        self.btn_clear_notches.clicked.connect(self.fft_viewer.clear_notches)
        self.btn_apply_notch.clicked.connect(self.apply_notch_filter)
        self.notch_filter_type.currentIndexChanged.connect(
            lambda: self._update_notch_order_visibility(
                self.notch_filter_type.currentText()))

        self.btn_inject_noise.clicked.connect(self.inject_noise)
        self.noise_type.currentIndexChanged.connect(
            lambda: self._update_noise_params_visibility(
                self.noise_type.currentText()))

        self.btn_draw_roi.toggled.connect(self.toggle_roi_mode)
        self.btn_compute_stats.clicked.connect(self.compute_roi_stats)
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.viewer.roi_selected.connect(self._on_roi_selected)

        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_val_label.setText(str(v)))
        self.btn_apply_thresh.clicked.connect(self.apply_threshold)
        self.btn_erode.clicked.connect(lambda: self.apply_morph("erode"))
        self.btn_dilate.clicked.connect(lambda: self.apply_morph("dilate"))
        self.btn_open.clicked.connect(lambda: self.apply_morph("open"))
        self.btn_close.clicked.connect(lambda: self.apply_morph("close"))
        self.btn_boundary.clicked.connect(lambda: self.apply_morph("boundary"))

    # ═══════════════════════════════════════════════════════════════════════
    # Visibility helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _update_sigma_visibility(self, filter_name: str):
        is_gauss = (filter_name == "Gaussian")
        self._sigma_label.setVisible(is_gauss)
        self.gauss_sigma.setVisible(is_gauss)

    def _update_notch_order_visibility(self, filter_name: str):
        is_bw = (filter_name == "Butterworth")
        self._order_label.setVisible(is_bw)
        self.notch_order.setVisible(is_bw)

    def _update_noise_params_visibility(self, noise_name: str):
        is_gauss = (noise_name == "Gaussian")
        self._noise_mean_lbl.setVisible(is_gauss)
        self.noise_mean.setVisible(is_gauss)
        self._noise_std_lbl.setVisible(is_gauss)
        self.noise_std.setVisible(is_gauss)
        self._noise_low_lbl.setVisible(not is_gauss)
        self.noise_low.setVisible(not is_gauss)
        self._noise_high_lbl.setVisible(not is_gauss)
        self.noise_high.setVisible(not is_gauss)

    # ═══════════════════════════════════════════════════════════════════════
    # Threading helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _set_busy(self, busy: bool):
        for btn in self._all_op_buttons:
            btn.setEnabled(not busy)
        if busy:
            self.statusBar().showMessage("Processing…")
        else:
            self.statusBar().clearMessage()

    def _start_operation(self, func, *args):
        """Run ``func(state.current, *args)`` on a background thread."""
        if self.state.current is None:
            return
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = Worker(func, self.state.current.copy(), *args)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_op_done)
        self._worker.error.connect(self._on_op_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _start_custom_task(self, func, args: tuple, on_done):
        """Run ``func(*args)`` on a background thread with a custom callback."""
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = Worker(func, *args)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_done)
        self._worker.error.connect(self._on_op_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    # ── Operation completion callbacks ────────────────────────────────────

    def _on_op_done(self, result):
        if isinstance(result, dict):
            self._on_edge_done(result)
            return
        self.pipeline.apply_result(result, self._pending_label,
                                   self._pending_params)
        self._set_busy(False)

    def _on_edge_done(self, result: dict):
        self._edge_gx_viewer.display_image(result["gx"])
        self._edge_gy_viewer.display_image(result["gy"])
        self._edge_mag_viewer.display_image(result["magnitude"])

        if not self._edge_tabs_visible:
            self.viewer_tabs.addTab(self._tab_gx,  "Gx (Horizontal)")
            self.viewer_tabs.addTab(self._tab_gy,  "Gy (Vertical)")
            self.viewer_tabs.addTab(self._tab_mag, "Magnitude")
            self._edge_tabs_visible = True

        self.viewer_tabs.setCurrentWidget(self._tab_mag)
        self.pipeline.apply_result(result["magnitude"],
                                   self._pending_label, self._pending_params)
        self._set_busy(False)

    def _on_fft_done(self, result):
        fshift, spectrum = result
        self._last_fshift = fshift
        self.fft_viewer.display_spectrum(spectrum)

        if not self._fft_tab_visible:
            self.viewer_tabs.addTab(self._fft_tab, "FFT Spectrum")
            self._fft_tab_visible = True

        self.viewer_tabs.setCurrentWidget(self._fft_tab)
        self._set_busy(False)

    def _on_op_error(self, msg):
        ErrorDialog.show(self, "Processing Error", msg, critical=True)
        self._set_busy(False)

    # ═══════════════════════════════════════════════════════════════════════
    # Viewer update + zoom
    # ═══════════════════════════════════════════════════════════════════════

    def update_viewer(self, img):
        scale = self.zoom_slider.value() / 100.0
        if scale != 1.0:
            func = (interpolation.nearest_neighbor
                    if self.zoom_method.currentText() == "Nearest Neighbor"
                    else interpolation.bilinear_interpolation)
            try:
                img = func(img, scale)
            except Exception:
                pass
        self.viewer.display_image(img)
        self._restore_roi_overlay()

    def apply_zoom(self):
        if self.state.current is None:
            return
        val = self.zoom_slider.value()
        scale = val / 100.0
        self.zoom_label.setText(f"Scale: {val}%")
        func = (interpolation.nearest_neighbor
                if self.zoom_method.currentText() == "Nearest Neighbor"
                else interpolation.bilinear_interpolation)
        try:
            zoomed = func(self.state.current, scale)
            self.viewer.display_image(zoomed)
            self._restore_roi_overlay()
        except Exception as e:
            self.show_error(str(e))

    def _fit_to_viewer(self):
        if self.state.current is None:
            return
        h, w = self.state.current.shape[:2]
        vw = max(1, self.viewer.width()  - 4)
        vh = max(1, self.viewer.height() - 4)
        scale = min(vw / w, vh / h, 1.0)
        val   = max(self.zoom_slider.minimum(), int(scale * 100))
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(val)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"Scale: {val}%")
        self.apply_zoom()

    def show_error(self, msg):
        QMessageBox.critical(self, "Processing Error", msg)

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline panel
    # ═══════════════════════════════════════════════════════════════════════

    def _refresh_pipeline_panel(self):
        if self.pipeline.branches:
            self.pipeline_panel.refresh(self.pipeline)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1 handlers
    # ═══════════════════════════════════════════════════════════════════════

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.dcm *.jpg *.jpeg *.bmp *.png)"
        )
        if not path:
            return
        try:
            img, meta, dicom_ds = io_handler.load_image(path)
            self.state.dicom_dataset = dicom_ds
            self.pipeline.load_initial(img, meta, path)
            self.metadata_label.setText(
                "\n".join(f"{k}: {v}" for k, v in meta.items()))
            self._roi_image_coords = None
            self.viewer.clear_roi()
            if self.btn_draw_roi.isChecked():
                self.btn_draw_roi.setChecked(False)
            self._fit_to_viewer()
        except Exception as e:
            self.show_error(f"Failed to load image: {e}")

    def save_image(self):
        if self.state.current is None:
            return

        # Offer DICOM format when the source was a .dcm file
        if self.state.dicom_dataset is not None:
            file_filter = "DICOM (*.dcm);;Images (*.jpg *.bmp *.png)"
        else:
            file_filter = "Images (*.jpg *.bmp *.png)"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", file_filter)
        if not path:
            return

        try:
            if path.lower().endswith(".dcm") and self.state.dicom_dataset is not None:
                io_handler.save_dicom(path, self.state.current,
                                      self.state.dicom_dataset)
            else:
                io_handler.save_image(path, self.state.current)

            self._export_operation_log(path)
            QMessageBox.information(self, "Success",
                                    "Image saved.\nOperation log exported alongside.")
        except Exception as e:
            self.show_error(f"Failed to save: {e}")

    def _export_operation_log(self, image_path: str):
        """Write a .txt operation log next to the saved image."""
        branch = self.pipeline.current_branch
        if branch is None:
            return

        base = os.path.splitext(image_path)[0]
        log_path = base + "_log.txt"

        lines = [
            "Clinical Image Analysis — Operation Log",
            f"Exported : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Branch   : {branch.name}",
        ]
        if branch.parent_branch_id is not None:
            parent_name = next(
                (b.name for b in self.pipeline.branches
                 if b.branch_id == branch.parent_branch_id), "?"
            )
            lines.append(
                f"Forked from : {parent_name} at step {branch.fork_step}")
        lines += ["", f"{'Step':<6} {'Operation':<26} Parameters",
                  "-" * 65]

        for i, entry in enumerate(branch.entries):
            params_str = (
                ", ".join(f"{k}={v}" for k, v in entry.params.items())
                if entry.params else "—"
            )
            lines.append(f"{i:<6} {entry.label:<26} {params_str}")

        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def apply_filter(self):
        if self.state.current is None:
            ErrorDialog.show(self, "No Image Loaded",
                             "Please load an image first.", critical=False)
            return
        try:
            ftype = self.filter_type.currentText()
            ksize = validate_kernel_size(self.kernel_size.value(), 3, 15)
            sigma = validate_sigma(self.gauss_sigma.value(), 0.1, 10.0)

            if ftype == "Average":
                self._pending_label  = f"Average k={ksize}"
                self._pending_params = {"kernel": ksize}
                self._start_operation(spatial_filters.apply_conv_wrapper,
                                      spatial_filters.average_filter(ksize))

            elif ftype == "Gaussian":
                self._pending_label  = f"Gaussian σ={sigma} k={ksize}"
                self._pending_params = {"sigma": sigma, "kernel": ksize}
                self._start_operation(spatial_filters.apply_conv_wrapper,
                                      spatial_filters.gaussian_filter(ksize, sigma))

            elif ftype == "Sobel":
                self._pending_label  = "Sobel (Magnitude)"
                self._pending_params = {}
                self._start_operation(spatial_filters.apply_sobel)

            elif ftype == "Prewitt":
                self._pending_label  = "Prewitt (Magnitude)"
                self._pending_params = {}
                self._start_operation(spatial_filters.apply_prewitt)

            elif ftype == "Median":
                self._pending_label  = f"Median k={ksize}"
                self._pending_params = {"kernel": ksize}
                self._start_operation(spatial_filters.apply_median, ksize)

        except ValueError as ve:
            ErrorDialog.show(self, "Invalid Parameters", str(ve), critical=False)

    def apply_local_he(self):
        if self.state.current is None:
            return
        bs = self.block_size.value()
        self._pending_label  = f"Local HE b={bs}"
        self._pending_params = {"block_size": bs}
        self._start_operation(histogram_eq.local_histogram_equalization, bs)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2 handlers — Frequency domain
    # ═══════════════════════════════════════════════════════════════════════

    def compute_fft_spectrum(self):
        if self.state.current is None:
            ErrorDialog.show(self, "No Image", "Load an image first.",
                             critical=False)
            return
        self._start_custom_task(
            frequency_filters.compute_fft,
            (self.state.current.copy(),),
            self._on_fft_done,
        )

    def apply_notch_filter(self):
        if self.state.current is None:
            return
        pts = self.fft_viewer.get_notch_points()
        if not pts:
            ErrorDialog.show(self, "No Notches",
                             "Click on the FFT spectrum tab to place notch points first.",
                             critical=False)
            return
        radius      = float(self.notch_radius.value())
        filter_type = self.notch_filter_type.currentText()
        order       = self.notch_order.value()

        self._pending_label  = f"Notch {filter_type} r={int(radius)}"
        self._pending_params = {"type": filter_type, "radius": radius,
                                "order": order, "points": len(pts)}
        self._start_operation(frequency_filters.apply_notch_filter,
                              pts, radius, filter_type, order)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2 handlers — Noise injection
    # ═══════════════════════════════════════════════════════════════════════

    def inject_noise(self):
        if self.state.current is None:
            return
        ntype = self.noise_type.currentText()
        if ntype == "Gaussian":
            mean = self.noise_mean.value()
            std  = self.noise_std.value()
            self._pending_label  = f"Noise Gauss μ={mean} σ={std}"
            self._pending_params = {"mean": mean, "std": std}
            self._start_operation(noise_injection.add_gaussian_noise, mean, std)
        else:
            low  = self.noise_low.value()
            high = self.noise_high.value()
            self._pending_label  = f"Noise Uniform [{low},{high}]"
            self._pending_params = {"low": low, "high": high}
            self._start_operation(noise_injection.add_uniform_noise, low, high)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2 handlers — ROI statistics
    # ═══════════════════════════════════════════════════════════════════════

    def toggle_roi_mode(self, active: bool):
        self.viewer.set_roi_mode(active)
        self.btn_draw_roi.setText(
            "Drawing ROI…  (click to stop)" if active else "Draw ROI")

    def _on_roi_selected(self, rect):
        scale = self.zoom_slider.value() / 100.0
        x1 = int(rect.x()                  / scale)
        y1 = int(rect.y()                  / scale)
        x2 = int((rect.x() + rect.width()) / scale)
        y2 = int((rect.y() + rect.height()) / scale)
        self._roi_image_coords = (x1, y1, x2, y2)
        if self.btn_draw_roi.isChecked():
            self.btn_draw_roi.setChecked(False)

    def _restore_roi_overlay(self):
        if self._roi_image_coords is None:
            return
        scale = self.zoom_slider.value() / 100.0
        x1, y1, x2, y2 = self._roi_image_coords
        rect = QRect(
            int(x1 * scale), int(y1 * scale),
            int((x2 - x1) * scale), int((y2 - y1) * scale),
        )
        self.viewer.set_roi_overlay(rect)

    def clear_roi(self):
        self._roi_image_coords = None
        self.viewer.clear_roi()

    def compute_roi_stats(self):
        if self.state.current is None:
            return
        if self._roi_image_coords is None:
            ErrorDialog.show(self, "No ROI Selected",
                             "Use the 'Draw ROI' button to select a region first.",
                             critical=False)
            return
        x1, y1, x2, y2 = self._roi_image_coords
        stats = roi_stats.compute_roi_stats(self.state.current, x1, y1, x2, y2)
        if stats is None:
            ErrorDialog.show(self, "Invalid ROI",
                             "The selected region is too small or out of bounds.",
                             critical=False)
            return
        dlg = RoiStatsDialog(stats, self)
        dlg.exec()

    # ═══════════════════════════════════════════════════════════════════════
    # Bonus handlers — Morphological engine
    # ═══════════════════════════════════════════════════════════════════════

    def apply_threshold(self):
        if self.state.current is None:
            return
        t = self.thresh_slider.value()
        self._pending_label  = f"Threshold t={t}"
        self._pending_params = {"threshold": t}
        self._start_operation(morphology.apply_threshold, t)

    def apply_morph(self, op: str):
        if self.state.current is None:
            return
        shape = self.se_shape.currentText()
        size  = self.se_size.value()
        se    = morphology.create_structuring_element(size, shape)

        op_map = {
            "erode":    morphology.erode,
            "dilate":   morphology.dilate,
            "open":     morphology.opening,
            "close":    morphology.closing,
            "boundary": morphology.boundary_extraction,
        }
        label_map = {
            "erode": "Erode", "dilate": "Dilate",
            "open":  "Open",  "close":  "Close",
            "boundary": "Boundary",
        }
        self._pending_label  = f"{label_map[op]} {shape} {size}×{size}"
        self._pending_params = {"op": op, "se_shape": shape, "se_size": size}
        self._start_operation(op_map[op], se)
