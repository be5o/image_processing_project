import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QFileDialog, QMessageBox, QGroupBox, QFormLayout, QSlider,
                             QMenuBar)
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QAction
from core.image_state import ImageState
from core.pipline_manager import PipelineManager
from ui.image_viewer import ImageViewer
from ui.dialogs import ErrorDialog
from utils.worker import Worker
from utils.validators import validate_kernel_size, validate_sigma
from processing import io_handler, interpolation, spatial_filters, histogram_eq

class MainWindow(QMainWindow):
    # Class-level list keeps every open window alive (prevents GC)
    _instances: list = []
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Image Analysis Workbench")
        self.resize(1200, 800)

        self.state = ImageState()
        self.pipeline = PipelineManager(self.state)
        self.pipeline.image_updated.connect(self.update_viewer)
        self.pipeline.error_occurred.connect(self.show_error)
        self._thread = None
        self._worker = None
        # Register this instance so Python keeps it alive
        MainWindow._instances.append(self)
        self.init_ui()
        self._build_menu()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Control Panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)

        # I/O
        io_group = QGroupBox("Image I/O")
        io_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_save = QPushButton("Save Current")
        io_layout.addWidget(self.btn_load)
        io_layout.addWidget(self.btn_save)
        io_group.setLayout(io_layout)
        control_layout.addWidget(io_group)

        # Zoom
        zoom_group = QGroupBox("Custom Zoom (From Scratch)")
        zoom_layout = QVBoxLayout()
        self.zoom_method = QComboBox()
        self.zoom_method.addItems(["Nearest Neighbor", "Bilinear"])
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)
        self.zoom_slider.setValue(100)
        zoom_layout.addWidget(QLabel("Interpolation Method:"))
        zoom_layout.addWidget(self.zoom_method)
        self.zoom_label = QLabel("Scale: 100%")
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_group.setLayout(zoom_layout)
        control_layout.addWidget(zoom_group)

        # Filters
        filter_group = QGroupBox("Spatial Filters")
        filter_layout = QFormLayout()
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
        filter_layout.addRow("Filter Type:", self.filter_type)
        filter_layout.addRow("Kernel Size (Odd):", self.kernel_size)
        # Keep references to the sigma label/widget so we can show/hide them
        self._sigma_label = QLabel("Gaussian \u03c3:")
        filter_layout.addRow(self._sigma_label, self.gauss_sigma)
        filter_layout.addRow(self.apply_filter_btn)
        filter_group.setLayout(filter_layout)
        control_layout.addWidget(filter_group)
        # Show sigma only for Gaussian; hide for all other filters
        self._update_sigma_visibility(self.filter_type.currentText())

        # Local HE
        he_group = QGroupBox("Local Histogram Eq")
        he_layout = QVBoxLayout()
        self.block_size = QSpinBox()
        self.block_size.setRange(4, 64)
        self.block_size.setValue(8)
        self.apply_he_btn = QPushButton("Apply Local HE")
        he_layout.addWidget(QLabel("Block Size:"))
        he_layout.addWidget(self.block_size)
        he_layout.addWidget(self.apply_he_btn)
        he_group.setLayout(he_layout)
        control_layout.addWidget(he_group)

        # Pipeline
        pipeline_group = QGroupBox("Pipeline Controls")
        p_layout = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_reset = QPushButton("Reset to Original")
        p_layout.addWidget(self.btn_undo)
        p_layout.addWidget(self.btn_reset)
        pipeline_group.setLayout(p_layout)
        control_layout.addWidget(pipeline_group)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        # Viewer & Tabs
        self.viewer_tabs = QTabWidget()

        # Main image viewer tab
        viewer_tab = QWidget()
        v_layout = QVBoxLayout(viewer_tab)
        self.viewer = ImageViewer()
        v_layout.addWidget(self.viewer)
        self.metadata_label = QLabel("Metadata will appear here.")
        self.metadata_label.setWordWrap(True)
        v_layout.addWidget(self.metadata_label)
        self.viewer_tabs.addTab(viewer_tab, "Image Viewer")

        # Edge output tabs (Gx, Gy, Magnitude) — created once, shown after edge filter
        self._edge_gx_viewer   = ImageViewer()
        self._edge_gy_viewer   = ImageViewer()
        self._edge_mag_viewer  = ImageViewer()
        self._tab_gx  = QWidget()
        self._tab_gy  = QWidget()
        self._tab_mag = QWidget()
        QVBoxLayout(self._tab_gx).addWidget(self._edge_gx_viewer)
        QVBoxLayout(self._tab_gy).addWidget(self._edge_gy_viewer)
        QVBoxLayout(self._tab_mag).addWidget(self._edge_mag_viewer)
        # Edge tabs are hidden until an edge filter is applied
        self._edge_tabs_visible = False

        tabs = QTabWidget()
        tabs.addTab(self.viewer_tabs, "Processing")
        main_layout.addWidget(tabs)

        self.connect_signals()

    def _build_menu(self):
        """Add a File menu with a New Window action."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        new_win_action = QAction("New Window", self)
        new_win_action.setShortcut("Ctrl+N")
        new_win_action.setStatusTip("Open a new independent image processing window")
        new_win_action.triggered.connect(MainWindow.open_new_window)
        file_menu.addAction(new_win_action)

        file_menu.addSeparator()

        close_action = QAction("Close Window", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

    @staticmethod
    def open_new_window():
        """Spawn a completely independent MainWindow instance."""
        w = MainWindow()
        w.show()
        # _instances already updated inside __init__

    def closeEvent(self, event):
        """Remove this window from the global list when it is closed."""
        try:
            MainWindow._instances.remove(self)
        except ValueError:
            pass
        super().closeEvent(event)

    def connect_signals(self):
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_image)
        self.zoom_slider.valueChanged.connect(self.apply_zoom)
        # Task 4: changing interpolation method immediately redraws at current scale
        self.zoom_method.currentIndexChanged.connect(self.apply_zoom)
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.apply_he_btn.clicked.connect(self.apply_local_he)
        self.btn_undo.clicked.connect(self.pipeline.undo)
        self.btn_reset.clicked.connect(self.pipeline.reset)
        # Task 3: show/hide Gaussian sigma field based on selected filter
        self.filter_type.currentIndexChanged.connect(
            lambda: self._update_sigma_visibility(self.filter_type.currentText())
        )

    def _update_sigma_visibility(self, filter_name: str):
        """Show the Gaussian sigma control only when Gaussian filter is selected."""
        is_gaussian = (filter_name == "Gaussian")
        self._sigma_label.setVisible(is_gaussian)
        self.gauss_sigma.setVisible(is_gaussian)

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

    def show_error(self, msg):
        QMessageBox.critical(self, "Processing Error", msg)

    def _set_busy(self, busy: bool):
        for btn in (self.apply_filter_btn, self.apply_he_btn,
                    self.btn_undo, self.btn_reset, self.btn_load, self.btn_save):
            btn.setEnabled(not busy)
        if busy:
            self.statusBar().showMessage("Processing…")
        else:
            self.statusBar().clearMessage()

    def _start_operation(self, func, *args):
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

    def _on_op_done(self, result):
        # Task 2: Sobel/Prewitt return a dict {gx, gy, magnitude}
        if isinstance(result, dict):
            self._on_edge_done(result)
            return
        self.pipeline.apply_result(result)
        self._set_busy(False)

    def _on_edge_done(self, result: dict):
        """Handle edge-filter dict result: show Gx/Gy/Magnitude tabs and store magnitude."""
        gx  = result["gx"]
        gy  = result["gy"]
        mag = result["magnitude"]

        # Display each output in its dedicated viewer
        self._edge_gx_viewer.display_image(gx)
        self._edge_gy_viewer.display_image(gy)
        self._edge_mag_viewer.display_image(mag)

        # Add edge tabs the first time an edge filter runs
        if not self._edge_tabs_visible:
            self.viewer_tabs.addTab(self._tab_gx,  "Gx (Horizontal)")
            self.viewer_tabs.addTab(self._tab_gy,  "Gy (Vertical)")
            self.viewer_tabs.addTab(self._tab_mag, "Magnitude")
            self._edge_tabs_visible = True

        # Switch to the Magnitude tab automatically
        self.viewer_tabs.setCurrentWidget(self._tab_mag)

        # Store magnitude in pipeline so further operations keep working
        self.pipeline.apply_result(mag)
        self._set_busy(False)

    def _on_op_error(self, msg):
        ErrorDialog.show(self, "Processing Error", msg, critical=True)
        self._set_busy(False)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.dcm *.jpg *.jpeg *.bmp *.png)")
        if path:
            try:
                img, meta = io_handler.load_image(path)
                self.pipeline.load_initial(img, meta, path)
                self.metadata_label.setText("\n".join(f"{k}: {v}" for k, v in meta.items()))
                self._fit_to_viewer()
            except Exception as e:
                self.show_error(f"Failed to load image: {e}")

    def save_image(self):
        if self.state.current is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.jpg *.bmp *.png)")
        if path:
            try:
                io_handler.save_image(path, self.state.current)
                QMessageBox.information(self, "Success", "Image saved.")
            except Exception as e:
                self.show_error(f"Failed to save: {e}")

    def _fit_to_viewer(self):
        if self.state.current is None:
            return
        h, w = self.state.current.shape[:2]
        vw = max(1, self.viewer.width() - 4)
        vh = max(1, self.viewer.height() - 4)
        scale = min(vw / w, vh / h, 1.0)
        val = max(self.zoom_slider.minimum(), int(scale * 100))
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(val)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"Scale: {val}%")
        self.apply_zoom()

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
        except Exception as e:
            self.show_error(str(e))

    def apply_filter(self):
        if self.state.current is None:
            ErrorDialog.show(self, "No Image Loaded",
                             "Please load an image first.", critical=False)
            return
        try:
            filter_type = self.filter_type.currentText()
            kernel_size = validate_kernel_size(self.kernel_size.value(), min_val=3, max_val=15)
            sigma = validate_sigma(self.gauss_sigma.value(), min_val=0.1, max_val=10.0)

            if filter_type == "Average":
                kernel = spatial_filters.average_filter(kernel_size)
                self._start_operation(spatial_filters.apply_conv_wrapper, kernel)
            elif filter_type == "Gaussian":
                kernel = spatial_filters.gaussian_filter(kernel_size, sigma)
                self._start_operation(spatial_filters.apply_conv_wrapper, kernel)
            elif filter_type == "Sobel":
                self._start_operation(spatial_filters.apply_sobel)
            elif filter_type == "Prewitt":
                self._start_operation(spatial_filters.apply_prewitt)
            elif filter_type == "Median":
                self._start_operation(spatial_filters.apply_median, kernel_size)
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

        except ValueError as ve:
            ErrorDialog.show(self, "Invalid Parameters", str(ve), critical=False)

    def apply_local_he(self):
        if self.state.current is None:
            return
        self._start_operation(histogram_eq.local_histogram_equalization,
                               self.block_size.value())