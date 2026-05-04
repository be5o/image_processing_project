from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QScrollArea
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from utils.image_convert import array_to_qimage

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)
        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll)

    def display_image(self, img_array):
        q_img = array_to_qimage(img_array)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()