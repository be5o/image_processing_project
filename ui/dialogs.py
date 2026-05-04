"""
Reusable dialog windows for metadata display and error reporting.
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QMessageBox, QHBoxLayout)
from PyQt6.QtCore import Qt
from typing import Dict, Any

class MetadataDialog(QDialog):
    """Modal dialog to display image metadata in a readable table."""
    def __init__(self, metadata: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Metadata")
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Extracted Metadata")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Metadata table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Property", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        
        # Populate rows
        table.setRowCount(len(metadata))
        for row, (key, value) in enumerate(metadata.items()):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
        
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(table)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class ErrorDialog:
    """Static helper for consistent error messaging."""
    @staticmethod
    def show(parent, title: str, message: str, critical: bool = False):
        """Show a QMessageBox with standardized styling."""
        box = QMessageBox(parent)
        box.setWindowTitle(title)
        box.setText(message)
        box.setIcon(QMessageBox.Icon.Critical if critical else QMessageBox.Icon.Warning)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

class ProcessingProgressDialog(QDialog):
    """Optional: Show progress for long operations (Phase 2+)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing...")
        self.setModal(True)
        self.setFixedSize(300, 100)
        
        layout = QVBoxLayout(self)
        self.label = QLabel("Applying filter...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
        # No cancel button for Phase 1 (keep it simple)