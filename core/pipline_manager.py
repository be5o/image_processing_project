from PyQt6.QtCore import QObject, pyqtSignal
from .image_state import ImageState
import numpy as np

class PipelineManager(QObject):
    image_updated = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, state: ImageState):
        super().__init__()
        self.state = state
        self.history: list[np.ndarray] = []
        self.current_idx: int = -1

    def load_initial(self, img: np.ndarray, metadata: dict, path: str):
        self.state.original = img.copy()
        self.state.current = img.copy()
        self.state.metadata = metadata
        self.state.file_path = path
        self.history = [img.copy()]
        self.current_idx = 0
        self.image_updated.emit(img)

    def apply_operation(self, func, *args, **kwargs):
        try:
            result = func(self.state.current, *args, **kwargs)
            if result is None: return
            self.history = self.history[:self.current_idx + 1]
            self.history.append(result.copy())
            self.current_idx += 1
            self.state.current = result.copy()
            self.image_updated.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def apply_result(self, result: np.ndarray):
        if result is None:
            return
        self.history = self.history[:self.current_idx + 1]
        self.history.append(result.copy())
        self.current_idx += 1
        self.state.current = result.copy()
        self.image_updated.emit(result)

    def undo(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.state.current = self.history[self.current_idx].copy()
            self.image_updated.emit(self.state.current)

    def reset(self):
        if self.state.original is not None:
            self.current_idx = 0
            self.history = [self.state.original.copy()]
            self.state.current = self.state.original.copy()
            self.image_updated.emit(self.state.current)