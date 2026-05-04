import numpy as np
from typing import Dict, Any

class ImageState:
    def __init__(self):
        self.original: np.ndarray = None
        self.current: np.ndarray = None
        self.metadata: Dict[str, Any] = {}
        self.file_path: str = ""