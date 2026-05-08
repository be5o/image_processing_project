import numpy as np
from typing import Dict, Any


class ImageState:
    def __init__(self):
        self.original: np.ndarray = None        # immutable reference; never overwritten after load
        self.current: np.ndarray = None         # working copy modified by pipeline
        self.metadata: Dict[str, Any] = {}
        self.file_path: str = ""
        self.dicom_dataset = None               # pydicom.Dataset when source was .dcm; None otherwise
