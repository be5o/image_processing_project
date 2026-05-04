import numpy as np
from PyQt6.QtGui import QImage

def array_to_qimage(img: np.ndarray) -> QImage:
    # Ensure C-contiguous for safe memory access
    img = np.clip(img, 0, 255).astype(np.uint8)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        return QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)