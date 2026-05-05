import os
import numpy as np
import pydicom
from PIL import Image
from typing import Dict, Any, Tuple

def load_image(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    ext = os.path.splitext(filepath)[1].lower()
    metadata = {"Path": filepath, "Width": 0, "Height": 0, "BitDepth": 8}

    if ext in ['.dcm', '.dicom']:
        ds = pydicom.dcmread(filepath)
        pixel = ds.pixel_array.astype(np.float32)
        # Handle multi-frame DICOM: take the middle frame so a single 2D array is shown
        if pixel.ndim == 3:
            mid = pixel.shape[0] // 2
            pixel = pixel[mid]
        elif pixel.ndim > 3:
            # Flatten extra leading dimensions down to 2D
            pixel = pixel.reshape(-1, pixel.shape[-2], pixel.shape[-1])[pixel.shape[0] // 2]
        # Normalise to 0-255
        if pixel.max() > pixel.min():
            pixel = (pixel - pixel.min()) / (pixel.max() - pixel.min()) * 255.0
        img = pixel.astype(np.uint8)
        # ds.Rows = number of rows = height, ds.Columns = number of columns = width
        metadata.update({
            "Width": ds.Columns, "Height": ds.Rows,
            "BitDepth": getattr(ds, 'BitsAllocated', 8),
            "PatientName": str(getattr(ds, 'PatientName', 'N/A')),
            "Modality": str(getattr(ds, 'Modality', 'N/A')),
            "BodyPart": str(getattr(ds, 'BodyPartExamined', 'N/A')),
            "Age": str(getattr(ds, 'PatientAge', 'N/A'))
        })
    elif ext in ['.jpg', '.jpeg', '.bmp', '.png']:
        with Image.open(filepath) as pil_img:
            img = np.array(pil_img.convert('L'))
            metadata.update({"Width": pil_img.width, "Height": pil_img.height, "BitDepth": 8})
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return img, metadata

def save_image(filepath: str, img: np.ndarray):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.jpg', '.bmp', '.png']:
        Image.fromarray(img).save(filepath)
    else:
        raise ValueError("Save only supports JPEG, BMP, PNG for Phase 1")