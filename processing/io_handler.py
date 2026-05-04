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
        img = ds.pixel_array.astype(np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        metadata.update({
            "Width": ds.Rows, "Height": ds.Columns,
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