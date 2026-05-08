"""Image I/O: load (DICOM + PIL) and save (PIL + native DICOM)."""

import copy
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image


def load_image(
    filepath: str,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[pydicom.Dataset]]:
    """Load an image file and return ``(uint8_array, metadata_dict, dicom_dataset)``.

    *dicom_dataset* is the original ``pydicom.Dataset`` when the source is a
    ``.dcm``/``.dicom`` file, otherwise ``None``.  Callers that only need the
    array + metadata can ignore the third value.
    """
    ext = os.path.splitext(filepath)[1].lower()
    metadata: Dict[str, Any] = {"Path": filepath, "Width": 0,
                                 "Height": 0, "BitDepth": 8}
    dicom_dataset: Optional[pydicom.Dataset] = None

    if ext in (".dcm", ".dicom"):
        ds = pydicom.dcmread(filepath)
        dicom_dataset = ds
        pixel = ds.pixel_array.astype(np.float32)

        if pixel.ndim == 3:
            mid = pixel.shape[0] // 2
            pixel = pixel[mid]
        elif pixel.ndim > 3:
            pixel = pixel.reshape(-1, pixel.shape[-2], pixel.shape[-1])[pixel.shape[0] // 2]

        if pixel.max() > pixel.min():
            pixel = (pixel - pixel.min()) / (pixel.max() - pixel.min()) * 255.0
        img = pixel.astype(np.uint8)

        metadata.update({
            "Width":       ds.Columns,
            "Height":      ds.Rows,
            "BitDepth":    getattr(ds, "BitsAllocated", 8),
            "PatientName": str(getattr(ds, "PatientName",       "N/A")),
            "Modality":    str(getattr(ds, "Modality",          "N/A")),
            "BodyPart":    str(getattr(ds, "BodyPartExamined",  "N/A")),
            "Age":         str(getattr(ds, "PatientAge",        "N/A")),
        })

    elif ext in (".jpg", ".jpeg", ".bmp", ".png"):
        with Image.open(filepath) as pil_img:
            img = np.array(pil_img.convert("L"))
            metadata.update({"Width": pil_img.width,
                              "Height": pil_img.height, "BitDepth": 8})
    else:
        raise ValueError(f"Unsupported format: {ext}")

    return img, metadata, dicom_dataset


def save_image(filepath: str, img: np.ndarray) -> None:
    """Save an ndarray as JPEG, BMP, or PNG via PIL."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".jpg", ".jpeg", ".bmp", ".png"):
        Image.fromarray(img).save(filepath)
    else:
        raise ValueError(f"save_image only supports JPEG/BMP/PNG; got '{ext}'")


def save_dicom(filepath: str, img: np.ndarray,
               original_dataset: pydicom.Dataset) -> None:
    """Save *img* as a DICOM file, preserving all metadata from *original_dataset*.

    The pixel array is stored as 8-bit monochrome.  All patient/study/series
    tags from the source dataset are retained unchanged.
    """
    ds = copy.deepcopy(original_dataset)

    arr = img if img.dtype == np.uint8 else img.astype(np.uint8)

    ds.PixelData               = arr.tobytes()
    ds.Rows                    = arr.shape[0]
    ds.Columns                 = arr.shape[1]
    ds.BitsAllocated           = 8
    ds.BitsStored              = 8
    ds.HighBit                 = 7
    ds.PixelRepresentation     = 0
    ds.SamplesPerPixel         = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # If the source was multi-frame we now have a single frame
    if hasattr(ds, "NumberOfFrames"):
        ds.NumberOfFrames = 1

    ds.save_as(filepath, write_like_original=False)
