"""Binary morphological operations implemented entirely from scratch.

No morphology libraries (cv2.erode, skimage.morphology, bwmorph, etc.) are used.
The sliding-window approach loops over SE elements (at most size² iterations)
and performs one vectorised numpy AND/OR per active SE element, avoiding any
per-pixel Python loops.
"""

import numpy as np


# ── Structuring Element ───────────────────────────────────────────────────────

def create_structuring_element(size: int, shape: str = 'square') -> np.ndarray:
    """Return a boolean structuring element.

    Parameters
    ----------
    size  : odd integer (3, 5, 7, …)
    shape : 'square' | 'cross'
    """
    if shape.lower() == 'cross':
        se = np.zeros((size, size), dtype=bool)
        mid = size // 2
        se[mid, :] = True   # horizontal bar
        se[:, mid] = True   # vertical bar
        return se
    return np.ones((size, size), dtype=bool)    # square (all ones)


# ── Thresholding ──────────────────────────────────────────────────────────────

def apply_threshold(img: np.ndarray, threshold: int) -> np.ndarray:
    """Global threshold: pixels >= threshold → 255, else → 0."""
    return (img.astype(np.int32) >= int(threshold)).astype(np.uint8) * 255


# ── Internal helper ───────────────────────────────────────────────────────────

def _to_bool(img: np.ndarray) -> np.ndarray:
    """Convert any uint8 image to a boolean foreground mask (> 127)."""
    return img > 127


# ── Core Operations ───────────────────────────────────────────────────────────

def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Binary erosion: an output pixel is foreground iff *every* SE-covered
    input pixel is foreground.

    Algorithm
    ---------
    Pad the binary image with False (border pixels always erode to 0).
    For each active SE element at offset (dr, dc), extract the shifted window
    and AND it into the running result.  Loop count = number of True cells in SE.
    """
    binary = _to_bool(img)
    h, w = binary.shape
    se_h, se_w = se.shape
    ph, pw = se_h // 2, se_w // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), constant_values=False)
    result = np.ones((h, w), dtype=bool)
    for r in range(se_h):
        for c in range(se_w):
            if se[r, c]:
                result &= padded[r:r + h, c:c + w]
    return result.astype(np.uint8) * 255


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Binary dilation: an output pixel is foreground iff *any* SE-covered
    input pixel is foreground.

    Algorithm
    ---------
    Pad with False (no phantom foreground pixels from outside the image).
    For each active SE element, extract shifted window and OR into result.
    """
    binary = _to_bool(img)
    h, w = binary.shape
    se_h, se_w = se.shape
    ph, pw = se_h // 2, se_w // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), constant_values=False)
    result = np.zeros((h, w), dtype=bool)
    for r in range(se_h):
        for c in range(se_w):
            if se[r, c]:
                result |= padded[r:r + h, c:c + w]
    return result.astype(np.uint8) * 255


# ── Compound Operations ───────────────────────────────────────────────────────

def opening(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological opening: erosion → dilation.  Removes small noise objects."""
    return dilate(erode(img, se), se)


def closing(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological closing: dilation → erosion.  Fills small holes/gaps."""
    return erode(dilate(img, se), se)


def boundary_extraction(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological boundary: original_binary − eroded_binary.

    Outlines foreground objects without their interiors.
    """
    eroded = erode(img, se)
    diff = img.astype(np.int32) - eroded.astype(np.int32)
    return np.clip(diff, 0, 255).astype(np.uint8)
