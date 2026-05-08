import numpy as np


def compute_roi_stats(img: np.ndarray,
                      x1: int, y1: int,
                      x2: int, y2: int) -> dict | None:
    """Compute local statistics for a rectangular ROI.

    All metrics are computed from scratch (no scipy/sklearn stats functions).

    Histogram  : frequency count per intensity bin 0-255 using np.add.at
                 (not np.histogram).
    Mean       : sum of pixel values / count.
    Variance   : E[(X - µ)²] = sum of squared deviations / count.

    Parameters
    ----------
    img        : 2-D uint8 grayscale image
    x1, y1     : top-left corner in image coordinates
    x2, y2     : bottom-right corner in image coordinates (exclusive)

    Returns
    -------
    dict with keys: histogram (256,), mean, variance, std, pixel_count,
                    roi_shape  — or None if the ROI is degenerate.
    """
    h, w = img.shape[:2]
    x1 = int(max(0, min(x1, w)))
    x2 = int(max(0, min(x2, w)))
    y1 = int(max(0, min(y1, h)))
    y2 = int(max(0, min(y2, h)))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = img[y1:y2, x1:x2]
    flat = roi.flatten().astype(np.float64)
    n = float(flat.size)

    # --- histogram from scratch (no np.histogram) ---
    hist = np.zeros(256, dtype=np.int64)
    np.add.at(hist, roi.flatten().astype(np.intp), 1)

    # --- mean from scratch ---
    mean_val = float(np.sum(flat)) / n

    # --- variance from scratch: E[(X - µ)^2] ---
    variance_val = float(np.sum((flat - mean_val) ** 2)) / n

    return {
        'histogram':   hist,
        'mean':        mean_val,
        'variance':    variance_val,
        'std':         variance_val ** 0.5,
        'pixel_count': int(n),
        'roi_shape':   roi.shape,
    }
