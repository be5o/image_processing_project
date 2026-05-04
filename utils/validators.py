"""
Input validation utilities for clinical image processing.
All functions raise ValueError with user-friendly messages.
"""
import numpy as np
from typing import Union

def validate_kernel_size(size: int, min_val: int = 3, max_val: int = 15) -> int:
    """Ensure kernel size is odd and within bounds."""
    if size < min_val or size > max_val:
        raise ValueError(f"Kernel size must be between {min_val} and {max_val}")
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd for symmetric padding")
    return size

def validate_block_size(size: int, min_val: int = 4, max_val: int = 64) -> int:
    """Validate block size for local histogram equalization."""
    if size < min_val or size > max_val:
        raise ValueError(f"Block size must be between {min_val} and {max_val}")
    return size

def validate_sigma(sigma: float, min_val: float = 0.1, max_val: float = 10.0) -> float:
    """Validate Gaussian sigma parameter."""
    if sigma < min_val or sigma > max_val:
        raise ValueError(f"Sigma must be between {min_val} and {max_val}")
    return sigma

def validate_image_array(img: np.ndarray, allow_color: bool = False) -> np.ndarray:
    """Ensure image is 2D grayscale uint8 or convertible."""
    if img is None or img.size == 0:
        raise ValueError("Image array is empty or None")
    
    # Convert to grayscale if 3-channel
    if img.ndim == 3 and allow_color:
        # Simple luminance conversion (from scratch)
        img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    elif img.ndim == 3 and not allow_color:
        raise ValueError("Color images not supported for this operation")
    elif img.ndim != 2:
        raise ValueError(f"Expected 2D array, got {img.ndim}D")
    
    # Ensure uint8 in [0, 255]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def validate_scale_factor(scale: float, min_scale: float = 0.25, max_scale: float = 4.0) -> float:
    """Validate zoom scale factor."""
    if scale < min_scale or scale > max_scale:
        raise ValueError(f"Scale must be between {min_scale} and {max_scale}")
    return scale