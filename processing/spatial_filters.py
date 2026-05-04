import numpy as np
from .convolution import convolve2d

def average_filter(size: int): return np.ones((size, size), dtype=np.float32) / (size * size)

def gaussian_filter(size: int, sigma: float):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_conv_wrapper(img, kernel):
    return np.clip(convolve2d(img, kernel), 0, 255).astype(np.uint8)

def apply_sobel(img):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    mag = np.sqrt(convolve2d(img, gx)**2 + convolve2d(img, gy)**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def apply_prewitt(img):
    gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    mag = np.sqrt(convolve2d(img, gx)**2 + convolve2d(img, gy)**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def apply_median(img: np.ndarray, size: int) -> np.ndarray:
    if size % 2 == 0: raise ValueError("Kernel size must be odd")
    pad = size // 2
    padded = np.pad(img, pad, mode='reflect')
    h, w = img.shape
    out = np.zeros((h, w), dtype=img.dtype)
    half = size * size // 2
    for i in range(h):
        for j in range(w):
            window = padded[i:i+size, j:j+size].flatten()
            out[i, j] = np.sort(window)[half]
    return out