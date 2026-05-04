import numpy as np

def pad_image(img: np.ndarray, pad_size: int, mode: str = 'zero') -> np.ndarray:
    if mode == 'zero': return np.pad(img, pad_size, mode='constant', constant_values=0)
    elif mode == 'replicate': return np.pad(img, pad_size, mode='edge')
    elif mode == 'reflect': return np.pad(img, pad_size, mode='reflect')
    raise ValueError("Invalid padding mode")

def convolve2d(img: np.ndarray, kernel: np.ndarray, padding: str = 'zero') -> np.ndarray:
    kh, kw = kernel.shape
    pad = kh // 2
    padded = pad_image(img, pad, padding)
    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    k_flat = kernel.flatten()
    # Explicit sliding window (Defense-friendly & strictly from scratch)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw].flatten() * k_flat)
    return out