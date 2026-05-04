import numpy as np

def local_histogram_equalization(img: np.ndarray, block_size: int) -> np.ndarray:
    h, w = img.shape
    out = img.copy().astype(np.float32)
    pad = block_size // 2
    padded = np.pad(img, pad, mode='reflect')
    for i in range(h):
        for j in range(w):
            block = padded[i:i+block_size, j:j+block_size].flatten()
            hist = np.bincount(block, minlength=256)
            cdf = np.cumsum(hist)
            nonzero = cdf[np.nonzero(cdf)]
            if len(nonzero) == 0: continue
            cdf_min = nonzero[0]
            out[i, j] = ((cdf[img[i, j]] - cdf_min) / (block.size - cdf_min)) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)