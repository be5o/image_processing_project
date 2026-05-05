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
            # Use padded center pixel (equivalent to img[i,j] but clearer for the local block)
            center_val = int(padded[i + pad, j + pad])
            # Add epsilon to denominator to avoid division-by-zero when cdf_min ~ block.size
            out[i, j] = ((cdf[center_val] - cdf_min) / (block.size - cdf_min + 1e-5)) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)