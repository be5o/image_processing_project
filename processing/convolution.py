import numpy as np

def pad_image(img: np.ndarray, pad_size: int, mode: str = 'zero') -> np.ndarray:
    h, w = img.shape
    padded = np.zeros((h + 2*pad_size, w + 2*pad_size))


    for i in range(h):
        for j in range(w):
            padded[i + pad_size][j + pad_size] = img[i][j]

    if mode == 'replicate':
        for i in range(h):
            for p in range(pad_size):
                padded[i + pad_size][p] = img[i][0]
                padded[i + pad_size][w + pad_size + p] = img[i][w-1]

        for j in range(w):
            for p in range(pad_size):
                padded[p][j + pad_size] = img[0][j]
                padded[h + pad_size + p][j + pad_size] = img[h-1][j]

    elif mode == 'reflect':
        for i in range(pad_size):
            for j in range(w):
                padded[i][j + pad_size] = img[pad_size - i][j]
                padded[h + pad_size + i][j + pad_size] = img[h - 2 - i][j]

        for i in range(h + 2*pad_size):
            for j in range(pad_size):
                padded[i][j] = padded[i][2*pad_size - j]
                padded[i][w + pad_size + j] = padded[i][w + pad_size - 2 - j]

    return padded


def convolve2d(img: np.ndarray, kernel: np.ndarray, padding: str = 'zero') -> np.ndarray:
    kh, kw = kernel.shape
    pad = kh // 2

    padded = pad_image(img, pad, padding)
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = 0.0
            for m in range(kh):
                for n in range(kw):
                    val += padded[i+m][j+n] * kernel[m][n]
            out[i][j] = val

    return out