import numpy as np

def nearest_neighbor(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    y_coords = np.floor(np.linspace(0, h - 1, new_h)).astype(int)
    x_coords = np.floor(np.linspace(0, w - 1, new_w)).astype(int)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    return img[Y, X] if img.ndim == 2 else img[Y, X, :]

def bilinear_interpolation(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    y_out = np.linspace(0, h - 1, new_h)
    x_out = np.linspace(0, w - 1, new_w)
    Y_out, X_out = np.meshgrid(y_out, x_out, indexing='ij')
    y0, x0 = np.floor(Y_out).astype(int), np.floor(X_out).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = Y_out - y0, X_out - x0
    out = (img[y0, x0]*(1-wy)*(1-wx) + img[y0, x1]*(1-wy)*wx +
           img[y1, x0]*wy*(1-wx) + img[y1, x1]*wy*wx)
    return out.astype(img.dtype)