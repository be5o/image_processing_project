import numpy as np

def rotate_90(img: np.ndarray, clockwise: bool = True) -> np.ndarray:
    return np.rot90(img, k=-1 if clockwise else 1)

def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return np.fliplr(img)

def flip_vertical(img: np.ndarray) -> np.ndarray:
    return np.flipud(img)
