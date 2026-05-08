import numpy as np


def add_gaussian_noise(img: np.ndarray,
                       mean: float = 0.0,
                       std: float = 25.0) -> np.ndarray:
    """Inject Gaussian noise using the Box-Muller transform (from scratch).

    Two independent uniform samples u1, u2 in (0,1) are converted to a
    standard-normal variate z via:
        z = sqrt(-2 * ln(u1)) * cos(2π * u2)
    The noise is then scaled and shifted: noise = mean + std * z.

    Called by _start_operation so signature is (img, mean, std).
    """
    shape = img.shape
    u1 = np.random.uniform(1e-10, 1.0, shape)   # avoid log(0)
    u2 = np.random.uniform(0.0,   1.0, shape)
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    noisy = img.astype(np.float64) + mean + std * z
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_uniform_noise(img: np.ndarray,
                      low: float = -30.0,
                      high: float = 30.0) -> np.ndarray:
    """Inject Uniform noise drawn from [low, high) (from scratch).

    Maps a uniform [0,1) sample to [low, high) via:
        noise = low + (high - low) * U(0,1)

    Called by _start_operation so signature is (img, low, high).
    """
    noise = low + (high - low) * np.random.random(img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
