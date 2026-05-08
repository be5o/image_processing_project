import numpy as np


def compute_fft(img: np.ndarray):
    """Compute the 2-D DFT of the image and return both the shifted spectrum
    (complex, for filter arithmetic) and a displayable uint8 log-magnitude.

    Returns
    -------
    fshift  : complex128 ndarray  – zero-frequency-centred FFT
    spectrum: uint8 ndarray       – log(1 + |fshift|) scaled to [0, 255]
    """
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)
    log_mag = np.log1p(np.abs(fshift))
    peak = log_mag.max()
    if peak > 0:
        spectrum = (log_mag / peak * 255.0).astype(np.uint8)
    else:
        spectrum = np.zeros(img.shape, dtype=np.uint8)
    return fshift, spectrum


def build_notch_filter(shape: tuple, notch_points: list,
                       radius: float, filter_type: str,
                       order: int = 2) -> np.ndarray:
    """Build a notch-reject filter mask H in the shifted frequency domain.

    Each click coordinate (u, v) and its conjugate-symmetric counterpart
    (rows-u, cols-v) are both suppressed so that IFFT(H * F) is real-valued.

    Parameters
    ----------
    shape        : (rows, cols)
    notch_points : list of (row, col) in the displayed spectrum
    radius       : notch radius in pixels
    filter_type  : 'Ideal' | 'Butterworth' | 'Gaussian'
    order        : Butterworth order (ignored for other types)

    Returns
    -------
    H : float64 ndarray in [0, 1]  (1 = pass, 0 = reject)
    """
    rows, cols = shape
    H = np.ones((rows, cols), dtype=np.float64)

    # Vectorised coordinate grids
    y_grid = np.arange(rows, dtype=np.float64).reshape(-1, 1)
    x_grid = np.arange(cols, dtype=np.float64).reshape(1, -1)

    for (u, v) in notch_points:
        # Conjugate-symmetric partner (modular wrap handles centre point)
        su = int((rows - u) % rows)
        sv = int((cols - v) % cols)

        for (nu, nv) in [(int(u), int(v)), (su, sv)]:
            d = np.sqrt((y_grid - nu) ** 2 + (x_grid - nv) ** 2)

            if filter_type == 'Ideal':
                H[d <= radius] = 0.0

            elif filter_type == 'Butterworth':
                # 1 / (1 + (D0/D)^(2n))  — notch reject form
                H *= 1.0 / (1.0 + (radius / (d + 1e-10)) ** (2 * order))

            else:  # Gaussian
                # 1 - exp(-D^2 / 2*D0^2)
                H *= 1.0 - np.exp(-d ** 2 / (2.0 * radius ** 2))

    return H


def apply_notch_filter(img: np.ndarray, notch_points: list,
                       radius: float, filter_type: str,
                       order: int = 2) -> np.ndarray:
    """Compute FFT, apply notch-reject mask, inverse-FFT → cleaned uint8 image.

    Called by _start_operation so signature is (img, *extra_args).
    """
    if not notch_points:
        return img.copy()

    fshift, _ = compute_fft(img)
    H = build_notch_filter(img.shape, notch_points, radius, filter_type, order)
    filtered = fshift * H
    f_ishift = np.fft.ifftshift(filtered)
    result = np.real(np.fft.ifft2(f_ishift))
    return np.clip(result, 0, 255).astype(np.uint8)
