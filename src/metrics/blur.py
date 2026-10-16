"""
Blur estimation metrics using Laplacian variance and frequency-domain analysis.
"""

from typing import Dict, Any, Union
import cv2
import numpy as np


def compute_blur_metrics(
    image: np.ndarray,
    blur_threshold: float = 85.0
) -> Dict[str, Any]:
    """
    Computes blur metrics using spatial Laplacian variance and 2D-FFT frequency ratio.

    Args:
        image: Input image (BGR, BGRA, or Grayscale numpy array).
        blur_threshold: Laplacian variance threshold below which frame is flagged as blurred.

    Returns:
        dict containing:
            - laplacian_variance (float): Pech-Pacheco variance of Laplacian.
            - fft_high_freq_ratio (float): Ratio of high-frequency energy in 2D spectrum.
            - is_blurred (bool): True if laplacian_variance < blur_threshold.
            - sharpness_score (float): Normalized score [0.0 - 1.0].
            - threshold (float): Applied blur threshold.
    """
    if image is None:
        return {
            "laplacian_variance": 0.0,
            "fft_high_freq_ratio": 0.0,
            "is_blurred": True,
            "sharpness_score": 0.0,
            "threshold": blur_threshold
        }

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be a numpy.ndarray, got {type(image).__name__}")

    if not isinstance(blur_threshold, (int, float)) or blur_threshold <= 0:
        raise ValueError("blur_threshold must be a positive number.")

    if image.size == 0:
        return {
            "laplacian_variance": 0.0,
            "fft_high_freq_ratio": 0.0,
            "is_blurred": True,
            "sharpness_score": 0.0,
            "threshold": blur_threshold
        }

    if image.ndim not in (2, 3):
        raise ValueError(f"Input image must have 2 or 3 dimensions, got shape {image.shape}")

    # Handle data types (e.g. float images normalized to [0, 1] or [0, 255])
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0 + 1e-4:
            uint8_img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_img = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        uint8_img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        uint8_img = image

    # Convert to grayscale
    if uint8_img.ndim == 3:
        channels = uint8_img.shape[2]
        if channels == 3:
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            gray = uint8_img[:, :, 0]
        else:
            raise ValueError(f"Expected 1, 3, or 4 color channels, got {channels}")
    else:
        gray = uint8_img.copy()

    # 1. Spatial Domain: Pech-Pacheco Laplacian Variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_var = float(laplacian.var())

    # 2. Frequency Domain: 2D FFT High-Frequency Energy Ratio
    # Downsample large frames for fast FFT computation
    h, w = gray.shape
    scale = min(1.0, 256.0 / max(h, w))
    if scale < 1.0:
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray

    sh, sw = small_gray.shape
    dft = np.fft.fft2(small_gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)

    # Low frequency mask: circle with radius r at center
    cy, cx = sh // 2, sw // 2
    r = max(1, min(cy, cx) // 3)
    y_indices, x_indices = np.ogrid[:sh, :sw]
    dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
    low_freq_mask = dist_from_center <= r

    total_energy = float(np.sum(magnitude_spectrum)) + 1e-7
    low_freq_energy = float(np.sum(magnitude_spectrum[low_freq_mask]))
    high_freq_energy = max(0.0, total_energy - low_freq_energy)
    fft_high_freq_ratio = float(high_freq_energy / total_energy)

    # 3. Normalized Sharpness Score [0.0 - 1.0] using smooth logarithmic mapping
    sharpness_score = float(np.clip(np.log1p(lap_var) / np.log1p(500.0), 0.0, 1.0))

    return {
        "laplacian_variance": round(lap_var, 2),
        "fft_high_freq_ratio": round(fft_high_freq_ratio, 4),
        "is_blurred": bool(lap_var < blur_threshold),
        "sharpness_score": round(sharpness_score, 4),
        "threshold": float(blur_threshold)
    }


def compute_laplacian_map(image: np.ndarray) -> np.ndarray:
    """
    Generates a normalized uint8 visualization map of Laplacian edge gradients.

    Args:
        image: Input image (BGR or Grayscale).

    Returns:
        uint8 single-channel visualization image showing high-frequency edge magnitudes.
    """
    if image is None or image.size == 0:
        return np.zeros((100, 100), dtype=np.uint8)

    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    abs_lap = np.absolute(laplacian)
    norm_lap = cv2.normalize(abs_lap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm_lap
