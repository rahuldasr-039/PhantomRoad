"""
Non-linear illumination and brightness correction using Gamma LUT transformation.
"""

import cv2
import numpy as np


def correct_illumination_gamma(
    image_bgr: np.ndarray,
    gamma: float = 1.8
) -> np.ndarray:
    """
    Applies gamma correction to adjust dynamic range.
    gamma > 1.0 brightens underexposed/night scenes.
    gamma < 1.0 tones down overexposed scenes.

    Args:
        image_bgr: Input BGR image (uint8 numpy array).
        gamma: Power factor.

    Returns:
        Gamma-corrected BGR image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # Clamp gamma to reasonable range
    gamma = float(np.clip(gamma, 0.2, 5.0))
    inv_gamma = 1.0 / gamma

    # Precompute lookup table for high-throughput frame processing
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255.0
        for i in range(256)
    ], dtype=np.uint8)

    return cv2.LUT(image_bgr, table)
