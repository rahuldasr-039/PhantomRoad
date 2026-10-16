"""
Contrast enhancement using Contrast Limited Adaptive Histogram Equalization (CLAHE).
Operates in LAB color space on the Luminance channel to preserve color fidelity.
"""

import cv2
import numpy as np


def enhance_contrast_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: int = 8
) -> np.ndarray:
    """
    Enhances local contrast while avoiding over-amplification of noise.

    Args:
        image_bgr: Input BGR image (uint8 numpy array).
        clip_limit: Threshold for contrast limiting (typical range: 1.5 - 4.0).
        tile_grid_size: Grid size for local histogram equalization.

    Returns:
        Contrast-enhanced BGR image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    if len(image_bgr.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(image_bgr)

    # Convert to CIELAB color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
