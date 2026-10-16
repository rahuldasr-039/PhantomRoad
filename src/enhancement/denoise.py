"""
Edge-preserving bilateral denoising for road scenes.
Removes high-frequency camera sensor noise while maintaining sharp lane boundaries.
"""

import cv2
import numpy as np


def denoise_edge_preserving(
    image_bgr: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 40.0,
    sigma_space: float = 40.0
) -> np.ndarray:
    """
    Applies bilateral filter for edge-preserving denoising.

    Args:
        image_bgr: Input BGR image (uint8 numpy array).
        diameter: Pixel neighborhood diameter.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.

    Returns:
        Denoised BGR image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    return cv2.bilateralFilter(
        image_bgr,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
