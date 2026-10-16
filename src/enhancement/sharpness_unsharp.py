"""
Sharpness enhancement using adaptive unsharp masking.
"""

import cv2
import numpy as np


def enhance_sharpness_unsharp_mask(
    image_bgr: np.ndarray,
    amount: float = 1.2,
    sigma: float = 1.0,
    threshold: int = 0
) -> np.ndarray:
    """
    Applies unsharp masking to enhance soft edges.
    amount: Sharpening multiplier (0.5 - 2.5).
    sigma: Standard deviation of Gaussian blur.
    threshold: Coring threshold (0 for standard full-frame unsharp mask).

    Returns:
        Sharpened BGR image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if threshold <= 0:
        return cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)

    # Coring threshold to avoid noise amplification in flat regions
    diff = cv2.absdiff(image_bgr, blurred)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    mask_norm = mask.astype(np.float32) / 255.0
    mask_3ch = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)

    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0).astype(np.float32)
    orig_f = image_bgr.astype(np.float32)
    output = sharpened * mask_3ch + orig_f * (1.0 - mask_3ch)
    return np.clip(output, 0, 255).astype(np.uint8)
