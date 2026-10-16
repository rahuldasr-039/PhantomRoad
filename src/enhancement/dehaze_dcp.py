"""
Atmospheric scattering compensation and dehazing using Dark Channel Prior (DCP)
and guided transmission recovery (He et al. 2011).
"""

import cv2
import numpy as np

from ..metrics.visibility import (
    compute_dark_channel,
    estimate_atmospheric_light,
    estimate_transmission_map
)


def guided_filter_refinement(
    guide_gray: np.ndarray,
    raw_transmission: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3
) -> np.ndarray:
    """
    Fast edge-preserving guided filter to refine transmission map and remove block halos.
    """
    guide = guide_gray.astype(np.float32) / 255.0
    p = raw_transmission.astype(np.float32)

    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * p, cv2.CV_32F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

    refined_t = mean_a * guide + mean_b
    return np.clip(refined_t, 0.10, 1.0)


def dehaze_dark_channel(
    image_bgr: np.ndarray,
    omega: float = 0.85,
    t_min: float = 0.15
) -> np.ndarray:
    """
    Recovers haze-free road imagery by inverting the atmospheric scattering model:
    J(x) = (I(x) - A) / max(t(x), t_min) + A

    Args:
        image_bgr: Input hazy/foggy BGR image (uint8 numpy array).
        omega: Haze retention factor (prevents unnatural over-saturation).
        t_min: Transmission floor to avoid noise explosion in deep fog.

    Returns:
        Dehazed BGR image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    # Downsample for fast atmospheric light and dark channel estimation
    scale = min(1.0, 360.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr

    dark_ch = compute_dark_channel(small, patch_size=15)
    atm_light = estimate_atmospheric_light(small, dark_ch)

    # Estimate transmission map on original resolution
    raw_t = estimate_transmission_map(image_bgr, atm_light, omega=omega, patch_size=15)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    refined_t = guided_filter_refinement(gray, raw_t, radius=15, eps=1e-3)

    # Invert atmospheric scattering model
    img_f = image_bgr.astype(np.float32)
    t_3ch = np.repeat(np.maximum(refined_t, t_min)[:, :, np.newaxis], 3, axis=2)

    recovered = (img_f - atm_light) / t_3ch + atm_light
    return np.clip(recovered, 0, 255).astype(np.uint8)
