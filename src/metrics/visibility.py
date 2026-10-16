"""
Visibility and degradation indicators using Dark Channel Prior (DCP) and Glare analysis.
Based on He et al. (IEEE TPAMI 2011) and specular highlight extraction.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


def compute_dark_channel(image_bgr: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Computes the dark channel prior map: J_dark(x) = min_{c} ( min_{y in patch} (I^c(y)) ).
    """
    # Channel-wise minimum across B, G, R
    min_channel = np.min(image_bgr, axis=2)
    # Patch-wise minimum using minimum filter (morphological erosion with rectangle kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def estimate_atmospheric_light(
    image_bgr: np.ndarray,
    dark_channel: np.ndarray,
    top_fraction: float = 0.001
) -> np.ndarray:
    """
    Estimates global atmospheric light A from the top brightest pixels in the dark channel.
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_candidates = max(1, int(num_pixels * top_fraction))

    # Flatten indices sorted by dark channel brightness descending
    flat_indices = np.argsort(dark_channel.ravel())[::-1][:num_candidates]

    flat_img = image_bgr.reshape(-1, 3)
    candidate_pixels = flat_img[flat_indices]

    # Select the candidate with highest intensity sum (or average across candidates)
    intensities = np.sum(candidate_pixels, axis=1)
    brightest_idx = np.argmax(intensities)
    atmospheric_light = candidate_pixels[brightest_idx].astype(np.float32)

    # Clamp to avoid division by zero or extreme darkness
    atmospheric_light = np.clip(atmospheric_light, 10.0, 255.0)
    return atmospheric_light


def estimate_transmission_map(
    image_bgr: np.ndarray,
    atmospheric_light: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15
) -> np.ndarray:
    """
    Estimates rough transmission map t(x) = 1 - omega * min_c ( min_y (I^c(y) / A^c) ).
    """
    normalized = image_bgr.astype(np.float32) / atmospheric_light
    min_channel = np.min(normalized, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_normalized = cv2.erode(min_channel, kernel)
    transmission = 1.0 - omega * dark_normalized
    return np.clip(transmission, 0.05, 1.0)


def compute_glare_metrics(image_bgr: np.ndarray, glare_threshold: int = 245) -> Dict[str, Any]:
    """
    Detects high-intensity blooming and specular glare clusters.
    """
    # Grayscale or minimum channel for glare clipping
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray >= glare_threshold).astype(np.uint8)

    glare_pixels = int(np.count_nonzero(mask))
    total_pixels = float(gray.size)
    glare_ratio = float(glare_pixels / total_pixels)

    # Connected components to distinguish small bright headlights from massive blinding glare
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    max_blob_area = 0
    max_blob_ratio = 0.0
    if num_labels > 1:
        # Ignore background label 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            max_blob_area = int(np.max(areas))
            max_blob_ratio = float(max_blob_area / total_pixels)

    is_severe_glare = bool(glare_ratio > 0.08 or max_blob_ratio > 0.03)

    return {
        "glare_ratio": round(glare_ratio, 4),
        "max_glare_blob_ratio": round(max_blob_ratio, 4),
        "is_severe_glare": is_severe_glare
    }


def compute_visibility_metrics(
    image: np.ndarray,
    min_acceptable_transmission: float = 0.38
) -> Dict[str, Any]:
    """
    Computes comprehensive visibility indicators including Dark Channel transmission,
    haze level, and specular glare.

    Args:
        image: Input BGR image (uint8 numpy array).
        min_acceptable_transmission: Minimum transmission before scene is flagged as haze/fog degraded.

    Returns:
        dict containing:
            - mean_transmission (float): Estimated atmospheric transmission [0.0 - 1.0].
            - dark_channel_mean (float): Average dark channel value [0 - 255].
            - atmospheric_light (list): Estimated BGR atmospheric light.
            - haze_level (float): Estimated haze index [0.0 - 1.0] (1.0 = thick fog/scattering).
            - is_hazy_or_foggy (bool): Measurable attenuation below threshold.
            - glare_ratio (float)
            - is_severe_glare (bool)
            - color_cast_score (float): Chromaticity imbalance indicator.
    """
    if image is None or image.size == 0:
        return {
            "mean_transmission": 0.0,
            "dark_channel_mean": 255.0,
            "atmospheric_light": [255.0, 255.0, 255.0],
            "haze_level": 1.0,
            "is_hazy_or_foggy": True,
            "glare_ratio": 0.0,
            "is_severe_glare": False,
            "color_cast_score": 0.0
        }

    if len(image.shape) == 2 or image.shape[2] == 1:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image

    # Downsample for fast processing in video pipelines
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 320.0 / max(h, w))
    if scale < 1.0:
        small_img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_img = img_bgr

    # 1. Dark Channel Prior
    dark_ch = compute_dark_channel(small_img, patch_size=15)
    dark_mean = float(np.mean(dark_ch))

    # 2. Atmospheric Light
    atm_light = estimate_atmospheric_light(small_img, dark_ch)

    # 3. Transmission Map
    transmission = estimate_transmission_map(small_img, atm_light, omega=0.95, patch_size=15)

    # In road imagery, compute transmission on the lower 75% to prevent sky false positives
    road_trans = transmission[int(small_img.shape[0] * 0.25):, :]
    mean_trans = float(np.mean(road_trans)) if road_trans.size > 0 else float(np.mean(transmission))

    # Haze level inversely related to transmission
    haze_level = float(np.clip(1.0 - mean_trans, 0.0, 1.0))
    is_hazy = bool((mean_trans < min_acceptable_transmission and dark_mean > 50.0) or dark_mean > 120.0)

    # 4. Glare analysis
    glare_info = compute_glare_metrics(small_img)

    # 5. Color Cast / Chromaticity Disparity
    means = np.mean(small_img, axis=(0, 1))  # B, G, R
    overall_mean = float(np.mean(means)) + 1e-6
    color_cast_score = float((np.abs(means[0] - means[1]) + np.abs(means[1] - means[2]) + np.abs(means[2] - means[0])) / (3.0 * overall_mean))

    return {
        "mean_transmission": round(mean_trans, 4),
        "dark_channel_mean": round(dark_mean, 2),
        "atmospheric_light": [round(float(c), 1) for c in atm_light],
        "haze_level": round(haze_level, 4),
        "is_hazy_or_foggy": is_hazy,
        "glare_ratio": glare_info["glare_ratio"],
        "max_glare_blob_ratio": glare_info["max_glare_blob_ratio"],
        "is_severe_glare": glare_info["is_severe_glare"],
        "color_cast_score": round(color_cast_score, 4),
    }
