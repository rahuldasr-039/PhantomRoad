"""
Illumination and contrast profiling metrics for road scenes.
"""

from typing import Dict, Any
import cv2
import numpy as np


def compute_illumination_metrics(
    image: np.ndarray,
    low_light_threshold: float = 45.0,
    overexposed_threshold: float = 200.0,
    min_rms_contrast: float = 0.12
) -> Dict[str, Any]:
    """
    Computes brightness, RMS contrast, clipping fractions, and dynamic range.

    Args:
        image: Input image (BGR, BGRA, or Grayscale numpy array).
        low_light_threshold: Luminance below which frame is flagged as low-light.
        overexposed_threshold: Luminance above which frame is flagged as overexposed.
        min_rms_contrast: Minimum RMS contrast for acceptable road edge separation.

    Returns:
        dict containing:
            - mean_brightness (float): Mean luminance [0 - 255].
            - rms_contrast (float): Root mean square contrast [0.0 - 1.0].
            - dynamic_range (int): 95th - 5th intensity percentile [0 - 255].
            - underexposed_ratio (float): Ratio of pixels in deep shadow (< 20).
            - overexposed_ratio (float): Ratio of pixels in specular clipping (> 235).
            - entropy (float): Shannon entropy of intensity histogram.
            - is_low_light (bool)
            - is_overexposed (bool)
            - is_low_contrast (bool)
    """
    if image is None:
        return {
            "mean_brightness": 0.0,
            "rms_contrast": 0.0,
            "dynamic_range": 0,
            "underexposed_ratio": 1.0,
            "overexposed_ratio": 0.0,
            "entropy": 0.0,
            "is_low_light": True,
            "is_overexposed": False,
            "is_low_contrast": True
        }

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be a numpy.ndarray, got {type(image).__name__}")

    if low_light_threshold < 0 or overexposed_threshold < 0 or min_rms_contrast < 0:
        raise ValueError("Threshold parameters must be non-negative.")

    if low_light_threshold >= overexposed_threshold:
        raise ValueError(
            f"low_light_threshold ({low_light_threshold}) must be strictly less than "
            f"overexposed_threshold ({overexposed_threshold})."
        )

    if image.size == 0:
        return {
            "mean_brightness": 0.0,
            "rms_contrast": 0.0,
            "dynamic_range": 0,
            "underexposed_ratio": 1.0,
            "overexposed_ratio": 0.0,
            "entropy": 0.0,
            "is_low_light": True,
            "is_overexposed": False,
            "is_low_contrast": True
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

    # Extract luminance channel
    if uint8_img.ndim == 3:
        channels = uint8_img.shape[2]
        if channels == 3:
            ycrcb = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2YCrCb)
            lum = ycrcb[:, :, 0]
        elif channels == 4:
            bgr = uint8_img[:, :, :3]
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            lum = ycrcb[:, :, 0]
        elif channels == 1:
            lum = uint8_img[:, :, 0]
        else:
            raise ValueError(f"Expected 1, 3, or 4 color channels, got {channels}")
    else:
        lum = uint8_img.copy()

    total_pixels = float(lum.size)
    mean_val = float(np.mean(lum))
    std_val = float(np.std(lum))

    # RMS Contrast: standard deviation normalized to [0, 1]
    rms_contrast = float(std_val / 255.0)

    # Intensity percentiles for effective dynamic range
    p5 = float(np.percentile(lum, 5))
    p95 = float(np.percentile(lum, 95))
    dynamic_range = int(round(p95 - p5))

    # Under and over exposure clipping fractions
    under_ratio = float(np.count_nonzero(lum < 20) / total_pixels)
    over_ratio = float(np.count_nonzero(lum > 235) / total_pixels)

    # Shannon Entropy of 256-bin histogram
    hist, _ = np.histogram(lum, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

    is_low_light = bool(mean_val < low_light_threshold or under_ratio > 0.40)
    is_overexposed = bool(mean_val > overexposed_threshold or over_ratio > 0.25)
    is_low_contrast = bool(rms_contrast < min_rms_contrast)

    return {
        "mean_brightness": round(mean_val, 2),
        "rms_contrast": round(rms_contrast, 4),
        "dynamic_range": dynamic_range,
        "underexposed_ratio": round(under_ratio, 4),
        "overexposed_ratio": round(over_ratio, 4),
        "entropy": round(entropy, 3),
        "is_low_light": is_low_light,
        "is_overexposed": is_overexposed,
        "is_low_contrast": is_low_contrast,
    }


def compute_intensity_histogram(image: np.ndarray) -> Dict[str, Any]:
    """
    Computes 256-bin histograms for luminance and individual RGB channels.

    Args:
        image: Input image (BGR or Grayscale uint8 array).

    Returns:
        dict containing histogram bin frequencies for plotting.
    """
    if image is None or image.size == 0:
        empty_bins = [0] * 256
        return {"bins": list(range(256)), "luminance": empty_bins}

    if image.ndim == 3 and image.shape[2] >= 3:
        bgr = image[:, :, :3]
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lum = ycrcb[:, :, 0]
        lum_hist, _ = np.histogram(lum, bins=256, range=(0, 256))
        b_hist, _ = np.histogram(bgr[:, :, 0], bins=256, range=(0, 256))
        g_hist, _ = np.histogram(bgr[:, :, 1], bins=256, range=(0, 256))
        r_hist, _ = np.histogram(bgr[:, :, 2], bins=256, range=(0, 256))
        return {
            "bins": list(range(256)),
            "luminance": lum_hist.tolist(),
            "blue": b_hist.tolist(),
            "green": g_hist.tolist(),
            "red": r_hist.tolist()
        }
    else:
        gray = image if image.ndim == 2 else image[:, :, 0]
        lum_hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        return {
            "bins": list(range(256)),
            "luminance": lum_hist.tolist()
        }
