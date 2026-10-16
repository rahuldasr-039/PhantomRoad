"""
Unit tests for OpenCV enhancements and before/after verification.
"""

import numpy as np
import cv2
import pytest

from src.enhancement.contrast_clahe import enhance_contrast_clahe
from src.enhancement.illumination_gamma import correct_illumination_gamma
from src.enhancement.sharpness_unsharp import enhance_sharpness_unsharp_mask
from src.enhancement.dehaze_dcp import dehaze_dark_channel
from src.enhancement.denoise import denoise_edge_preserving
from src.enhancement.verification import evaluate_enhancement_improvement
from src.metrics.quality_assessor import QualityAssessor


@pytest.fixture
def dark_road_image() -> np.ndarray:
    """Creates a dark low-light road frame."""
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[0:150, :] = [25, 20, 20]  # Night sky
    img[150:, :] = [20, 20, 20]   # Dark asphalt
    cv2.line(img, (240, 150), (60, 360), (90, 90, 90), 4)  # Dim road lane
    return img


@pytest.fixture
def foggy_road_image() -> np.ndarray:
    """Creates a hazy/foggy road frame with low contrast."""
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[0:150, :] = [180, 180, 190]
    img[150:, :] = [160, 160, 165]
    cv2.line(img, (240, 150), (60, 360), (220, 220, 225), 4)
    return img


@pytest.fixture
def soft_blur_image() -> np.ndarray:
    """Creates a soft blurred image."""
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[150:, :] = [80, 80, 80]
    cv2.line(img, (240, 150), (60, 360), (255, 255, 255), 6)
    return cv2.GaussianBlur(img, (15, 15), 4.0)


def test_enhance_contrast_clahe(foggy_road_image):
    enhanced = enhance_contrast_clahe(foggy_road_image, clip_limit=3.0)
    assert enhanced.shape == foggy_road_image.shape
    # Standard deviation / contrast should increase
    std_before = np.std(foggy_road_image)
    std_after = np.std(enhanced)
    assert std_after > std_before


def test_correct_illumination_gamma(dark_road_image):
    brightened = correct_illumination_gamma(dark_road_image, gamma=2.0)
    assert brightened.shape == dark_road_image.shape
    mean_before = np.mean(dark_road_image)
    mean_after = np.mean(brightened)
    assert mean_after > mean_before


def test_enhance_sharpness_unsharp(soft_blur_image):
    sharpened = enhance_sharpness_unsharp_mask(soft_blur_image, amount=1.5)
    assert sharpened.shape == soft_blur_image.shape
    lap_before = cv2.Laplacian(cv2.cvtColor(soft_blur_image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    lap_after = cv2.Laplacian(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert lap_after > lap_before


def test_dehaze_dark_channel(foggy_road_image):
    dehazed = dehaze_dark_channel(foggy_road_image)
    assert dehazed.shape == foggy_road_image.shape
    # Foggy images have low standard deviation; dehazing restores dynamic contrast
    assert np.std(dehazed) > np.std(foggy_road_image)


def test_verification_acceptance(dark_road_image):
    assessor = QualityAssessor()
    m_before = assessor.assess_frame(dark_road_image, 0)
    brightened = correct_illumination_gamma(dark_road_image, gamma=2.2)
    m_after = assessor.assess_frame(brightened, 0)

    res = evaluate_enhancement_improvement(m_before, m_after, action_name="APPLY_GAMMA_CORRECTION")
    assert res.accepted
    assert "Accepted" in res.reason
    assert res.metric_deltas["brightness_delta"] > 0
