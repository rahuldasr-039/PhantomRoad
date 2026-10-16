"""
Unit tests for computer vision quality metrics.
"""

import numpy as np
import cv2
import pytest

from src.metrics.blur import compute_blur_metrics
from src.metrics.illumination import compute_illumination_metrics
from src.metrics.visibility import compute_visibility_metrics
from src.metrics.obstruction import compute_spatial_obstruction, TemporalObstructionFilter
from src.metrics.quality_assessor import QualityAssessor, FrameQualityMetrics


@pytest.fixture
def sharp_road_image() -> np.ndarray:
    """Creates a high-contrast synthetic road frame with road lanes and texture."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Sky
    img[0:200, :] = [210, 180, 140]
    # Road surface
    img[200:, :] = [70, 70, 70]
    # Lane markings (high contrast edges)
    cv2.line(img, (320, 200), (100, 480), (255, 255, 255), 8)
    cv2.line(img, (320, 200), (540, 480), (255, 255, 255), 8)
    # Dashed center line
    for y in range(220, 480, 40):
        cv2.line(img, (320, y), (320, y + 20), (0, 255, 255), 5)
    return img


@pytest.fixture
def blurred_road_image(sharp_road_image: np.ndarray) -> np.ndarray:
    """Applies heavy Gaussian blur to simulate camera motion/defocus."""
    return cv2.GaussianBlur(sharp_road_image, (35, 35), 15.0)


@pytest.fixture
def low_light_image(sharp_road_image: np.ndarray) -> np.ndarray:
    """Simulates dark road conditions."""
    return (sharp_road_image * 0.15).astype(np.uint8)


@pytest.fixture
def hazy_fog_image(sharp_road_image: np.ndarray) -> np.ndarray:
    """Simulates atmospheric scattering / fog by blending with atmospheric light."""
    fog_layer = np.full_like(sharp_road_image, 215, dtype=np.uint8)
    return cv2.addWeighted(sharp_road_image, 0.35, fog_layer, 0.65, 0)


@pytest.fixture
def obstructed_image(sharp_road_image: np.ndarray) -> np.ndarray:
    """Simulates mud / tape blockage over 40% of the lens."""
    obs = sharp_road_image.copy()
    obs[150:400, 100:400] = [45, 50, 45]  # Flat matte mud blob
    return obs


def test_blur_metrics(sharp_road_image, blurred_road_image):
    sharp_m = compute_blur_metrics(sharp_road_image, blur_threshold=80.0)
    blurred_m = compute_blur_metrics(blurred_road_image, blur_threshold=80.0)

    assert sharp_m["laplacian_variance"] > blurred_m["laplacian_variance"]
    assert not sharp_m["is_blurred"]
    assert blurred_m["is_blurred"]
    assert sharp_m["sharpness_score"] > blurred_m["sharpness_score"]


def test_illumination_metrics(sharp_road_image, low_light_image):
    normal_m = compute_illumination_metrics(sharp_road_image)
    dark_m = compute_illumination_metrics(low_light_image)

    assert normal_m["mean_brightness"] > dark_m["mean_brightness"]
    assert dark_m["is_low_light"]
    assert not normal_m["is_low_light"]
    assert dark_m["underexposed_ratio"] > 0.30


def test_visibility_metrics(sharp_road_image, hazy_fog_image):
    clear_m = compute_visibility_metrics(sharp_road_image)
    hazy_m = compute_visibility_metrics(hazy_fog_image)

    assert clear_m["mean_transmission"] > hazy_m["mean_transmission"]
    assert hazy_m["is_hazy_or_foggy"]
    assert hazy_m["haze_level"] > clear_m["haze_level"]


def test_temporal_obstruction_persistence(sharp_road_image, obstructed_image):
    tracker = TemporalObstructionFilter(history_len=5, min_persistence_frames=3)

    # Frame 1: obstructed image (transient trigger, not yet persistent)
    obs_m1 = compute_spatial_obstruction(obstructed_image)
    res1 = tracker.update(obs_m1)
    assert res1["is_transient"]
    assert not res1["is_persistent_obstruction"]

    # Frame 2: obstructed
    res2 = tracker.update(obs_m1)
    assert not res2["is_persistent_obstruction"]

    # Frame 3: obstructed (reaches min_persistence_frames=3 -> persistent)
    res3 = tracker.update(obs_m1)
    assert res3["is_persistent_obstruction"]
    assert res3["persistence_streak"] >= 3

    # Cleared frame: wiper clears obstruction
    clear_m = compute_spatial_obstruction(sharp_road_image)
    res4 = tracker.update(clear_m)
    assert not res4["is_persistent_obstruction"]


def test_quality_assessor_full_telemetry(sharp_road_image, hazy_fog_image):
    assessor = QualityAssessor()
    telemetry_clear = assessor.assess_frame(sharp_road_image, frame_index=0, timestamp_sec=0.0)
    assert isinstance(telemetry_clear, FrameQualityMetrics)
    assert telemetry_clear.composite_quality_score > 0.50
    assert len(telemetry_clear.measurable_degradations) == 0

    assessor.reset_temporal_state()
    telemetry_hazy = assessor.assess_frame(hazy_fog_image, frame_index=1, timestamp_sec=0.033)
    assert telemetry_hazy.is_hazy_or_foggy
    assert len(telemetry_hazy.measurable_degradations) > 0
    assert any("ATMOSPHERIC_ATTENUATION" in d for d in telemetry_hazy.measurable_degradations)
