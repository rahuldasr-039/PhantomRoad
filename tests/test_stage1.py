"""
Unit tests for Stage 1: Video Ingestion, Frame Extraction, and Optical Metrics.
"""

import os
import tempfile
import numpy as np
import cv2
import pytest

from src.video.frame_extractor import VideoLoader, VideoMetadata, validate_video_path
from src.video.synthetic_generator import generate_synthetic_road_video
from src.metrics.blur import compute_blur_metrics, compute_laplacian_map
from src.metrics.illumination import compute_illumination_metrics, compute_intensity_histogram


@pytest.fixture
def synthetic_video_path():
    """Generates a temporary small valid MP4 road video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "valid_test_video.mp4")
        generate_synthetic_road_video(
            scenario="clear_day",
            output_path=path,
            num_frames=12,
            fps=24.0,
            width=320,
            height=240
        )
        yield path


@pytest.fixture
def sharp_frame() -> np.ndarray:
    """Creates a high-contrast synthetic road frame with edges."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[0:100, :] = [210, 180, 140]  # Sky
    img[100:, :] = [70, 70, 70]      # Road
    cv2.line(img, (160, 100), (40, 240), (255, 255, 255), 4)  # Left lane
    cv2.line(img, (160, 100), (280, 240), (255, 255, 255), 4) # Right lane
    for y in range(110, 240, 30):
        cv2.line(img, (160, y), (160, y + 15), (0, 255, 255), 3) # Dashed line
    return img


@pytest.fixture
def blurred_frame(sharp_frame: np.ndarray) -> np.ndarray:
    """Applies heavy Gaussian blur to simulate severe camera defocus."""
    return cv2.GaussianBlur(sharp_frame, (25, 25), 10.0)


# =====================================================================
# 1. VIDEO VALIDATION TESTS
# =====================================================================

def test_validate_video_path_valid(synthetic_video_path):
    is_valid, msg, meta = validate_video_path(synthetic_video_path)
    assert is_valid is True
    assert meta is not None
    assert meta.total_frames == 12
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 24.0
    assert meta.duration_sec == 0.5


def test_validate_video_path_invalid_inputs():
    # Empty string
    is_valid, msg, meta = validate_video_path("")
    assert is_valid is False
    assert meta is None

    # Non-string
    is_valid, msg, meta = validate_video_path(None)
    assert is_valid is False
    assert meta is None

    # Non-existent file
    is_valid, msg, meta = validate_video_path("non_existent_file.mp4")
    assert is_valid is False
    assert "not found" in msg.lower()


def test_validate_video_path_unsupported_extension():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"plain text file")
        txt_path = f.name

    try:
        is_valid, msg, meta = validate_video_path(txt_path)
        assert is_valid is False
        assert "unsupported file extension" in msg.lower()
    finally:
        if os.path.exists(txt_path):
            os.remove(txt_path)


def test_validate_video_path_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        empty_path = f.name  # 0 bytes

    try:
        is_valid, msg, meta = validate_video_path(empty_path)
        assert is_valid is False
        assert "empty (0 bytes)" in msg.lower()
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)


def test_validate_video_path_corrupt_file():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"THIS IS NOT A VALID VIDEO HEADER OR STREAM")
        corrupt_path = f.name

    try:
        is_valid, msg, meta = validate_video_path(corrupt_path)
        assert is_valid is False
        assert "failed" in msg.lower() or "invalid" in msg.lower()
    finally:
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)


# =====================================================================
# 2. VIDEO LOADER & FRAME EXTRACTION TESTS
# =====================================================================

def test_video_loader_operations(synthetic_video_path):
    with VideoLoader(synthetic_video_path) as loader:
        assert loader.total_frames == 12
        assert loader.width == 320
        assert loader.height == 240

        # Valid frame extraction
        frame_0 = loader.get_frame(0)
        assert frame_0 is not None
        assert frame_0.shape == (240, 320, 3)
        assert frame_0.dtype == np.uint8

        # Cache check
        frame_0_cached = loader.get_frame(0)
        assert frame_0_cached is not None
        np.testing.assert_array_equal(frame_0, frame_0_cached)

        # Boundary checks
        assert loader.get_frame(-1) is None
        assert loader.get_frame(12) is None
        assert loader.get_frame(999) is None

        # Sequential iteration
        frames_iterated = list(loader.iterate_frames())
        assert len(frames_iterated) == 12
        for idx, ts, f in frames_iterated:
            assert f.shape == (240, 320, 3)
            assert ts == pytest.approx(idx / 24.0, abs=1e-3)


def test_video_loader_invalid_initialization():
    # Invalid types
    with pytest.raises(TypeError):
        VideoLoader(12345)

    with pytest.raises(TypeError):
        VideoLoader("")

    # Non-existent
    with pytest.raises(FileNotFoundError):
        VideoLoader("does_not_exist.mp4")

    # 0-byte file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        empty_path = f.name
    try:
        with pytest.raises(ValueError):
            VideoLoader(empty_path)
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)


def test_video_loader_get_frame_type_error(synthetic_video_path):
    with VideoLoader(synthetic_video_path) as loader:
        with pytest.raises(TypeError):
            loader.get_frame("first")


# =====================================================================
# 3. BLUR ESTIMATION METRICS TESTS
# =====================================================================

def test_blur_metrics_calculation(sharp_frame, blurred_frame):
    sharp_m = compute_blur_metrics(sharp_frame, blur_threshold=85.0)
    blurred_m = compute_blur_metrics(blurred_frame, blur_threshold=85.0)

    # Real metric values verification
    assert sharp_m["laplacian_variance"] > blurred_m["laplacian_variance"]
    assert sharp_m["is_blurred"] is False
    assert blurred_m["is_blurred"] is True
    assert sharp_m["sharpness_score"] > blurred_m["sharpness_score"]
    assert sharp_m["fft_high_freq_ratio"] > 0.0
    assert sharp_m["threshold"] == 85.0


def test_blur_metrics_input_validation(sharp_frame):
    # None input returns fallback
    none_res = compute_blur_metrics(None)
    assert none_res["is_blurred"] is True
    assert none_res["laplacian_variance"] == 0.0

    # Non-numpy input raises TypeError
    with pytest.raises(TypeError):
        compute_blur_metrics([[1, 2], [3, 4]])

    # Invalid threshold raises ValueError
    with pytest.raises(ValueError):
        compute_blur_metrics(sharp_frame, blur_threshold=-10.0)

    # Empty array returns fallback
    empty_res = compute_blur_metrics(np.array([], dtype=np.uint8))
    assert empty_res["is_blurred"] is True

    # 1D array raises ValueError
    with pytest.raises(ValueError):
        compute_blur_metrics(np.array([1, 2, 3], dtype=np.uint8))


def test_blur_metrics_formats(sharp_frame):
    # Grayscale
    gray = cv2.cvtColor(sharp_frame, cv2.COLOR_BGR2GRAY)
    gray_m = compute_blur_metrics(gray)
    assert gray_m["laplacian_variance"] > 0.0

    # 4-Channel BGRA
    bgra = cv2.cvtColor(sharp_frame, cv2.COLOR_BGR2BGRA)
    bgra_m = compute_blur_metrics(bgra)
    assert bgra_m["laplacian_variance"] > 0.0

    # Float array [0.0 - 1.0]
    float_img = sharp_frame.astype(np.float32) / 255.0
    float_m = compute_blur_metrics(float_img)
    assert float_m["laplacian_variance"] > 0.0


def test_compute_laplacian_map(sharp_frame):
    lap_map = compute_laplacian_map(sharp_frame)
    assert isinstance(lap_map, np.ndarray)
    assert lap_map.shape == (240, 320)
    assert lap_map.dtype == np.uint8
    assert np.max(lap_map) > 0


# =====================================================================
# 4. BRIGHTNESS & CONTRAST METRICS TESTS
# =====================================================================

def test_illumination_metrics_calculation(sharp_frame):
    normal_m = compute_illumination_metrics(sharp_frame)

    # Real calculated metric values
    assert 50.0 < normal_m["mean_brightness"] < 200.0
    assert normal_m["rms_contrast"] > 0.10
    assert normal_m["dynamic_range"] > 50
    assert 0.0 <= normal_m["underexposed_ratio"] <= 1.0
    assert 0.0 <= normal_m["overexposed_ratio"] <= 1.0
    assert normal_m["entropy"] > 0.0
    assert normal_m["is_low_light"] is False
    assert normal_m["is_overexposed"] is False
    assert normal_m["is_low_contrast"] is False


def test_illumination_low_light_and_overexposed(sharp_frame):
    # Dark road condition
    dark_frame = (sharp_frame * 0.10).astype(np.uint8)
    dark_m = compute_illumination_metrics(dark_frame, low_light_threshold=45.0)
    assert dark_m["mean_brightness"] < 45.0
    assert dark_m["is_low_light"] is True
    assert dark_m["underexposed_ratio"] > 0.50

    # Overexposed road condition
    bright_frame = np.clip(sharp_frame.astype(np.float32) + 160.0, 0, 255).astype(np.uint8)
    bright_m = compute_illumination_metrics(bright_frame, overexposed_threshold=200.0)
    assert bright_m["mean_brightness"] > 200.0
    assert bright_m["is_overexposed"] is True
    assert bright_m["overexposed_ratio"] > 0.20


def test_illumination_low_contrast():
    # Uniform gray with very low standard deviation
    flat_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    flat_m = compute_illumination_metrics(flat_frame, min_rms_contrast=0.12)
    assert flat_m["rms_contrast"] < 0.01
    assert flat_m["is_low_contrast"] is True
    assert flat_m["dynamic_range"] == 0


def test_illumination_metrics_input_validation():
    # None input returns fallback
    none_m = compute_illumination_metrics(None)
    assert none_m["is_low_light"] is True
    assert none_m["mean_brightness"] == 0.0

    # Non-numpy input raises TypeError
    with pytest.raises(TypeError):
        compute_illumination_metrics("invalid_image")

    # Invalid thresholds raise ValueError
    with pytest.raises(ValueError):
        compute_illumination_metrics(np.zeros((10, 10), dtype=np.uint8), low_light_threshold=-5.0)

    with pytest.raises(ValueError):
        compute_illumination_metrics(
            np.zeros((10, 10), dtype=np.uint8),
            low_light_threshold=100.0,
            overexposed_threshold=50.0  # low_light >= overexposed
        )


def test_illumination_edge_cases():
    # Pure black image
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    b_m = compute_illumination_metrics(black_img)
    assert b_m["mean_brightness"] == 0.0
    assert b_m["rms_contrast"] == 0.0
    assert b_m["underexposed_ratio"] == 1.0
    assert b_m["is_low_light"] is True

    # Pure white image
    white_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    w_m = compute_illumination_metrics(white_img)
    assert w_m["mean_brightness"] == 255.0
    assert w_m["rms_contrast"] == 0.0
    assert w_m["overexposed_ratio"] == 1.0
    assert w_m["is_overexposed"] is True


def test_compute_intensity_histogram(sharp_frame):
    hist = compute_intensity_histogram(sharp_frame)
    assert "bins" in hist
    assert len(hist["bins"]) == 256
    assert "luminance" in hist
    assert len(hist["luminance"]) == 256
    assert sum(hist["luminance"]) == 240 * 320
    assert "red" in hist and "green" in hist and "blue" in hist
