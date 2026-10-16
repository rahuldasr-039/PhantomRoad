"""
Unit and integration tests for Stage 3: Downstream Road-Scene Object Detector & Reliability Annotation.
Tests real MobileNet-SSD inference, clear vs degraded comparison, no hallucinated detections,
offline fallback with actionable setup message, and reliability status decoration.
"""

import os
import numpy as np
import cv2
import pytest

from src.detector.base import BaseRoadSceneDetector, DetectionBox, DetectionResult
from src.detector.opencv_dnn import OpenCVDNNDetector, ROAD_CLASSES
from src.detector.reliability_wrapper import ReliableRoadSceneAnalyzer
from src.detector.comparator import RoadScenePerceptionComparator, PerceptionComparisonReport
from src.agent.state import ReliabilityState, AgentFrameDecision
from src.metrics.quality_assessor import FrameQualityMetrics
from src.video.synthetic_generator import _draw_base_road_frame


@pytest.fixture
def base_road_scene() -> np.ndarray:
    """Creates a high-contrast road scene with a moving vehicle."""
    return _draw_base_road_frame(frame_idx=5, total_frames=20, width=480, height=360)


@pytest.fixture
def degraded_road_scene(base_road_scene: np.ndarray) -> np.ndarray:
    """Applies atmospheric fog and Gaussian blur to degrade visual quality."""
    fog_layer = np.full_like(base_road_scene, 190, dtype=np.uint8)
    hazy = cv2.addWeighted(base_road_scene, 0.25, fog_layer, 0.75, 0)
    return cv2.GaussianBlur(hazy, (25, 25), 10.0)


def _make_decision(state: ReliabilityState) -> AgentFrameDecision:
    dummy_m = FrameQualityMetrics(
        frame_index=0,
        timestamp_sec=0.0,
        laplacian_variance=140.0 if state == ReliabilityState.RELIABLE else 25.0,
        fft_high_freq_ratio=0.35,
        sharpness_score=0.8,
        is_blurred=(state != ReliabilityState.RELIABLE),
        mean_brightness=115.0,
        rms_contrast=0.22 if state == ReliabilityState.RELIABLE else 0.08,
        dynamic_range=110,
        underexposed_ratio=0.0,
        overexposed_ratio=0.0,
        entropy=4.5,
        is_low_light=False,
        is_overexposed=False,
        is_low_contrast=(state != ReliabilityState.RELIABLE),
        mean_transmission=0.72 if state == ReliabilityState.RELIABLE else 0.35,
        dark_channel_mean=32.0,
        haze_level=0.28,
        is_hazy_or_foggy=(state != ReliabilityState.RELIABLE),
        glare_ratio=0.0,
        is_severe_glare=False,
        color_cast_score=0.02,
        obstructed_cell_ratio=0.0 if state != ReliabilityState.CRITICAL_OBSTRUCTION else 0.45,
        global_flatness_score=0.0,
        is_instantaneous_obstruction=(state == ReliabilityState.CRITICAL_OBSTRUCTION),
        is_persistent_obstruction=(state == ReliabilityState.CRITICAL_OBSTRUCTION),
        temporal_persistence_streak=3 if state == ReliabilityState.CRITICAL_OBSTRUCTION else 0,
        is_transient_occlusion=False,
        composite_quality_score=0.78 if state == ReliabilityState.RELIABLE else 0.38,
        measurable_degradations=["BLUR", "FOG"] if state != ReliabilityState.RELIABLE else []
    )
    return AgentFrameDecision(
        frame_index=0,
        timestamp_sec=0.0,
        initial_state=state,
        final_state=state,
        initial_metrics=dummy_m,
        final_metrics=dummy_m,
        attempts_made=0,
        max_attempts=3,
        human_review_required=(state in (ReliabilityState.CRITICAL_OBSTRUCTION, ReliabilityState.HUMAN_REVIEW_REQUIRED))
    )


# =====================================================================
# 1. MODEL AVAILABILITY & CONFIGURATION TESTS
# =====================================================================

def test_detector_model_availability():
    detector = OpenCVDNNDetector()
    # If weights exist on disk, must be available and initialized
    weights_exist = os.path.exists("models/MobileNetSSD_deploy.caffemodel")
    if weights_exist:
        assert detector.is_available() is True
        assert detector.net is not None
        assert detector.model_name == "MobileNet-SSD (OpenCV DNN)"
    else:
        assert detector.is_available() is False


def test_detector_offline_graceful_fallback():
    """
    Verifies that when weights are absent or path is invalid:
    - Never throws an unhandled exception.
    - is_available() returns False.
    - detect() returns 0 detections (never fake results).
    - setup_message contains actionable instructions.
    """
    offline_detector = OpenCVDNNDetector(
        prototxt_path="models/nonexistent.prototxt",
        weights_path="models/nonexistent.caffemodel"
    )

    assert offline_detector.is_available() is False
    assert offline_detector.net is None

    # Run inference on frame
    blank_frame = np.zeros((360, 480, 3), dtype=np.uint8)
    res = offline_detector.detect(blank_frame)

    assert isinstance(res, DetectionResult)
    assert res.model_available is False
    assert len(res.detections) == 0
    assert res.setup_message is not None
    assert "download_models.py" in res.setup_message
    assert "models/" in res.setup_message


# =====================================================================
# 2. HONEST INFERENCE TESTS (NO HALLUCINATIONS)
# =====================================================================

def test_detector_never_invents_detections():
    """
    Verifies that the detector returns 0 detections on blank, saturated, or noise frames.
    It must never invent or simulate fake obstacle detections.
    """
    detector = OpenCVDNNDetector()
    if not detector.is_available():
        pytest.skip("MobileNet-SSD weights not found in models/.")

    # 1. Pure black frame
    black_frame = np.zeros((360, 480, 3), dtype=np.uint8)
    black_res = detector.detect(black_frame, conf_threshold=0.30)
    assert len(black_res.detections) == 0

    # 2. Pure white saturated frame
    white_frame = np.full((360, 480, 3), 255, dtype=np.uint8)
    white_res = detector.detect(white_frame, conf_threshold=0.30)
    assert len(white_res.detections) == 0

    # 3. Uniform gray frame
    gray_frame = np.full((360, 480, 3), 128, dtype=np.uint8)
    gray_res = detector.detect(gray_frame, conf_threshold=0.30)
    assert len(gray_res.detections) == 0


def test_detector_inference_on_actual_frame(base_road_scene):
    """
    Tests real inference execution on a synthetic road frame.
    Verifies that detected coordinates are valid and bounded within frame resolution.
    """
    detector = OpenCVDNNDetector()
    if not detector.is_available():
        pytest.skip("MobileNet-SSD weights not found in models/.")

    h, w = base_road_scene.shape[:2]
    res = detector.detect(base_road_scene, conf_threshold=0.20)

    assert isinstance(res, DetectionResult)
    assert res.model_available is True

    # If any objects detected, verify coordinate validity
    for d in res.detections:
        assert isinstance(d, DetectionBox)
        assert d.label in ROAD_CLASSES
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.box_xyxy
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h


# =====================================================================
# 3. RELIABILITY WRAPPER & SAFETY ANNOTATION TESTS
# =====================================================================

def test_reliability_wrapper_state_decorations(base_road_scene):
    detector = OpenCVDNNDetector()
    analyzer = ReliableRoadSceneAnalyzer(detector)

    # 1. Nominal Reliable Frame
    dec_reliable = _make_decision(ReliabilityState.RELIABLE)
    res_rel = analyzer.analyze_frame(base_road_scene, dec_reliable)
    assert res_rel.frame_reliability_state == "RELIABLE"
    for d in res_rel.detections:
        assert d.is_reliable is True
        assert d.reliability_status == "TRUSTED"
        assert d.suppression_reason is None

    # 2. Degraded Frame (e.g. low light, fog, or blur)
    dec_degraded = _make_decision(ReliabilityState.DEGRADED_CORRECTABLE)
    res_deg = analyzer.analyze_frame(base_road_scene, dec_degraded)
    assert res_deg.frame_reliability_state == "DEGRADED_CORRECTABLE"
    for d in res_deg.detections:
        assert d.is_reliable is False
        assert d.reliability_status == "UNVERIFIED_DEGRADED"
        assert d.suppression_reason is not None
        assert "compromised" in d.suppression_reason.lower()

    # 3. Critical Obstruction / Sensor Failure
    dec_obs = _make_decision(ReliabilityState.CRITICAL_OBSTRUCTION)
    res_obs = analyzer.analyze_frame(base_road_scene, dec_obs)
    assert res_obs.frame_reliability_state == "CRITICAL_OBSTRUCTION"
    assert res_obs.warning_notice is not None
    assert "CRITICAL" in res_obs.warning_notice
    for d in res_obs.detections:
        assert d.is_reliable is False
        assert d.reliability_status == "SUPPRESSED_OBSTRUCTED"


# =====================================================================
# 4. PERCEPTION COMPARATOR TESTS (STAGE 3 EVALUATION)
# =====================================================================

def test_perception_comparator_clear_vs_degraded(base_road_scene, degraded_road_scene):
    detector = OpenCVDNNDetector()
    comparator = RoadScenePerceptionComparator(detector)

    dec_clear = _make_decision(ReliabilityState.RELIABLE)
    dec_deg = _make_decision(ReliabilityState.DEGRADED_CORRECTABLE)

    report = comparator.compare_frames(
        clear_frame=base_road_scene,
        degraded_frame=degraded_road_scene,
        clear_decision=dec_clear,
        degraded_decision=dec_deg,
        conf_threshold=0.25
    )

    assert isinstance(report, PerceptionComparisonReport)
    assert report.clear_result.frame_reliability_state == "RELIABLE"
    assert report.degraded_result.frame_reliability_state == "DEGRADED_CORRECTABLE"

    # All detections on degraded frames must be flagged UNVERIFIED_DEGRADED or SUPPRESSED
    for d in report.degraded_result.detections:
        assert d.reliability_status != "TRUSTED"
        assert d.is_reliable is False

    # Dictionary conversion verification
    r_dict = report.to_dict()
    assert "clear_detection_count" in r_dict
    assert "degraded_detection_count" in r_dict
    assert "summary_narrative" in r_dict
    assert len(r_dict["summary_narrative"]) > 10


# =====================================================================
# 5. VISUALIZATION RENDERING TESTS
# =====================================================================

def test_render_annotated_frame_palette(base_road_scene):
    detector = OpenCVDNNDetector()
    analyzer = ReliableRoadSceneAnalyzer(detector)

    # Create dummy detection result with one of each status
    boxes = [
        DetectionBox(label="car", confidence=0.85, box_xyxy=(50, 50, 150, 150), is_reliable=True, reliability_status="TRUSTED"),
        DetectionBox(label="bus", confidence=0.55, box_xyxy=(160, 50, 260, 150), is_reliable=False, reliability_status="UNVERIFIED_DEGRADED"),
        DetectionBox(label="person", confidence=0.40, box_xyxy=(270, 50, 320, 150), is_reliable=False, reliability_status="SUPPRESSED_OBSTRUCTED")
    ]
    det_result = DetectionResult(detections=boxes, model_name="TestModel", model_available=True)

    annotated = analyzer.render_annotated_frame(base_road_scene, det_result)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == base_road_scene.shape
    assert annotated.dtype == np.uint8
    # Canvas should not be identical to original because bounding boxes and text were drawn
    assert not np.array_equal(annotated, base_road_scene)
