"""
Unit tests for the road-scene object detector and reliability wrapper.
"""

import numpy as np
import pytest

from src.detector.base import BaseRoadSceneDetector, DetectionResult, DetectionBox
from src.detector.opencv_dnn import OpenCVDNNDetector
from src.detector.reliability_wrapper import ReliableRoadSceneAnalyzer
from src.agent.state import ReliabilityState, AgentFrameDecision
from src.metrics.quality_assessor import FrameQualityMetrics


class MockDetector(BaseRoadSceneDetector):
    """Mock detector for testing reliability decoration without external weights."""

    def __init__(self, available: bool = True):
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def get_setup_message(self) -> str:
        return "Setup instructions: Place model in models/."

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.40) -> DetectionResult:
        if not self._available:
            return DetectionResult(
                detections=[],
                model_name="MockDetector",
                model_available=False,
                setup_message=self.get_setup_message()
            )
        # Return a simulated test car detection
        return DetectionResult(
            detections=[
                DetectionBox(label="car", confidence=0.88, box_xyxy=(100, 100, 200, 200))
            ],
            model_name="MockDetector",
            model_available=True
        )


def _create_mock_decision(final_state: ReliabilityState) -> AgentFrameDecision:
    dummy_metrics = FrameQualityMetrics(
        frame_index=0,
        timestamp_sec=0.0,
        laplacian_variance=120.0,
        fft_high_freq_ratio=0.3,
        sharpness_score=0.7,
        is_blurred=False,
        mean_brightness=110.0,
        rms_contrast=0.20,
        dynamic_range=120,
        underexposed_ratio=0.0,
        overexposed_ratio=0.0,
        entropy=4.5,
        is_low_light=False,
        is_overexposed=False,
        is_low_contrast=False,
        mean_transmission=0.75,
        dark_channel_mean=30.0,
        haze_level=0.25,
        is_hazy_or_foggy=False,
        glare_ratio=0.0,
        is_severe_glare=False,
        color_cast_score=0.01,
        obstructed_cell_ratio=0.0,
        global_flatness_score=0.0,
        is_instantaneous_obstruction=False,
        is_persistent_obstruction=False,
        temporal_persistence_streak=0,
        is_transient_occlusion=False,
        composite_quality_score=0.75,
        measurable_degradations=["LOW_CONTRAST"] if final_state != ReliabilityState.RELIABLE else []
    )
    return AgentFrameDecision(
        frame_index=0,
        timestamp_sec=0.0,
        initial_state=final_state,
        final_state=final_state,
        initial_metrics=dummy_metrics,
        final_metrics=dummy_metrics,
        attempts_made=0,
        max_attempts=3
    )


def test_detector_offline_honest_message():
    detector = OpenCVDNNDetector(prototxt_path="models/nonexistent.prototxt")
    assert not detector.is_available()
    res = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert not res.model_available
    assert len(res.detections) == 0
    assert "MobileNet-SSD" in res.setup_message


def test_reliability_wrapper_states():
    mock_det = MockDetector(available=True)
    analyzer = ReliableRoadSceneAnalyzer(mock_det)
    dummy_frame = np.zeros((360, 480, 3), dtype=np.uint8)

    # 1. Reliable state -> Trusted detections
    dec_reliable = _create_mock_decision(ReliabilityState.RELIABLE)
    res_rel = analyzer.analyze_frame(dummy_frame, dec_reliable)
    assert res_rel.detections[0].is_reliable
    assert res_rel.detections[0].reliability_status == "TRUSTED"
    assert res_rel.warning_notice is None

    # 2. Degraded state -> Unverified detections
    dec_degraded = _create_mock_decision(ReliabilityState.DEGRADED_UNCERTAIN)
    res_deg = analyzer.analyze_frame(dummy_frame, dec_degraded)
    assert not res_deg.detections[0].is_reliable
    assert res_deg.detections[0].reliability_status == "UNVERIFIED_DEGRADED"
    assert "CAUTION" in res_deg.warning_notice

    # 3. Obstruction / Human review state -> Suppressed detections
    dec_obs = _create_mock_decision(ReliabilityState.HUMAN_REVIEW_REQUIRED)
    res_obs = analyzer.analyze_frame(dummy_frame, dec_obs)
    assert not res_obs.detections[0].is_reliable
    assert res_obs.detections[0].reliability_status == "SUPPRESSED_OBSTRUCTED"
    assert "CRITICAL" in res_obs.warning_notice


def test_annotated_frame_render():
    mock_det = MockDetector(available=True)
    analyzer = ReliableRoadSceneAnalyzer(mock_det)
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    dec = _create_mock_decision(ReliabilityState.RELIABLE)
    res = analyzer.analyze_frame(frame, dec)

    annotated = analyzer.render_annotated_frame(frame, res)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
