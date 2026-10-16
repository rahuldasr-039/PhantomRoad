"""
Reliability wrapper associating downstream detections with agent reliability states.
Ensures degraded frames flag detections as UNVERIFIED rather than claiming safety.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from .base import BaseRoadSceneDetector, DetectionResult, DetectionBox
from ..agent.state import ReliabilityState, AgentFrameDecision


class ReliableRoadSceneAnalyzer:
    """
    Combines computer vision object detection with the Adaptive Reliability Agent's decisions.
    """

    def __init__(self, detector: BaseRoadSceneDetector):
        self.detector = detector

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        decision: AgentFrameDecision,
        conf_threshold: float = 0.40
    ) -> DetectionResult:
        """
        Runs object detection and decorates results with reliability status.

        Crucial Safety Rule:
        Poor image quality alone does NOT prove that a detected object is absent.
        Detections made on degraded frames are flagged as unverified.
        """
        raw_result = self.detector.detect(frame_bgr, conf_threshold=conf_threshold)

        if not raw_result.model_available:
            return raw_result

        state = decision.final_state
        raw_result.frame_reliability_state = state.value

        if state == ReliabilityState.RELIABLE:
            for d in raw_result.detections:
                d.is_reliable = True
                d.reliability_status = "TRUSTED"
                d.suppression_reason = None
            raw_result.warning_notice = None

        elif state in (ReliabilityState.DEGRADED_CORRECTABLE, ReliabilityState.DEGRADED_UNCERTAIN):
            for d in raw_result.detections:
                d.is_reliable = False
                d.reliability_status = "UNVERIFIED_DEGRADED"
                d.suppression_reason = (
                    f"Frame visual quality compromised ({'; '.join(decision.final_metrics.measurable_degradations)}). "
                    "Detector precision and recall cannot be guaranteed."
                )
            raw_result.warning_notice = (
                "CAUTION: Image degradation detected. Bounding boxes are unverified. "
                "Lack of detections does NOT prove absence of obstacles."
            )

        else:  # CRITICAL_OBSTRUCTION or HUMAN_REVIEW_REQUIRED
            for d in raw_result.detections:
                d.is_reliable = False
                d.reliability_status = "SUPPRESSED_OBSTRUCTED"
                d.suppression_reason = (
                    "Sensor obstruction or unresolvable degradation. Perception stream untrusted."
                )
            raw_result.warning_notice = (
                "CRITICAL WARNING: Sensor visual reliability failed. Downstream detections suppressed. "
                "Immediate human review or sensor cleaning required."
            )

        return raw_result

    def render_annotated_frame(
        self,
        frame_bgr: np.ndarray,
        detection_result: DetectionResult
    ) -> np.ndarray:
        """
        Draws color-coded bounding boxes and reliability HUD badges onto the frame.
        """
        canvas = frame_bgr.copy()

        # Color codes: Green (Trusted), Amber (Unverified), Red (Suppressed)
        status_colors = {
            "TRUSTED": (0, 220, 0),
            "UNVERIFIED_DEGRADED": (0, 180, 255),
            "SUPPRESSED_OBSTRUCTED": (40, 40, 240)
        }

        for det in detection_result.detections:
            x1, y1, x2, y2 = det.box_xyxy
            color = status_colors.get(det.reliability_status, (200, 200, 200))

            # Bounding box
            thick = 2 if det.is_reliable else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

            # Label badge
            tag = f"{det.label.upper()} {det.confidence:.2f} [{det.reliability_status}]"
            font_scale = 0.45
            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(
                canvas,
                tag,
                (x1 + 2, max(th + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return canvas
