"""
Road Scene Perception Comparator for Stage 3.
Compares object detector inference across clear, degraded, and adaptive enhanced frames.
Ensures detections are truthfully extracted without fabrication and annotated
with optical reliability status.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from .base import BaseRoadSceneDetector, DetectionResult, DetectionBox
from .reliability_wrapper import ReliableRoadSceneAnalyzer
from ..agent.state import ReliabilityState, AgentFrameDecision
from ..metrics.quality_assessor import QualityAssessor, FrameQualityMetrics


@dataclass
class PerceptionComparisonReport:
    """
    Structured comparative evaluation of object detection across varying visual conditions.
    """
    clear_result: DetectionResult
    degraded_result: DetectionResult
    enhanced_result: Optional[DetectionResult] = None
    clear_detection_count: int = 0
    degraded_detection_count: int = 0
    enhanced_detection_count: int = 0
    clear_confidence_mean: float = 0.0
    degraded_confidence_mean: float = 0.0
    enhanced_confidence_mean: float = 0.0
    confidence_delta: float = 0.0
    unverified_count: int = 0
    suppressed_count: int = 0
    trusted_count: int = 0
    summary_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clear_detection_count": self.clear_detection_count,
            "degraded_detection_count": self.degraded_detection_count,
            "enhanced_detection_count": self.enhanced_detection_count,
            "clear_confidence_mean": round(self.clear_confidence_mean, 3),
            "degraded_confidence_mean": round(self.degraded_confidence_mean, 3),
            "enhanced_confidence_mean": round(self.enhanced_confidence_mean, 3),
            "confidence_delta": round(self.confidence_delta, 3),
            "unverified_count": self.unverified_count,
            "suppressed_count": self.suppressed_count,
            "trusted_count": self.trusted_count,
            "summary_narrative": self.summary_narrative,
            "clear_detections": [d.to_dict() for d in self.clear_result.detections],
            "degraded_detections": [d.to_dict() for d in self.degraded_result.detections],
            "enhanced_detections": [d.to_dict() for d in self.enhanced_result.detections] if self.enhanced_result else []
        }


class RoadScenePerceptionComparator:
    """
    Performs comparative inference across visual conditions, quantifying how
    optical degradation affects detection confidence and bounding box stability.
    """

    def __init__(self, detector: BaseRoadSceneDetector):
        self.detector = detector
        self.analyzer = ReliableRoadSceneAnalyzer(detector)

    def compare_frames(
        self,
        clear_frame: np.ndarray,
        degraded_frame: np.ndarray,
        enhanced_frame: Optional[np.ndarray] = None,
        clear_decision: Optional[AgentFrameDecision] = None,
        degraded_decision: Optional[AgentFrameDecision] = None,
        enhanced_decision: Optional[AgentFrameDecision] = None,
        conf_threshold: float = 0.35
    ) -> PerceptionComparisonReport:
        """
        Runs object detection inference across clear, degraded, and enhanced frames.
        Never fabricates detections; all results are derived from actual model forward passes.
        """
        # 1. Clear frame analysis
        if clear_decision is not None:
            clear_res = self.analyzer.analyze_frame(clear_frame, clear_decision, conf_threshold=conf_threshold)
        else:
            raw_clear = self.detector.detect(clear_frame, conf_threshold=conf_threshold)
            clear_res = raw_clear

        # 2. Degraded frame analysis
        if degraded_decision is not None:
            degraded_res = self.analyzer.analyze_frame(degraded_frame, degraded_decision, conf_threshold=conf_threshold)
        else:
            # Default to degraded state if decision not supplied
            raw_deg = self.detector.detect(degraded_frame, conf_threshold=conf_threshold)
            for d in raw_deg.detections:
                d.is_reliable = False
                d.reliability_status = "UNVERIFIED_DEGRADED"
                d.suppression_reason = "Image quality degraded; detector precision cannot be guaranteed."
            raw_deg.warning_notice = "CAUTION: Detections on degraded frame marked UNVERIFIED."
            raw_deg.frame_reliability_state = "DEGRADED_CORRECTABLE"
            degraded_res = raw_deg

        # 3. Enhanced frame analysis (if supplied)
        enhanced_res: Optional[DetectionResult] = None
        if enhanced_frame is not None:
            if enhanced_decision is not None:
                enhanced_res = self.analyzer.analyze_frame(enhanced_frame, enhanced_decision, conf_threshold=conf_threshold)
            else:
                enhanced_res = self.detector.detect(enhanced_frame, conf_threshold=conf_threshold)

        # 4. Compute comparative telemetry
        clear_confs = [d.confidence for d in clear_res.detections]
        deg_confs = [d.confidence for d in degraded_res.detections]
        enh_confs = [d.confidence for d in enhanced_res.detections] if enhanced_res else []

        clear_mean = float(np.mean(clear_confs)) if clear_confs else 0.0
        deg_mean = float(np.mean(deg_confs)) if deg_confs else 0.0
        enh_mean = float(np.mean(enh_confs)) if enh_confs else 0.0

        trusted_cnt = sum(1 for d in degraded_res.detections if d.reliability_status == "TRUSTED")
        unverified_cnt = sum(1 for d in degraded_res.detections if d.reliability_status == "UNVERIFIED_DEGRADED")
        suppressed_cnt = sum(1 for d in degraded_res.detections if d.reliability_status == "SUPPRESSED_OBSTRUCTED")

        # 5. Build summary narrative
        conf_drop = round(clear_mean - deg_mean, 3)
        if len(clear_res.detections) > len(degraded_res.detections):
            narrative = (
                f"Optical degradation caused {len(clear_res.detections) - len(degraded_res.detections)} "
                f"false negative(s) (missed obstacles). Mean detection confidence dropped from {clear_mean:.2f} to {deg_mean:.2f}."
            )
        elif len(degraded_res.detections) > 0 and unverified_cnt > 0:
            narrative = (
                f"Detector returned {len(degraded_res.detections)} object candidate(s) under degraded visibility. "
                f"All {unverified_cnt} bounding boxes were tagged UNVERIFIED_DEGRADED to alert downstream path planners."
            )
        else:
            narrative = (
                f"Detection count: {len(clear_res.detections)} on clear vs {len(degraded_res.detections)} on degraded. "
                f"Mean confidence delta: {conf_drop:+.2f}."
            )

        return PerceptionComparisonReport(
            clear_result=clear_res,
            degraded_result=degraded_res,
            enhanced_result=enhanced_res,
            clear_detection_count=len(clear_res.detections),
            degraded_detection_count=len(degraded_res.detections),
            enhanced_detection_count=len(enhanced_res.detections) if enhanced_res else 0,
            clear_confidence_mean=clear_mean,
            degraded_confidence_mean=deg_mean,
            enhanced_confidence_mean=enh_mean,
            confidence_delta=conf_drop,
            unverified_count=unverified_cnt,
            suppressed_count=suppressed_cnt,
            trusted_count=trusted_cnt,
            summary_narrative=narrative
        )
