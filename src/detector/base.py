"""
Base classes and interfaces for downstream road-scene object detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class DetectionBox:
    """
    Individual detected road object with reliability annotations.
    """
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    is_reliable: bool = True
    reliability_status: str = "TRUSTED"
    suppression_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box_xyxy": list(self.box_xyxy),
            "is_reliable": self.is_reliable,
            "reliability_status": self.reliability_status,
            "suppression_reason": self.suppression_reason
        }


@dataclass
class DetectionResult:
    """
    Complete inference result for an analyzed frame.
    """
    detections: List[DetectionBox] = field(default_factory=list)
    model_name: str = "Unknown"
    model_available: bool = False
    setup_message: Optional[str] = None
    frame_reliability_state: str = "RELIABLE"
    warning_notice: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_available": self.model_available,
            "setup_message": self.setup_message,
            "frame_reliability_state": self.frame_reliability_state,
            "warning_notice": self.warning_notice,
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections]
        }


class BaseRoadSceneDetector(ABC):
    """
    Abstract interface for pluggable road-scene detectors.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the required model weights and dependencies are installed."""
        pass

    @abstractmethod
    def get_setup_message(self) -> str:
        """Returns setup instructions if model is unavailable."""
        pass

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.40) -> DetectionResult:
        """Performs object detection on the input frame."""
        pass
