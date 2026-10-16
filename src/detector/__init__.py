"""
Downstream road-scene object detector and reliability integration module.
"""

from .base import DetectionBox, DetectionResult, BaseRoadSceneDetector
from .opencv_dnn import OpenCVDNNDetector
from .reliability_wrapper import ReliableRoadSceneAnalyzer
from .comparator import RoadScenePerceptionComparator, PerceptionComparisonReport

__all__ = [
    "DetectionBox",
    "DetectionResult",
    "BaseRoadSceneDetector",
    "OpenCVDNNDetector",
    "ReliableRoadSceneAnalyzer",
    "RoadScenePerceptionComparator",
    "PerceptionComparisonReport",
]

