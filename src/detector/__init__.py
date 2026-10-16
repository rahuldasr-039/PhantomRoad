"""
Downstream road-scene object detector and reliability integration module.
"""

from .base import DetectionBox, DetectionResult, BaseRoadSceneDetector
from .opencv_dnn import OpenCVDNNDetector, ROAD_CLASSES
from .reliability_wrapper import ReliableRoadSceneAnalyzer
from .comparator import RoadScenePerceptionComparator, PerceptionComparisonReport

__all__ = [
    "DetectionBox",
    "DetectionResult",
    "BaseRoadSceneDetector",
    "OpenCVDNNDetector",
    "ROAD_CLASSES",
    "ReliableRoadSceneAnalyzer",
    "RoadScenePerceptionComparator",
    "PerceptionComparisonReport",
]


