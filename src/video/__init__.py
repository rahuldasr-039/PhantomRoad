"""
Video processing and synthetic scenario generation module for PhantomRoad.
"""

from .frame_extractor import VideoLoader, VideoMetadata, validate_video_path
from .synthetic_generator import generate_synthetic_road_video, AVAILABLE_SCENARIOS

__all__ = [
    "VideoLoader",
    "VideoMetadata",
    "validate_video_path",
    "generate_synthetic_road_video",
    "AVAILABLE_SCENARIOS",
]
