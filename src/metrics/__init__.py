"""
Quality metrics module for PhantomRoad.
Includes blur estimation, illumination profiling, visibility/haze indicators,
and multi-indicator obstruction detection.
"""

from .blur import compute_blur_metrics, compute_laplacian_map
from .illumination import compute_illumination_metrics, compute_intensity_histogram
from .visibility import compute_visibility_metrics
from .obstruction import compute_spatial_obstruction, TemporalObstructionFilter
from .quality_assessor import QualityAssessor, FrameQualityMetrics
from .batch_profiler import (
    FrameTelemetryRecord,
    BatchAnalysisSummary,
    profile_single_frame,
    profile_video_batch,
    summarize_records
)

__all__ = [
    "compute_blur_metrics",
    "compute_laplacian_map",
    "compute_illumination_metrics",
    "compute_intensity_histogram",
    "compute_visibility_metrics",
    "compute_spatial_obstruction",
    "TemporalObstructionFilter",
    "QualityAssessor",
    "FrameQualityMetrics",
    "FrameTelemetryRecord",
    "BatchAnalysisSummary",
    "profile_single_frame",
    "profile_video_batch",
    "summarize_records",
]

