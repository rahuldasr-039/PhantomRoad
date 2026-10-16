"""
Enhancement algorithms and before/after verification for PhantomRoad.
"""

from .contrast_clahe import enhance_contrast_clahe
from .illumination_gamma import correct_illumination_gamma
from .sharpness_unsharp import enhance_sharpness_unsharp_mask
from .dehaze_dcp import dehaze_dark_channel
from .denoise import denoise_edge_preserving
from .verification import evaluate_enhancement_improvement, EnhancementResult

__all__ = [
    "enhance_contrast_clahe",
    "correct_illumination_gamma",
    "enhance_sharpness_unsharp_mask",
    "dehaze_dark_channel",
    "denoise_edge_preserving",
    "evaluate_enhancement_improvement",
    "EnhancementResult",
]
