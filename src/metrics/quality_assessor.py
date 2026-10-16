"""
Quality Assessor aggregating all low-level metrics into a structured telemetry profile.
Strictly separates measurable optical degradation from inferred atmospheric causes.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import numpy as np

from .blur import compute_blur_metrics
from .illumination import compute_illumination_metrics
from .visibility import compute_visibility_metrics
from .obstruction import compute_spatial_obstruction, TemporalObstructionFilter


@dataclass
class FrameQualityMetrics:
    """
    Structured telemetry profile for an evaluated video frame.
    """
    frame_index: int
    timestamp_sec: float

    # 1. Sharpness & Blur
    laplacian_variance: float
    fft_high_freq_ratio: float
    sharpness_score: float
    is_blurred: bool

    # 2. Illumination & Contrast
    mean_brightness: float
    rms_contrast: float
    dynamic_range: int
    underexposed_ratio: float
    overexposed_ratio: float
    entropy: float
    is_low_light: bool
    is_overexposed: bool
    is_low_contrast: bool

    # 3. Visibility & Atmospheric Attenuation
    mean_transmission: float
    dark_channel_mean: float
    haze_level: float
    is_hazy_or_foggy: bool
    glare_ratio: float
    is_severe_glare: bool
    color_cast_score: float

    # 4. Obstruction
    obstructed_cell_ratio: float
    global_flatness_score: float
    is_instantaneous_obstruction: bool
    is_persistent_obstruction: bool
    temporal_persistence_streak: int
    is_transient_occlusion: bool

    # Composite Quality Index (0.0 to 1.0)
    composite_quality_score: float

    # Documented Degradations vs Inferred Causes
    measurable_degradations: List[str] = field(default_factory=list)
    inferred_causes: List[str] = field(default_factory=list)
    evidence_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts telemetry to a JSON-serializable dictionary."""
        d = asdict(self)
        return d


class QualityAssessor:
    """
    Master assessor that computes all quality indicators for a frame and
    maintains temporal obstruction state.
    """

    def __init__(
        self,
        blur_threshold: float = 85.0,
        low_light_threshold: float = 45.0,
        overexposed_threshold: float = 200.0,
        min_rms_contrast: float = 0.12,
        min_transmission: float = 0.38,
        temporal_window: int = 5,
        min_persistence: int = 3
    ):
        self.blur_threshold = blur_threshold
        self.low_light_threshold = low_light_threshold
        self.overexposed_threshold = overexposed_threshold
        self.min_rms_contrast = min_rms_contrast
        self.min_transmission = min_transmission
        self.obstruction_tracker = TemporalObstructionFilter(
            history_len=temporal_window,
            min_persistence_frames=min_persistence
        )

    def reset_temporal_state(self) -> None:
        """Resets temporal tracking state for a new video clip."""
        self.obstruction_tracker.reset()

    def assess_frame(
        self,
        image_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0
    ) -> FrameQualityMetrics:
        """
        Runs comprehensive analysis on a frame and returns structured telemetry.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image cannot be None or empty.")

        # 1. Blur
        blur_m = compute_blur_metrics(image_bgr, blur_threshold=self.blur_threshold)

        # 2. Illumination & Contrast
        illum_m = compute_illumination_metrics(
            image_bgr,
            low_light_threshold=self.low_light_threshold,
            overexposed_threshold=self.overexposed_threshold,
            min_rms_contrast=self.min_rms_contrast
        )

        # 3. Visibility & Glare
        vis_m = compute_visibility_metrics(
            image_bgr,
            min_acceptable_transmission=self.min_transmission
        )

        # 4. Obstruction & Temporal Filter
        spatial_obs = compute_spatial_obstruction(image_bgr)
        temporal_obs = self.obstruction_tracker.update(spatial_obs)

        # 5. Measurable Physical Degradations
        degradations = []
        if blur_m["is_blurred"]:
            degradations.append(f"LOW_SHARPNESS (Laplacian var {blur_m['laplacian_variance']} < {self.blur_threshold})")
        if illum_m["is_low_light"]:
            degradations.append(f"LOW_ILLUMINATION (Mean luminance {illum_m['mean_brightness']} < {self.low_light_threshold})")
        if illum_m["is_overexposed"]:
            degradations.append(f"HIGH_EXPOSURE_CLIPPING (Mean luminance {illum_m['mean_brightness']} > {self.overexposed_threshold})")
        if illum_m["is_low_contrast"]:
            degradations.append(f"LOW_RMS_CONTRAST (RMS {illum_m['rms_contrast']} < {self.min_rms_contrast})")
        if vis_m["is_hazy_or_foggy"]:
            degradations.append(f"ATMOSPHERIC_ATTENUATION (DCP transmission {vis_m['mean_transmission']} < {self.min_transmission})")
        if vis_m["is_severe_glare"]:
            degradations.append(f"SPECULAR_GLARE_SATURATION (Glare ratio {vis_m['glare_ratio']*100:.1f}%)")
        if temporal_obs["is_persistent_obstruction"]:
            degradations.append(f"PERSISTENT_LENS_OBSTRUCTION (Blocked ratio {spatial_obs['obstructed_cell_ratio']*100:.1f}%, streak={temporal_obs['persistence_streak']})")
        elif temporal_obs["is_transient"]:
            degradations.append("TRANSIENT_SURFACE_OCCLUSION (Temporary blockage detected, tracking persistence)")

        # 6. Inferred Atmospheric Causes (with explicit uncertainty notation)
        inferred = []
        if vis_m["is_hazy_or_foggy"] and illum_m["is_low_contrast"]:
            inferred.append("Hypothesis: Atmospheric aerosol scattering (fog / mist / heavy spray).")
        if illum_m["is_low_light"] and not vis_m["is_severe_glare"]:
            inferred.append("Hypothesis: Nighttime or unlit tunnel road environment.")
        if vis_m["is_severe_glare"]:
            inferred.append("Hypothesis: Direct oncoming headlights, low-angle sun, or wet surface reflection.")
        if blur_m["is_blurred"] and illum_m["mean_brightness"] > 60:
            inferred.append("Hypothesis: High camera velocity motion blur or lens vibration.")
        if temporal_obs["is_persistent_obstruction"]:
            inferred.append("Hypothesis: Foreign matter on lens casing (mud deposit, opaque grime, or hardware failure).")

        # 7. Composite Quality Score [0.0 - 1.0]
        # Weighted combination of normalized factors
        sharpness_w = blur_m["sharpness_score"] * 0.25
        contrast_w = np.clip(illum_m["rms_contrast"] / 0.30, 0.0, 1.0) * 0.25
        transmission_w = vis_m["mean_transmission"] * 0.25
        illum_w = (1.0 - abs(illum_m["mean_brightness"] - 128.0) / 128.0) * 0.15
        obstruction_penalty = (1.0 - spatial_obs["obstructed_cell_ratio"]) * 0.10

        composite_score = float(np.clip(
            sharpness_w + contrast_w + transmission_w + illum_w + obstruction_penalty,
            0.0, 1.0
        ))

        # Severe obstruction or complete black/white collapses composite score
        if temporal_obs["is_persistent_obstruction"] or spatial_obs["global_flatness_score"] > 0.85:
            composite_score = min(composite_score, 0.20)

        # 8. Evidence Summary
        if not degradations:
            evidence_summary = (
                f"Nominal visual quality: Sharpness={blur_m['laplacian_variance']:.1f}, "
                f"Luminance={illum_m['mean_brightness']:.1f}, RMS Contrast={illum_m['rms_contrast']:.3f}, "
                f"Transmission={vis_m['mean_transmission']:.2f}. No optical degradation flagged."
            )
        else:
            evidence_summary = f"Flagged {len(degradations)} degradation indicators: {'; '.join(degradations)}."

        return FrameQualityMetrics(
            frame_index=frame_index,
            timestamp_sec=round(timestamp_sec, 3),
            laplacian_variance=blur_m["laplacian_variance"],
            fft_high_freq_ratio=blur_m["fft_high_freq_ratio"],
            sharpness_score=blur_m["sharpness_score"],
            is_blurred=blur_m["is_blurred"],
            mean_brightness=illum_m["mean_brightness"],
            rms_contrast=illum_m["rms_contrast"],
            dynamic_range=illum_m["dynamic_range"],
            underexposed_ratio=illum_m["underexposed_ratio"],
            overexposed_ratio=illum_m["overexposed_ratio"],
            entropy=illum_m["entropy"],
            is_low_light=illum_m["is_low_light"],
            is_overexposed=illum_m["is_overexposed"],
            is_low_contrast=illum_m["is_low_contrast"],
            mean_transmission=vis_m["mean_transmission"],
            dark_channel_mean=vis_m["dark_channel_mean"],
            haze_level=vis_m["haze_level"],
            is_hazy_or_foggy=vis_m["is_hazy_or_foggy"],
            glare_ratio=vis_m["glare_ratio"],
            is_severe_glare=vis_m["is_severe_glare"],
            color_cast_score=vis_m["color_cast_score"],
            obstructed_cell_ratio=spatial_obs["obstructed_cell_ratio"],
            global_flatness_score=spatial_obs["global_flatness_score"],
            is_instantaneous_obstruction=spatial_obs["is_instantaneous_obstruction"],
            is_persistent_obstruction=temporal_obs["is_persistent_obstruction"],
            temporal_persistence_streak=temporal_obs["persistence_streak"],
            is_transient_occlusion=temporal_obs["is_transient"],
            composite_quality_score=round(composite_score, 4),
            measurable_degradations=degradations,
            inferred_causes=inferred,
            evidence_summary=evidence_summary
        )
