"""
Modular Reliability Decision Engine for PhantomRoad (Stage 2).
Provides rule-grounded, explainable inference that chooses whether to inspect
neighboring frames, select a sharper frame, enhance contrast or brightness,
or request human review based on quantitative optical evidence.
"""

from typing import Tuple, Set, Optional, List
from .state import ReliabilityState, ActionType, ActionAuditStep
from ..metrics.quality_assessor import FrameQualityMetrics


class ReliabilityDecisionEngine:
    """
    Modular, deterministic decision engine for visual evidence reliability.
    Chooses corrective actions dynamically based on measured optical telemetry.
    """

    def __init__(
        self,
        reliability_composite_threshold: float = 0.65,
        blur_laplacian_threshold: float = 85.0,
        low_light_threshold: float = 45.0,
        overexposed_threshold: float = 200.0,
        min_rms_contrast: float = 0.12,
        haze_transmission_threshold: float = 0.55
    ):
        self.reliability_composite_threshold = reliability_composite_threshold
        self.blur_laplacian_threshold = blur_laplacian_threshold
        self.low_light_threshold = low_light_threshold
        self.overexposed_threshold = overexposed_threshold
        self.min_rms_contrast = min_rms_contrast
        self.haze_transmission_threshold = haze_transmission_threshold

    def evaluate_reliability_state(
        self,
        metrics: FrameQualityMetrics
    ) -> Tuple[ReliabilityState, str]:
        """
        Evaluates frame telemetry to determine the operational reliability state
        and a transparent, rule-grounded evidence rationale.

        Returns:
            (ReliabilityState, evidence_rationale)
        """
        # 1. Critical Physical Obstruction check
        if metrics.is_persistent_obstruction:
            return (
                ReliabilityState.CRITICAL_OBSTRUCTION,
                f"Physical lens obstruction persistent across {metrics.temporal_persistence_streak} consecutive frames. "
                f"Spatial edge sparsity ratio: {metrics.obstructed_cell_ratio * 100:.1f}%. "
                "Hardware sensor blockage detected; software enhancement infeasible."
            )

        # Instantaneous total blackout or blinding whiteout (flatness > 0.85 and extreme brightness)
        is_blackout = (metrics.mean_brightness < 8.0 and metrics.global_flatness_score > 0.85)
        is_whiteout = (metrics.mean_brightness > 248.0 and metrics.global_flatness_score > 0.85)
        if is_blackout or is_whiteout:
            condition_type = "blackout" if is_blackout else "whiteout"
            return (
                ReliabilityState.CRITICAL_OBSTRUCTION,
                f"Total sensor {condition_type} "
                f"(Mean luminance: {metrics.mean_brightness:.1f}, Flatness score: {metrics.global_flatness_score:.2f}). "
                "Zero scene structure detectable."
            )

        # 2. Check for actionable optical degradations
        has_blur = metrics.is_blurred
        has_illumination_issue = metrics.is_low_light or metrics.is_overexposed
        has_contrast_issue = metrics.is_low_contrast
        has_haze = metrics.is_hazy_or_foggy
        has_glare = metrics.is_severe_glare

        # Nominal Reliability: no degradations and high composite quality
        if not (has_blur or has_illumination_issue or has_contrast_issue or has_haze or has_glare):
            if metrics.composite_quality_score >= self.reliability_composite_threshold:
                return (
                    ReliabilityState.RELIABLE,
                    f"Visual evidence is reliable (Composite score: {metrics.composite_quality_score:.3f}). "
                    f"Laplacian sharpness={metrics.laplacian_variance:.1f}, RMS contrast={metrics.rms_contrast:.3f}, "
                    f"Transmission={metrics.mean_transmission:.2f}."
                )

        # Degraded Correctable: has measurable degradations that can be addressed by software
        degradations = metrics.measurable_degradations
        if degradations:
            return (
                ReliabilityState.DEGRADED_CORRECTABLE,
                f"Visual evidence degraded by measurable factors: {'; '.join(degradations)}. "
                "Initiating adaptive corrective processing."
            )

        # Borderline quality / High Uncertainty
        return (
            ReliabilityState.DEGRADED_UNCERTAIN,
            f"Marginal optical quality (Composite score: {metrics.composite_quality_score:.3f} < {self.reliability_composite_threshold:.2f}). "
            "Proceeding with adaptive verification."
        )

    def decide_next_action(
        self,
        metrics: FrameQualityMetrics,
        attempted_actions: Set[ActionType],
        has_temporal_neighbors: bool = True,
        previous_step: Optional[ActionAuditStep] = None
    ) -> Tuple[ActionType, str]:
        """
        Dynamically selects the next corrective action based on current telemetry,
        prioritizing root causes, avoiding repeated or failed actions, and explaining
        the exact rationale for the decision.

        Returns:
            (ActionType, decision_reason)
        """
        # Priority 1: Severe Illumination (Low Light or Overexposure)
        # Exposure must be balanced first because photon starvation or sensor saturation
        # suppresses edge amplitudes and invalidates spatial gradient measurements.
        if metrics.is_low_light:
            if ActionType.APPLY_GAMMA_CORRECTION not in attempted_actions:
                return (
                    ActionType.APPLY_GAMMA_CORRECTION,
                    f"Low-light condition detected (Mean brightness: {metrics.mean_brightness:.1f} < {self.low_light_threshold:.1f}, "
                    f"Shadow ratio: {metrics.underexposed_ratio * 100:.1f}%). Applying non-linear gamma expansion (gamma > 1.0)."
                )

        if metrics.is_overexposed:
            if ActionType.APPLY_GAMMA_CORRECTION not in attempted_actions:
                return (
                    ActionType.APPLY_GAMMA_CORRECTION,
                    f"High-exposure saturation detected (Mean brightness: {metrics.mean_brightness:.1f} > {self.overexposed_threshold:.1f}, "
                    f"Clipping ratio: {metrics.overexposed_ratio * 100:.1f}%). Applying gamma compression (gamma < 1.0)."
                )

        # Priority 2: Motion Blur / Defocus
        if metrics.is_blurred:
            if has_temporal_neighbors and ActionType.SELECT_SHARPER_NEIGHBOR not in attempted_actions:
                return (
                    ActionType.SELECT_SHARPER_NEIGHBOR,
                    f"Detected severe blur (Laplacian variance: {metrics.laplacian_variance:.1f} < {self.blur_laplacian_threshold:.1f}). "
                    "Inspecting neighboring frames in temporal buffer to select a sharper acquisition."
                )
            if ActionType.APPLY_UNSHARP_MASK not in attempted_actions:
                return (
                    ActionType.APPLY_UNSHARP_MASK,
                    f"Motion/defocus blur persists (Laplacian: {metrics.laplacian_variance:.1f}) and temporal neighbor search "
                    f"{'exhausted' if ActionType.SELECT_SHARPER_NEIGHBOR in attempted_actions else 'unavailable'}. "
                    "Applying high-frequency spatial unsharp masking."
                )

        # Priority 3: Atmospheric Haze / Fog Attenuation
        if metrics.is_hazy_or_foggy:
            if ActionType.APPLY_DEHAZING not in attempted_actions:
                return (
                    ActionType.APPLY_DEHAZING,
                    f"Atmospheric haze detected (Mean transmission: {metrics.mean_transmission:.2f} < {self.haze_transmission_threshold:.2f}, "
                    f"Dark channel mean: {metrics.dark_channel_mean:.1f}). Applying Dark Channel Prior transmission recovery."
                )

        # Priority 4: Low Scene Contrast
        if metrics.is_low_contrast:
            if ActionType.APPLY_CONTRAST_CLAHE not in attempted_actions:
                return (
                    ActionType.APPLY_CONTRAST_CLAHE,
                    f"Low scene contrast detected (RMS contrast: {metrics.rms_contrast:.3f} < {self.min_rms_contrast:.3f}). "
                    "Applying localized Contrast Limited Adaptive Histogram Equalization (CLAHE)."
                )


        # Priority 5: Secondary / Marginal Quality Improvements
        if metrics.rms_contrast < 0.20 and ActionType.APPLY_CONTRAST_CLAHE not in attempted_actions:
            return (
                ActionType.APPLY_CONTRAST_CLAHE,
                f"Secondary contrast optimization needed (RMS contrast: {metrics.rms_contrast:.3f}). "
                "Applying CLAHE across luminance channel."
            )

        if metrics.laplacian_variance < 150.0 and ActionType.APPLY_UNSHARP_MASK not in attempted_actions:
            return (
                ActionType.APPLY_UNSHARP_MASK,
                f"Secondary sharpness enhancement needed (Laplacian variance: {metrics.laplacian_variance:.1f}). "
                "Applying unsharp masking."
            )

        # Priority 6: All viable actions exhausted -> Escalate to Human Review
        return (
            ActionType.REQUEST_HUMAN_REVIEW,
            "All candidate software enhancement actions have been evaluated or exhausted without resolving degradation. "
            "Escalating visual evidence for human review."
        )


# Global default engine instance for backward-compatible functional interface
_DEFAULT_ENGINE = ReliabilityDecisionEngine()


def determine_initial_state(metrics: FrameQualityMetrics) -> Tuple[ReliabilityState, str]:
    """
    Evaluates frame telemetry using the default ReliabilityDecisionEngine.
    """
    return _DEFAULT_ENGINE.evaluate_reliability_state(metrics)


def select_corrective_action(
    metrics: FrameQualityMetrics,
    attempted_actions: Set[ActionType],
    has_temporal_neighbors: bool = True
) -> ActionType:
    """
    Selects the next corrective action using the default ReliabilityDecisionEngine.
    Returns ActionType for backward compatibility.
    """
    action, _ = _DEFAULT_ENGINE.decide_next_action(
        metrics=metrics,
        attempted_actions=attempted_actions,
        has_temporal_neighbors=has_temporal_neighbors
    )
    return action
