"""
Verification engine for before-and-after enhancement evaluation.
Ensures actions are accepted ONLY when quantitative evidence proves image quality improved.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
from ..metrics.quality_assessor import FrameQualityMetrics


@dataclass
class EnhancementResult:
    """
    Detailed audit outcome of a corrective action.
    """
    action_name: str
    accepted: bool
    quality_delta: float
    before_score: float
    after_score: float
    reason: str
    metric_deltas: Dict[str, float]


def evaluate_enhancement_improvement(
    before_metrics: FrameQualityMetrics,
    after_metrics: FrameQualityMetrics,
    action_name: str,
    min_quality_delta: float = 0.02
) -> EnhancementResult:
    """
    Quantitatively compares metrics before and after an enhancement action.

    Args:
        before_metrics: Quality metrics before action.
        after_metrics: Quality metrics after action.
        action_name: Name of action taken (e.g. 'APPLY_CLAHE', 'APPLY_GAMMA').
        min_quality_delta: Minimum positive gain in composite score or targeted metric.

    Returns:
        EnhancementResult with acceptance decision, delta score, and audit reason.
    """
    q_before = before_metrics.composite_quality_score
    q_after = after_metrics.composite_quality_score
    overall_delta = round(q_after - q_before, 4)

    deltas = {
        "sharpness_delta": round(after_metrics.laplacian_variance - before_metrics.laplacian_variance, 2),
        "contrast_delta": round(after_metrics.rms_contrast - before_metrics.rms_contrast, 4),
        "brightness_delta": round(after_metrics.mean_brightness - before_metrics.mean_brightness, 2),
        "transmission_delta": round(after_metrics.mean_transmission - before_metrics.mean_transmission, 4),
        "overall_quality_delta": overall_delta
    }

    accepted = False
    reason = ""

    if "CLAHE" in action_name or "CONTRAST" in action_name:
        # Expect contrast increase and reasonable dynamic range expansion
        contrast_gain = deltas["contrast_delta"]
        if contrast_gain >= 0.015 and after_metrics.overexposed_ratio <= before_metrics.overexposed_ratio + 0.10:
            accepted = True
            reason = (
                f"Accepted: RMS contrast improved from {before_metrics.rms_contrast:.3f} "
                f"to {after_metrics.rms_contrast:.3f} (+{contrast_gain:.3f}) with composite gain {overall_delta:+.3f}."
            )
        else:
            accepted = False
            reason = (
                f"Rejected: Insufficient contrast gain ({contrast_gain:+.3f}) or severe highlight clipping. "
                "Rolled back to original."
            )

    elif "GAMMA" in action_name or "BRIGHTNESS" in action_name:
        # For low light: expect brightness increase and shadow reduction
        if before_metrics.is_low_light:
            b_gain = deltas["brightness_delta"]
            under_reduction = before_metrics.underexposed_ratio - after_metrics.underexposed_ratio
            if b_gain >= 10.0 or under_reduction >= 0.08:
                accepted = True
                reason = (
                    f"Accepted: Mean luminance raised from {before_metrics.mean_brightness:.1f} to "
                    f"{after_metrics.mean_brightness:.1f} (+{b_gain:.1f}), reducing shadow clipping by {under_reduction*100:.1f}%."
                )
            else:
                accepted = False
                reason = f"Rejected: Luminance gain insufficient ({b_gain:+.1f}). Rolled back."
        elif before_metrics.is_overexposed:
            over_reduction = before_metrics.overexposed_ratio - after_metrics.overexposed_ratio
            if over_reduction >= 0.05:
                accepted = True
                reason = (
                    f"Accepted: High-exposure clipping reduced by {over_reduction*100:.1f}% "
                    f"(from {before_metrics.overexposed_ratio*100:.1f}% to {after_metrics.overexposed_ratio*100:.1f}%)."
                )
            else:
                accepted = False
                reason = "Rejected: Overexposure mitigation was ineffective. Rolled back."
        else:
            accepted = overall_delta >= min_quality_delta
            reason = f"{'Accepted' if accepted else 'Rejected'}: Overall composite quality delta {overall_delta:+.3f}."

    elif "SHARP" in action_name:
        sharp_gain = deltas["sharpness_delta"]
        if sharp_gain >= 15.0 and overall_delta >= -0.01:
            accepted = True
            reason = (
                f"Accepted: Laplacian variance improved from {before_metrics.laplacian_variance:.1f} "
                f"to {after_metrics.laplacian_variance:.1f} (+{sharp_gain:.1f})."
            )
        else:
            accepted = False
            reason = (
                f"Rejected: Sharpness gain too low ({sharp_gain:+.1f}) or excessive noise amplified. "
                "Rolled back."
            )

    elif "DEHAZE" in action_name:
        trans_gain = deltas["transmission_delta"]
        dark_reduction = before_metrics.dark_channel_mean - after_metrics.dark_channel_mean
        if trans_gain >= 0.05 or dark_reduction >= 15.0 or deltas["contrast_delta"] >= 0.02:
            accepted = True
            reason = (
                f"Accepted: Atmospheric transmission increased by {trans_gain:+.3f} and "
                f"dark channel haze reduced by {dark_reduction:.1f}. Contrast gain: {deltas['contrast_delta']:+.3f}."
            )
        else:
            accepted = False
            reason = (
                f"Rejected: Dehazing did not meaningfully improve transmission ({trans_gain:+.3f}). "
                "Rolled back."
            )

    elif "NEIGHBOR" in action_name:
        sharp_ratio = (after_metrics.laplacian_variance / (before_metrics.laplacian_variance + 1e-5))
        if sharp_ratio >= 1.20 or overall_delta >= 0.05:
            accepted = True
            reason = (
                f"Accepted: Neighbor frame provides {sharp_ratio:.2f}x higher Laplacian sharpness "
                f"({after_metrics.laplacian_variance:.1f} vs {before_metrics.laplacian_variance:.1f})."
            )
        else:
            accepted = False
            reason = (
                f"Rejected: Neighbor frame does not offer significant sharpness benefit ({sharp_ratio:.2f}x). "
                "Retained target frame."
            )

    else:
        # Generic fallback comparison
        accepted = bool(overall_delta >= min_quality_delta)
        reason = f"{'Accepted' if accepted else 'Rejected'}: Quality delta is {overall_delta:+.3f}."

    return EnhancementResult(
        action_name=action_name,
        accepted=accepted,
        quality_delta=overall_delta,
        before_score=q_before,
        after_score=q_after,
        reason=reason,
        metric_deltas=deltas
    )
