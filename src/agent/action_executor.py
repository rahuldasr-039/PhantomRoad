"""
Adaptive Action Executor orchestrating the feedback and re-evaluation loop (Stage 2).
Executes actions, recalculates quality metrics after every attempt, compares against
both preceding and original measurements, rejects degradations with rollback,
and manages human escalation.
"""

from typing import Optional, List, Set, Tuple, Dict, Any
import cv2
import numpy as np

from .state import (
    ReliabilityState,
    ActionType,
    StopCondition,
    ActionOutcome,
    ActionAuditStep,
    AgentFrameDecision
)
from .decision_engine import (
    ReliabilityDecisionEngine,
    determine_initial_state,
    select_corrective_action
)
from ..metrics.quality_assessor import QualityAssessor, FrameQualityMetrics
from ..enhancement.contrast_clahe import enhance_contrast_clahe
from ..enhancement.illumination_gamma import correct_illumination_gamma
from ..enhancement.sharpness_unsharp import enhance_sharpness_unsharp_mask
from ..enhancement.dehaze_dcp import dehaze_dark_channel
from ..enhancement.verification import evaluate_enhancement_improvement
from ..video.frame_extractor import VideoLoader


class AdaptiveReliabilityAgent:
    """
    Modular reliability decision engine and adaptive action controller.
    Executes a verifiable closed-loop cycle:
    1. Assess initial quality & reliability state.
    2. Select appropriate corrective action (neighbor inspection, contrast, brightness, dehazing, or human review).
    3. Execute action on candidate frame.
    4. Recalculate metrics on candidate frame.
    5. Compare against preceding metrics AND original unenhanced frame.
    6. Accept only if verified improved; otherwise rollback.
    7. Log full trace with reason, before/after measurements, outcome, and stop condition.
    8. Stop on success or when max attempts exhausted, escalating to human review if unresolved.
    """

    def __init__(
        self,
        assessor: Optional[QualityAssessor] = None,
        decision_engine: Optional[ReliabilityDecisionEngine] = None,
        max_attempts: int = 3
    ):
        self.assessor = assessor or QualityAssessor()
        self.decision_engine = decision_engine or ReliabilityDecisionEngine()
        self.max_attempts = max_attempts

    def reset_temporal_state(self) -> None:
        """Resets assessor temporal tracking for new clip."""
        self.assessor.reset_temporal_state()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
        video_loader: Optional[VideoLoader] = None,
        neighbor_frames: Optional[List[Tuple[int, np.ndarray]]] = None
    ) -> AgentFrameDecision:
        """
        Executes the adaptive reliability loop for a single video frame.

        Args:
            frame_bgr: Input camera frame in BGR format.
            frame_index: Sequential index of the frame.
            timestamp_sec: Timestamp in seconds.
            video_loader: Optional VideoLoader for temporal neighborhood extraction.
            neighbor_frames: Optional list of (index, frame_bgr) tuples for testing or buffering.

        Returns:
            AgentFrameDecision with complete audit trail, action log, and original comparison.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr cannot be empty.")

        # 1. Calculate Initial Quality Indicators
        original_frame = frame_bgr.copy()
        initial_metrics = self.assessor.assess_frame(
            original_frame,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec
        )

        # 2. Determine Initial Reliability State & Evidence
        initial_state, initial_rationale = self.decision_engine.evaluate_reliability_state(initial_metrics)

        # 3. If already RELIABLE, stop immediately (no corrective action needed)
        if initial_state == ReliabilityState.RELIABLE:
            comparison = self._compute_comparison_vs_original(initial_metrics, initial_metrics, initial_state, initial_state)
            return AgentFrameDecision(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                initial_state=initial_state,
                final_state=ReliabilityState.RELIABLE,
                initial_metrics=initial_metrics,
                final_metrics=initial_metrics,
                attempts_made=0,
                max_attempts=self.max_attempts,
                audit_trace=[],
                human_review_required=False,
                escalation_reason=None,
                final_frame_bgr=original_frame,
                was_enhanced=False,
                stop_condition=StopCondition.STOP_ALREADY_RELIABLE.value,
                comparison_vs_original=comparison
            )

        # 4. If CRITICAL OBSTRUCTION, escalate directly without futile retries
        if initial_state == ReliabilityState.CRITICAL_OBSTRUCTION:
            audit_step = ActionAuditStep(
                attempt_index=1,
                action=ActionType.MARK_OBSTRUCTED,
                accepted=True,
                quality_delta=0.0,
                before_composite=initial_metrics.composite_quality_score,
                after_composite=initial_metrics.composite_quality_score,
                evidence_reason="Foreign matter physically obstructing camera lens; software correction infeasible.",
                metric_deltas={},
                reason=initial_rationale,
                outcome=ActionOutcome.CRITICAL_OBSTRUCTION_DETECTED.value,
                stop_condition=StopCondition.STOP_CRITICAL_OBSTRUCTION.value,
                before_metrics=initial_metrics.to_dict(),
                after_metrics=initial_metrics.to_dict(),
                original_metrics=initial_metrics.to_dict(),
                quality_delta_vs_original=0.0
            )
            comparison = self._compute_comparison_vs_original(
                initial_metrics, initial_metrics, initial_state, ReliabilityState.HUMAN_REVIEW_REQUIRED
            )
            return AgentFrameDecision(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                initial_state=initial_state,
                final_state=ReliabilityState.HUMAN_REVIEW_REQUIRED,
                initial_metrics=initial_metrics,
                final_metrics=initial_metrics,
                attempts_made=1,
                max_attempts=self.max_attempts,
                audit_trace=[audit_step],
                human_review_required=True,
                escalation_reason="Critical lens obstruction detected. Manual lens cleaning or sensor replacement required.",
                final_frame_bgr=original_frame,
                was_enhanced=False,
                stop_condition=StopCondition.STOP_CRITICAL_OBSTRUCTION.value,
                comparison_vs_original=comparison
            )

        # 5. Adaptive Action Loop
        curr_frame = original_frame.copy()
        curr_metrics = initial_metrics
        attempted_actions: Set[ActionType] = set()
        audit_trace: List[ActionAuditStep] = []
        was_enhanced = False
        final_stop_condition = StopCondition.CONTINUE.value
        last_step: Optional[ActionAuditStep] = None

        has_temporal = (video_loader is not None) or (neighbor_frames is not None and len(neighbor_frames) > 0)

        for attempt in range(1, self.max_attempts + 1):
            action, action_reason = self.decision_engine.decide_next_action(
                metrics=curr_metrics,
                attempted_actions=attempted_actions,
                has_temporal_neighbors=has_temporal,
                previous_step=last_step
            )

            if action == ActionType.REQUEST_HUMAN_REVIEW:
                final_stop_condition = StopCondition.STOP_NO_VIABLE_ACTIONS.value
                break

            attempted_actions.add(action)
            candidate_frame = curr_frame.copy()
            exec_note = ""

            # Execute Selected Action
            if action == ActionType.SELECT_SHARPER_NEIGHBOR:
                candidates: List[Tuple[int, np.ndarray]] = []
                if neighbor_frames is not None:
                    candidates = neighbor_frames
                elif video_loader is not None:
                    candidates = video_loader.get_temporal_neighborhood(frame_index, radius=2)

                best_neighbor = None
                best_neighbor_idx = -1
                best_lap = curr_metrics.laplacian_variance
                best_composite = curr_metrics.composite_quality_score

                for n_idx, n_frame in candidates:
                    if n_frame is None or n_frame.size == 0:
                        continue
                    n_gray = cv2.cvtColor(n_frame, cv2.COLOR_BGR2GRAY)
                    lap = float(cv2.Laplacian(n_gray, cv2.CV_64F).var())
                    # Neighbor must be significantly sharper (at least 15% gain)
                    if lap > best_lap * 1.15:
                        n_metrics = self.assessor.assess_frame(n_frame, frame_index=n_idx, timestamp_sec=timestamp_sec)
                        if n_metrics.composite_quality_score >= best_composite:
                            best_lap = lap
                            best_composite = n_metrics.composite_quality_score
                            best_neighbor = n_frame
                            best_neighbor_idx = n_idx

                if best_neighbor is not None:
                    candidate_frame = best_neighbor.copy()
                    exec_note = (
                        f"Found sharper neighbor frame #{best_neighbor_idx} "
                        f"(Laplacian: {best_lap:.1f} vs current: {curr_metrics.laplacian_variance:.1f}, "
                        f"+{((best_lap / max(1e-3, curr_metrics.laplacian_variance)) - 1.0) * 100:.1f}% gain)."
                    )
                else:
                    # No neighbor provided superior sharpness
                    candidate_frame = curr_frame.copy()
                    exec_note = "Inspected neighboring frames in temporal buffer; none offered the required 15% sharpness gain."

            elif action == ActionType.APPLY_CONTRAST_CLAHE:
                candidate_frame = enhance_contrast_clahe(curr_frame, clip_limit=2.5)
                exec_note = "Applied Contrast Limited Adaptive Histogram Equalization (CLAHE, clip_limit=2.5) across LAB luminance."

            elif action == ActionType.APPLY_GAMMA_CORRECTION:
                gamma = 2.0 if curr_metrics.is_low_light else 0.65
                candidate_frame = correct_illumination_gamma(curr_frame, gamma=gamma)
                mode = "expansion (low light)" if curr_metrics.is_low_light else "compression (overexposure)"
                exec_note = f"Applied non-linear gamma lookup table correction (gamma={gamma:.2f}) for {mode}."

            elif action == ActionType.APPLY_DEHAZING:
                candidate_frame = dehaze_dark_channel(curr_frame)
                exec_note = "Applied Dark Channel Prior atmospheric transmission recovery with guided filter."

            elif action == ActionType.APPLY_UNSHARP_MASK:
                candidate_frame = enhance_sharpness_unsharp_mask(curr_frame, amount=1.2)
                exec_note = "Applied high-frequency Gaussian unsharp masking (amount=1.2)."

            # Recalculate metrics on candidate frame
            cand_metrics = self.assessor.assess_frame(
                candidate_frame,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec
            )

            # Quantitatively verify improvement vs preceding metrics
            eval_res = evaluate_enhancement_improvement(
                before_metrics=curr_metrics,
                after_metrics=cand_metrics,
                action_name=action.value
            )

            # Compare against the original unenhanced input frame
            quality_delta_vs_orig = round(
                cand_metrics.composite_quality_score - initial_metrics.composite_quality_score, 4
            )

            # Determine step outcome and stop condition
            if eval_res.accepted:
                state_check, _ = self.decision_engine.evaluate_reliability_state(cand_metrics)
                if state_check == ReliabilityState.RELIABLE:
                    step_outcome = ActionOutcome.ACCEPTED_RELIABLE.value
                    step_stop = StopCondition.STOP_RELIABLE_ACHIEVED.value
                else:
                    step_outcome = ActionOutcome.ACCEPTED_IMPROVED.value
                    step_stop = StopCondition.CONTINUE.value
            else:
                step_outcome = ActionOutcome.REJECTED_ROLLED_BACK.value
                step_stop = StopCondition.CONTINUE.value

            if attempt == self.max_attempts and step_stop == StopCondition.CONTINUE.value:
                step_stop = StopCondition.STOP_MAX_ATTEMPTS_EXHAUSTED.value

            audit_step = ActionAuditStep(
                attempt_index=attempt,
                action=action,
                accepted=eval_res.accepted,
                quality_delta=eval_res.quality_delta,
                before_composite=curr_metrics.composite_quality_score,
                after_composite=cand_metrics.composite_quality_score,
                evidence_reason=f"{eval_res.reason} {exec_note}".strip(),
                metric_deltas=eval_res.metric_deltas,
                reason=action_reason,
                outcome=step_outcome,
                stop_condition=step_stop,
                before_metrics=curr_metrics.to_dict(),
                after_metrics=cand_metrics.to_dict(),
                original_metrics=initial_metrics.to_dict(),
                quality_delta_vs_original=quality_delta_vs_orig
            )
            audit_trace.append(audit_step)
            last_step = audit_step

            if eval_res.accepted:
                curr_frame = candidate_frame
                curr_metrics = cand_metrics
                was_enhanced = True

                if step_outcome == ActionOutcome.ACCEPTED_RELIABLE.value:
                    final_stop_condition = StopCondition.STOP_RELIABLE_ACHIEVED.value
                    break
            else:
                # Roll back candidate; curr_frame and curr_metrics remain untouched
                pass

        # 6. Final State Evaluation & Comparison
        final_state_check, final_reason = self.decision_engine.evaluate_reliability_state(curr_metrics)

        if final_state_check == ReliabilityState.RELIABLE:
            final_state = ReliabilityState.RELIABLE
            human_review = False
            escalation_msg = None
            if final_stop_condition == StopCondition.CONTINUE.value:
                final_stop_condition = StopCondition.STOP_RELIABLE_ACHIEVED.value
        else:
            final_state = ReliabilityState.HUMAN_REVIEW_REQUIRED
            human_review = True
            if final_stop_condition == StopCondition.CONTINUE.value:
                final_stop_condition = (
                    StopCondition.STOP_MAX_ATTEMPTS_EXHAUSTED.value
                    if len(audit_trace) >= self.max_attempts
                    else StopCondition.STOP_NO_VIABLE_ACTIONS.value
                )
            escalation_msg = (
                f"Exhausted {len(audit_trace)} corrective action attempt(s) without achieving "
                f"acceptable operational reliability (Final score: {curr_metrics.composite_quality_score:.3f}). "
                f"Stop condition: {final_stop_condition}. "
                f"Active degradations: {'; '.join(curr_metrics.measurable_degradations)}."
            )

        comparison = self._compute_comparison_vs_original(
            initial_metrics=initial_metrics,
            final_metrics=curr_metrics,
            initial_state=initial_state,
            final_state=final_state
        )

        return AgentFrameDecision(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            initial_state=initial_state,
            final_state=final_state,
            initial_metrics=initial_metrics,
            final_metrics=curr_metrics,
            attempts_made=len(audit_trace),
            max_attempts=self.max_attempts,
            audit_trace=audit_trace,
            human_review_required=human_review,
            escalation_reason=escalation_msg,
            final_frame_bgr=curr_frame,
            was_enhanced=was_enhanced,
            stop_condition=final_stop_condition,
            comparison_vs_original=comparison
        )

    def _compute_comparison_vs_original(
        self,
        initial_metrics: FrameQualityMetrics,
        final_metrics: FrameQualityMetrics,
        initial_state: ReliabilityState,
        final_state: ReliabilityState
    ) -> Dict[str, Any]:
        """
        Computes summary comparison deltas between final frame and original input frame.
        """
        composite_delta = round(final_metrics.composite_quality_score - initial_metrics.composite_quality_score, 4)
        sharpness_delta = round(final_metrics.laplacian_variance - initial_metrics.laplacian_variance, 2)
        contrast_delta = round(final_metrics.rms_contrast - initial_metrics.rms_contrast, 4)
        brightness_delta = round(final_metrics.mean_brightness - initial_metrics.mean_brightness, 2)
        transmission_delta = round(final_metrics.mean_transmission - initial_metrics.mean_transmission, 4)

        return {
            "composite_delta": composite_delta,
            "sharpness_delta": sharpness_delta,
            "contrast_delta": contrast_delta,
            "brightness_delta": brightness_delta,
            "transmission_delta": transmission_delta,
            "original_state": initial_state.value if hasattr(initial_state, "value") else str(initial_state),
            "final_state": final_state.value if hasattr(final_state, "value") else str(final_state),
            "improved": bool(composite_delta > 0.0),
            "net_improvement_percentage": round(composite_delta * 100.0, 2),
            "original_composite_score": round(initial_metrics.composite_quality_score, 4),
            "final_composite_score": round(final_metrics.composite_quality_score, 4)
        }
