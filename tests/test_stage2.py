"""
Unit and integration tests for Stage 2: Modular Reliability Decision Engine & Adaptive Action Closed-Loop.
Verifies dynamic pivoting, rejection rollback, neighbor inspection, before/after recalculation,
comparison against original, action log integrity, and stop conditions.
"""

import numpy as np
import cv2
import pytest

from src.agent.state import (
    ReliabilityState,
    ActionType,
    StopCondition,
    ActionOutcome,
    ActionAuditStep,
    AgentFrameDecision
)
from src.agent.decision_engine import ReliabilityDecisionEngine
from src.agent.action_executor import AdaptiveReliabilityAgent
from src.metrics.quality_assessor import QualityAssessor


@pytest.fixture
def base_clear_road() -> np.ndarray:
    """Creates a high-contrast sharp road scene."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[0:100, :] = [210, 180, 140]  # Sky
    img[100:, :] = [70, 70, 70]      # Road surface
    cv2.line(img, (160, 100), (40, 240), (255, 255, 255), 4)   # Left lane
    cv2.line(img, (160, 100), (280, 240), (255, 255, 255), 4)  # Right lane
    for y in range(110, 240, 30):
        cv2.line(img, (160, y), (160, y + 15), (0, 220, 255), 3)  # Center dashed
    return img


@pytest.fixture
def dark_road(base_clear_road: np.ndarray) -> np.ndarray:
    """Simulates underexposed low-light road scene."""
    return (base_clear_road * 0.12).astype(np.uint8)


@pytest.fixture
def blurry_road(base_clear_road: np.ndarray) -> np.ndarray:
    """Simulates motion/defocus blur via Gaussian filter."""
    return cv2.GaussianBlur(base_clear_road, (21, 21), 9.0)


@pytest.fixture
def low_contrast_road(base_clear_road: np.ndarray) -> np.ndarray:
    """Simulates flat, low-contrast overcast road scene."""
    gray = cv2.cvtColor(base_clear_road, cv2.COLOR_BGR2GRAY)
    flat = cv2.addWeighted(gray, 0.2, np.full_like(gray, 120), 0.8, 0)
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)


# =====================================================================
# 1. MODULAR DECISION ENGINE TESTS
# =====================================================================

def test_decision_engine_initial_state_and_explainable_reasons(
    base_clear_road, dark_road, blurry_road
):
    engine = ReliabilityDecisionEngine()
    assessor = QualityAssessor()

    # Clear frame -> RELIABLE
    clear_metrics = assessor.assess_frame(base_clear_road)
    state, reason = engine.evaluate_reliability_state(clear_metrics)
    assert state == ReliabilityState.RELIABLE
    assert "reliable" in reason.lower()
    assert "composite score" in reason.lower()

    # Dark frame -> DEGRADED_CORRECTABLE
    dark_metrics = assessor.assess_frame(dark_road)
    state, reason = engine.evaluate_reliability_state(dark_metrics)
    assert state == ReliabilityState.DEGRADED_CORRECTABLE
    assert "low-light" in reason.lower() or "underexposed" in reason.lower() or "degraded" in reason.lower()

    # Blurry frame -> DEGRADED_CORRECTABLE
    blur_metrics = assessor.assess_frame(blurry_road)
    state, reason = engine.evaluate_reliability_state(blur_metrics)
    assert state == ReliabilityState.DEGRADED_CORRECTABLE
    assert "blur" in reason.lower() or "degraded" in reason.lower()


def test_decision_engine_action_selection_priority(blurry_road, dark_road):
    engine = ReliabilityDecisionEngine()
    assessor = QualityAssessor()

    # For blurry frame with temporal neighbors available: selects neighbor inspection first
    blur_m = assessor.assess_frame(blurry_road)
    action, reason = engine.decide_next_action(blur_m, attempted_actions=set(), has_temporal_neighbors=True)
    assert action == ActionType.SELECT_SHARPER_NEIGHBOR
    assert "neighboring frames" in reason.lower() or "temporal buffer" in reason.lower()

    # If neighbor inspection was already attempted: pivots to unsharp masking
    action2, reason2 = engine.decide_next_action(
        blur_m,
        attempted_actions={ActionType.SELECT_SHARPER_NEIGHBOR},
        has_temporal_neighbors=True
    )
    assert action2 == ActionType.APPLY_UNSHARP_MASK
    assert "unsharp" in reason2.lower()

    # For low-light frame: selects gamma correction
    dark_m = assessor.assess_frame(dark_road)
    action_dark, reason_dark = engine.decide_next_action(dark_m, attempted_actions=set())
    assert action_dark == ActionType.APPLY_GAMMA_CORRECTION
    assert "gamma" in reason_dark.lower() or "luminance" in reason_dark.lower()


# =====================================================================
# 2. CLOSED-LOOP DYNAMIC ADAPTATION (NO FIXED SEQUENCE)
# =====================================================================

def test_agent_changes_action_based_on_intermediate_measurements(base_clear_road):
    """
    Proves the agent adapts dynamically:
    Creates a compound degraded frame (dark + low contrast / blur).
    Step 1: Agent observes low light and applies gamma correction.
    Step 2: Recalculates metrics. Brightness is resolved, but remaining degradation persists.
    Agent dynamically chooses the next action based on recalculated telemetry, proving closed-loop observation!
    """
    # Create dark frame with preserved structure
    dark_flat = (base_clear_road * 0.15).astype(np.uint8)
    gray = cv2.cvtColor(dark_flat, cv2.COLOR_BGR2GRAY)
    flat_gray = cv2.addWeighted(gray, 0.35, np.full_like(gray, 20), 0.65, 0)
    compound_degraded = cv2.cvtColor(flat_gray, cv2.COLOR_GRAY2BGR)

    agent = AdaptiveReliabilityAgent(max_attempts=3)
    decision = agent.process_frame(compound_degraded, frame_index=0)

    assert decision.attempts_made >= 2
    step1 = decision.action_log[0]
    step2 = decision.action_log[1]

    # Step 1 was chosen dynamically to address low light
    assert step1.action == ActionType.APPLY_GAMMA_CORRECTION
    assert step1.accepted is True

    # Step 2 was chosen dynamically based on the RECALCULATED intermediate metrics
    assert step2.action in [ActionType.APPLY_UNSHARP_MASK, ActionType.APPLY_CONTRAST_CLAHE]
    assert step2.action != step1.action


def test_agent_rejection_and_rollback_dynamic_pivot(base_clear_road):
    """
    Proves that when an enhancement candidate fails to quantitatively improve quality,
    the agent rejects it, rolls back, and dynamically pivots to an alternative action.
    """
    # Create a heavily blurred frame (Laplacian variance < 30)
    blurry_target = cv2.GaussianBlur(base_clear_road, (25, 25), 10.0)

    # When neighbors are available but none are sharper, neighbor inspection is rejected
    more_blurry = cv2.GaussianBlur(blurry_target, (15, 15), 5.0)
    neighbors = [(0, more_blurry), (2, more_blurry)]

    agent = AdaptiveReliabilityAgent(max_attempts=3)
    decision = agent.process_frame(blurry_target, frame_index=1, neighbor_frames=neighbors)

    assert decision.attempts_made >= 2
    step1 = decision.action_log[0]
    step2 = decision.action_log[1]

    # Step 1 attempted neighbor inspection, verified no neighbor was sharper, rejected and rolled back
    assert step1.action == ActionType.SELECT_SHARPER_NEIGHBOR
    assert step1.accepted is False
    assert step1.outcome == ActionOutcome.REJECTED_ROLLED_BACK.value
    assert "rejected" in step1.evidence_reason.lower()

    # Step 2 dynamically pivoted to another action (APPLY_UNSHARP_MASK) without repeating failed action
    assert step2.action != step1.action
    assert step2.action == ActionType.APPLY_UNSHARP_MASK



# =====================================================================
# 3. NEIGHBOR INSPECTION & SHARPER FRAME SELECTION
# =====================================================================

def test_neighbor_inspection_selects_sharper_frame(base_clear_road, blurry_road):
    """
    Tests that when a blurry target frame has a sharp neighbor in the temporal buffer,
    the agent inspects the neighbors, selects the sharper neighbor, recalculates metrics,
    and accepts it.
    """
    agent = AdaptiveReliabilityAgent(max_attempts=3)

    # Neighborhood: Frame #1 is blurry, Frame #0 is sharp, Frame #2 is blurry
    neighbors = [
        (0, base_clear_road),
        (2, blurry_road)
    ]

    decision = agent.process_frame(
        frame_bgr=blurry_road,
        frame_index=1,
        neighbor_frames=neighbors
    )

    assert decision.attempts_made >= 1
    step1 = decision.action_log[0]

    assert step1.action == ActionType.SELECT_SHARPER_NEIGHBOR
    assert step1.accepted is True
    assert step1.outcome in [ActionOutcome.ACCEPTED_RELIABLE.value, ActionOutcome.ACCEPTED_IMPROVED.value]
    assert step1.quality_delta > 0.0
    assert "sharper neighbor frame #0" in step1.evidence_reason.lower()

    # Verified that final frame matches the sharp neighbor
    assert decision.final_metrics.laplacian_variance > step1.before_metrics["laplacian_variance"]


def test_neighbor_inspection_fallback_when_no_neighbor_is_sharper(blurry_road):
    """
    Tests that when all neighbors are equally or more blurry, neighbor selection is rejected,
    and the agent dynamically pivots to unsharp masking on the next step.
    """
    agent = AdaptiveReliabilityAgent(max_attempts=3)

    # Neighborhood has only equally blurry frames
    more_blurry = cv2.GaussianBlur(blurry_road, (15, 15), 5.0)
    neighbors = [
        (0, more_blurry),
        (2, more_blurry)
    ]

    decision = agent.process_frame(
        frame_bgr=blurry_road,
        frame_index=1,
        neighbor_frames=neighbors
    )

    assert decision.attempts_made >= 2
    step1 = decision.action_log[0]
    step2 = decision.action_log[1]

    # Step 1 attempted neighbor selection but found no sharper candidate
    assert step1.action == ActionType.SELECT_SHARPER_NEIGHBOR
    assert step1.accepted is False
    assert step1.outcome == ActionOutcome.REJECTED_ROLLED_BACK.value

    # Step 2 dynamically pivoted to spatial unsharp masking
    assert step2.action == ActionType.APPLY_UNSHARP_MASK


# =====================================================================
# 4. ACTION LOG & COMPARISON AGAINST ORIGINAL FRAME
# =====================================================================

def test_action_log_and_recalculated_metrics_integrity(dark_road):
    """
    Verifies that every action log entry records reason, before/after measurements,
    original measurements, deltas, outcome, and stop condition.
    """
    agent = AdaptiveReliabilityAgent(max_attempts=3)
    decision = agent.process_frame(dark_road, frame_index=0)

    assert len(decision.action_log) > 0

    for step in decision.action_log:
        assert isinstance(step, ActionAuditStep)
        # Reason must explain why the action was chosen
        assert len(step.reason) > 5
        # Measurements before action
        assert "laplacian_variance" in step.before_metrics
        assert "mean_brightness" in step.before_metrics
        assert "rms_contrast" in step.before_metrics
        assert "composite_quality_score" in step.before_metrics
        # Measurements after action
        assert "laplacian_variance" in step.after_metrics
        assert "mean_brightness" in step.after_metrics
        assert "rms_contrast" in step.after_metrics
        assert "composite_quality_score" in step.after_metrics
        # Original measurements
        assert "composite_quality_score" in step.original_metrics
        # Deltas
        assert step.quality_delta == pytest.approx(
            step.after_composite - step.before_composite, abs=1e-3
        )
        assert step.quality_delta_vs_original == pytest.approx(
            step.after_composite - step.original_metrics["composite_quality_score"], abs=1e-3
        )
        # Outcome & Stop condition
        assert step.outcome != ""
        assert step.stop_condition in [
            StopCondition.CONTINUE.value,
            StopCondition.STOP_RELIABLE_ACHIEVED.value,
            StopCondition.STOP_MAX_ATTEMPTS_EXHAUSTED.value,
            StopCondition.STOP_CRITICAL_OBSTRUCTION.value
        ]

    # Verify comparison vs original dictionary
    comp = decision.comparison_vs_original
    assert "composite_delta" in comp
    assert "sharpness_delta" in comp
    assert "contrast_delta" in comp
    assert "brightness_delta" in comp
    assert "improved" in comp
    assert "net_improvement_percentage" in comp
    assert comp["original_state"] == decision.initial_state.value
    assert comp["final_state"] == decision.final_state.value


# =====================================================================
# 5. STOP CONDITIONS COVERAGE
# =====================================================================

def test_stop_condition_already_reliable(base_clear_road):
    agent = AdaptiveReliabilityAgent(max_attempts=3)
    decision = agent.process_frame(base_clear_road, frame_index=0)

    assert decision.stop_condition == StopCondition.STOP_ALREADY_RELIABLE.value
    assert decision.initial_state == ReliabilityState.RELIABLE
    assert decision.final_state == ReliabilityState.RELIABLE
    assert decision.attempts_made == 0
    assert not decision.human_review_required


def test_stop_condition_critical_obstruction(base_clear_road):
    assessor = QualityAssessor(min_persistence=2)
    agent = AdaptiveReliabilityAgent(assessor=assessor, max_attempts=3)

    # Blackout frame simulates covered camera lens
    blackout = np.zeros_like(base_clear_road)
    agent.process_frame(blackout, frame_index=0)
    decision = agent.process_frame(blackout, frame_index=1)

    assert decision.stop_condition == StopCondition.STOP_CRITICAL_OBSTRUCTION.value
    assert decision.initial_state == ReliabilityState.CRITICAL_OBSTRUCTION
    assert decision.final_state == ReliabilityState.HUMAN_REVIEW_REQUIRED
    assert decision.human_review_required is True
    assert "critical lens obstruction" in decision.escalation_reason.lower()


def test_stop_condition_max_attempts_exhausted(base_clear_road):
    """
    Frame with extreme synthetic degradation that cannot achieve the 0.65 threshold.
    Verifies stop condition STOP_MAX_ATTEMPTS_EXHAUSTED and human review escalation.
    """
    fog = np.full_like(base_clear_road, 200, dtype=np.uint8)
    extreme_degraded = cv2.addWeighted(base_clear_road, 0.10, fog, 0.90, 0)
    extreme_degraded = cv2.GaussianBlur(extreme_degraded, (51, 51), 25.0)

    agent = AdaptiveReliabilityAgent(max_attempts=2)
    decision = agent.process_frame(extreme_degraded, frame_index=0)

    assert decision.attempts_made == 2
    assert decision.stop_condition == StopCondition.STOP_MAX_ATTEMPTS_EXHAUSTED.value
    assert decision.final_state == ReliabilityState.HUMAN_REVIEW_REQUIRED
    assert decision.human_review_required is True
    assert decision.escalation_reason is not None


# =====================================================================
# 6. SERIALIZATION & DICTIONARY CONVERSION
# =====================================================================

def test_decision_to_dict_serialization(dark_road):
    agent = AdaptiveReliabilityAgent(max_attempts=3)
    decision = agent.process_frame(dark_road, frame_index=5, timestamp_sec=0.25)

    d_dict = decision.to_dict()
    assert d_dict["frame_index"] == 5
    assert d_dict["timestamp_sec"] == 0.25
    assert d_dict["stop_condition"] == decision.stop_condition
    assert "comparison_vs_original" in d_dict
    assert "action_log" in d_dict
    assert len(d_dict["action_log"]) == len(decision.action_log)
    for step_dict in d_dict["action_log"]:
        assert "reason" in step_dict
        assert "outcome" in step_dict
        assert "stop_condition" in step_dict
        assert "before_metrics" in step_dict
        assert "after_metrics" in step_dict
        assert "original_metrics" in step_dict
