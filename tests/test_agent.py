"""
Unit tests for the Adaptive Reliability Agent and Decision Engine.
"""

import numpy as np
import cv2
import pytest

from src.agent.state import ReliabilityState, ActionType, AgentFrameDecision
from src.agent.action_executor import AdaptiveReliabilityAgent
from src.metrics.quality_assessor import QualityAssessor


@pytest.fixture
def clear_road() -> np.ndarray:
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[0:150, :] = [210, 180, 140]
    img[150:, :] = [70, 70, 70]
    cv2.line(img, (240, 150), (60, 360), (255, 255, 255), 6)
    cv2.line(img, (240, 150), (420, 360), (255, 255, 255), 6)
    for y in range(160, 360, 30):
        cv2.line(img, (240, y), (240, y + 15), (0, 220, 255), 4)
    return img


@pytest.fixture
def underexposed_road(clear_road: np.ndarray) -> np.ndarray:
    return (clear_road * 0.15).astype(np.uint8)


def test_agent_clear_frame(clear_road):
    agent = AdaptiveReliabilityAgent(max_attempts=3)
    decision = agent.process_frame(clear_road, frame_index=0)

    assert isinstance(decision, AgentFrameDecision)
    assert decision.initial_state == ReliabilityState.RELIABLE
    assert decision.final_state == ReliabilityState.RELIABLE
    assert decision.attempts_made == 0
    assert not decision.human_review_required
    assert not decision.was_enhanced


def test_agent_enhances_underexposed_frame(underexposed_road):
    agent = AdaptiveReliabilityAgent(max_attempts=3)
    decision = agent.process_frame(underexposed_road, frame_index=0)

    assert decision.initial_state != ReliabilityState.RELIABLE
    assert decision.attempts_made > 0
    # First action attempted for underexposed frame should be gamma correction
    first_step = decision.audit_trace[0]
    assert first_step.action == ActionType.APPLY_GAMMA_CORRECTION
    assert first_step.accepted
    assert decision.was_enhanced


def test_agent_max_attempts_cap(clear_road):
    # Construct an image with extreme atmospheric fog and heavy blur that cannot be fully restored
    fog_layer = np.full_like(clear_road, 190, dtype=np.uint8)
    degraded = cv2.addWeighted(clear_road, 0.15, fog_layer, 0.85, 0)
    degraded = cv2.GaussianBlur(degraded, (41, 41), 18.0)

    agent = AdaptiveReliabilityAgent(max_attempts=2)
    decision = agent.process_frame(degraded, frame_index=0)

    # Attempts must never exceed configured max_attempts (2)
    assert decision.attempts_made <= 2
    assert decision.human_review_required
    assert decision.final_state == ReliabilityState.HUMAN_REVIEW_REQUIRED
    assert decision.escalation_reason is not None


def test_agent_critical_obstruction(clear_road):
    assessor = QualityAssessor(min_persistence=2)
    agent = AdaptiveReliabilityAgent(assessor=assessor, max_attempts=3)

    # Completely blackened sensor frame
    blackout = np.zeros_like(clear_road)

    # Run consecutive frames to trigger persistent obstruction
    agent.process_frame(blackout, frame_index=0)
    decision = agent.process_frame(blackout, frame_index=1)

    assert decision.initial_state == ReliabilityState.CRITICAL_OBSTRUCTION
    assert decision.final_state == ReliabilityState.HUMAN_REVIEW_REQUIRED
    assert decision.human_review_required
    assert any(step.action == ActionType.MARK_OBSTRUCTED for step in decision.audit_trace)
