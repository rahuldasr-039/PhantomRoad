"""
State definitions and audit data structures for the Adaptive Reliability Agent.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from ..metrics.quality_assessor import FrameQualityMetrics


class ReliabilityState(str, Enum):
    """
    Visual evidence reliability states.
    """
    RELIABLE = "RELIABLE"
    DEGRADED_CORRECTABLE = "DEGRADED_CORRECTABLE"
    DEGRADED_UNCERTAIN = "DEGRADED_UNCERTAIN"
    CRITICAL_OBSTRUCTION = "CRITICAL_OBSTRUCTION"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class ActionType(str, Enum):
    """
    Discrete actions selectable by the agent.
    """
    NO_ACTION_NEEDED = "NO_ACTION_NEEDED"
    SELECT_SHARPER_NEIGHBOR = "SELECT_SHARPER_NEIGHBOR"
    APPLY_CONTRAST_CLAHE = "APPLY_CONTRAST_CLAHE"
    APPLY_GAMMA_CORRECTION = "APPLY_GAMMA_CORRECTION"
    APPLY_DEHAZING = "APPLY_DEHAZING"
    APPLY_UNSHARP_MASK = "APPLY_UNSHARP_MASK"
    MARK_OBSTRUCTED = "MARK_OBSTRUCTED"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"


class StopCondition(str, Enum):
    """
    Conditions that terminate the adaptive reliability decision loop.
    """
    STOP_ALREADY_RELIABLE = "STOP_ALREADY_RELIABLE"
    STOP_RELIABLE_ACHIEVED = "STOP_RELIABLE_ACHIEVED"
    STOP_CRITICAL_OBSTRUCTION = "STOP_CRITICAL_OBSTRUCTION"
    STOP_MAX_ATTEMPTS_EXHAUSTED = "STOP_MAX_ATTEMPTS_EXHAUSTED"
    STOP_NO_VIABLE_ACTIONS = "STOP_NO_VIABLE_ACTIONS"
    CONTINUE = "CONTINUE"


class ActionOutcome(str, Enum):
    """
    Outcome status of an individual action attempt.
    """
    ACCEPTED_IMPROVED = "ACCEPTED_IMPROVED"
    ACCEPTED_RELIABLE = "ACCEPTED_RELIABLE"
    REJECTED_ROLLED_BACK = "REJECTED_ROLLED_BACK"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    CRITICAL_OBSTRUCTION_DETECTED = "CRITICAL_OBSTRUCTION_DETECTED"
    HUMAN_REVIEW_TRIGGERED = "HUMAN_REVIEW_TRIGGERED"
    ALREADY_RELIABLE = "ALREADY_RELIABLE"


@dataclass
class ActionAuditStep:
    """
    Audit record for an individual action attempt within the agent loop.
    Enriched for Stage 2 closed-loop explainability and original comparison.
    """
    attempt_index: int
    action: ActionType
    accepted: bool
    quality_delta: float
    before_composite: float
    after_composite: float
    evidence_reason: str
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    outcome: str = ""
    stop_condition: Optional[str] = None
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)
    original_metrics: Dict[str, float] = field(default_factory=dict)
    quality_delta_vs_original: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "action": self.action.value if hasattr(self.action, "value") else str(self.action),
            "reason": self.reason or self.evidence_reason,
            "accepted": self.accepted,
            "outcome": self.outcome or ("ACCEPTED_IMPROVED" if self.accepted else "REJECTED_ROLLED_BACK"),
            "quality_delta": round(self.quality_delta, 4),
            "quality_delta_vs_original": round(self.quality_delta_vs_original, 4),
            "before_composite": round(self.before_composite, 4),
            "after_composite": round(self.after_composite, 4),
            "stop_condition": self.stop_condition or "CONTINUE",
            "evidence_reason": self.evidence_reason,
            "metric_deltas": self.metric_deltas,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "original_metrics": self.original_metrics
        }


@dataclass
class AgentFrameDecision:
    """
    Complete audit outcome for a processed video frame.
    """
    frame_index: int
    timestamp_sec: float
    initial_state: ReliabilityState
    final_state: ReliabilityState
    initial_metrics: FrameQualityMetrics
    final_metrics: FrameQualityMetrics
    attempts_made: int
    max_attempts: int
    audit_trace: List[ActionAuditStep] = field(default_factory=list)
    human_review_required: bool = False
    escalation_reason: Optional[str] = None
    final_frame_bgr: Optional[np.ndarray] = None
    was_enhanced: bool = False
    stop_condition: str = StopCondition.STOP_ALREADY_RELIABLE.value
    comparison_vs_original: Dict[str, Any] = field(default_factory=dict)

    @property
    def action_log(self) -> List[ActionAuditStep]:
        """Alias for audit_trace satisfying Stage 2 action log interface."""
        return self.audit_trace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "initial_state": self.initial_state.value if hasattr(self.initial_state, "value") else str(self.initial_state),
            "final_state": self.final_state.value if hasattr(self.final_state, "value") else str(self.final_state),
            "attempts_made": self.attempts_made,
            "max_attempts": self.max_attempts,
            "stop_condition": self.stop_condition,
            "human_review_required": self.human_review_required,
            "escalation_reason": self.escalation_reason,
            "was_enhanced": self.was_enhanced,
            "comparison_vs_original": self.comparison_vs_original,
            "action_log": [step.to_dict() for step in self.audit_trace],
            "audit_trace": [step.to_dict() for step in self.audit_trace],
            "initial_metrics": self.initial_metrics.to_dict(),
            "final_metrics": self.final_metrics.to_dict()
        }

