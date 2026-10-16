"""
Audit Report Generator producing verifiable JSON and tabular CSV compliance records.
"""

from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime, timezone
import pandas as pd

from ..agent.state import AgentFrameDecision, ReliabilityState


class AuditReportGenerator:
    """
    Consolidates frame-by-frame agent decisions and telemetry into exportable audit files.
    """

    def __init__(self, decisions: List[AgentFrameDecision], video_name: str = "road_scene"):
        self.decisions = decisions
        self.video_name = video_name

    def generate_summary_stats(self) -> Dict[str, Any]:
        """
        Calculates aggregate key performance indicators across all analyzed frames.
        """
        total = len(self.decisions)
        if total == 0:
            return {
                "total_frames_analyzed": 0,
                "reliable_percentage": 0.0,
                "human_escalation_percentage": 0.0,
                "recovery_percentage": 0.0
            }

        reliable_count = sum(1 for d in self.decisions if d.final_state == ReliabilityState.RELIABLE)
        escalated_count = sum(1 for d in self.decisions if d.human_review_required)
        enhanced_count = sum(1 for d in self.decisions if d.was_enhanced)
        initially_degraded = sum(1 for d in self.decisions if d.initial_state != ReliabilityState.RELIABLE)
        recovered_count = sum(
            1 for d in self.decisions
            if d.initial_state != ReliabilityState.RELIABLE and d.final_state == ReliabilityState.RELIABLE
        )

        recovery_pct = (recovered_count / initially_degraded * 100.0) if initially_degraded > 0 else 100.0

        return {
            "total_frames_analyzed": total,
            "reliable_frames": reliable_count,
            "reliable_percentage": round(reliable_count / total * 100.0, 1),
            "initially_degraded_frames": initially_degraded,
            "recovered_frames": recovered_count,
            "recovery_percentage": round(recovery_pct, 1),
            "human_escalations": escalated_count,
            "human_escalation_percentage": round(escalated_count / total * 100.0, 1),
            "enhanced_actions_applied": enhanced_count
        }

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Creates full nested JSON audit report.
        """
        summary = self.generate_summary_stats()
        return {
            "audit_metadata": {
                "system_name": "PhantomRoad: Adaptive Vision Reliability Agent",
                "competition": "OpenCV AI Competition 2026",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "video_source": self.video_name
            },
            "summary_metrics": summary,
            "frame_audit_records": [d.to_dict() for d in self.decisions]
        }

    def export_json(self, output_path: str) -> str:
        """
        Saves full audit document to JSON.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = self.to_json_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return os.path.abspath(output_path)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flattens frame telemetry into a pandas DataFrame.
        """
        rows = []
        for d in self.decisions:
            im = d.initial_metrics
            fm = d.final_metrics
            rows.append({
                "frame_index": d.frame_index,
                "timestamp_sec": d.timestamp_sec,
                "initial_state": d.initial_state.value,
                "final_state": d.final_state.value,
                "was_enhanced": d.was_enhanced,
                "attempts_made": d.attempts_made,
                "human_review_required": d.human_review_required,
                "escalation_reason": d.escalation_reason or "",
                "laplacian_variance_initial": im.laplacian_variance,
                "laplacian_variance_final": fm.laplacian_variance,
                "brightness_initial": im.mean_brightness,
                "brightness_final": fm.mean_brightness,
                "rms_contrast_initial": im.rms_contrast,
                "rms_contrast_final": fm.rms_contrast,
                "transmission_initial": im.mean_transmission,
                "transmission_final": fm.mean_transmission,
                "composite_quality_initial": im.composite_quality_score,
                "composite_quality_final": fm.composite_quality_score,
                "degradations_flagged": "; ".join(im.measurable_degradations)
            })
        return pd.DataFrame(rows)

    def export_csv(self, output_path: str) -> str:
        """
        Saves tabular summary to CSV.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        return os.path.abspath(output_path)
