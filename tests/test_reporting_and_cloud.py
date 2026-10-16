"""
Unit tests for audit reporting and AWS cloud module.
"""

import os
import tempfile
import pytest

from src.reporting.audit_report import AuditReportGenerator
from src.cloud.aws_integration import AWSVisionAuditManager
from src.agent.state import ReliabilityState, AgentFrameDecision
from src.metrics.quality_assessor import FrameQualityMetrics


def _mock_decision(state: ReliabilityState, frame_idx: int = 0) -> AgentFrameDecision:
    m = FrameQualityMetrics(
        frame_index=frame_idx,
        timestamp_sec=frame_idx * 0.05,
        laplacian_variance=150.0,
        fft_high_freq_ratio=0.4,
        sharpness_score=0.8,
        is_blurred=False,
        mean_brightness=120.0,
        rms_contrast=0.22,
        dynamic_range=130,
        underexposed_ratio=0.0,
        overexposed_ratio=0.0,
        entropy=5.0,
        is_low_light=False,
        is_overexposed=False,
        is_low_contrast=False,
        mean_transmission=0.70,
        dark_channel_mean=28.0,
        haze_level=0.30,
        is_hazy_or_foggy=False,
        glare_ratio=0.0,
        is_severe_glare=False,
        color_cast_score=0.02,
        obstructed_cell_ratio=0.0,
        global_flatness_score=0.0,
        is_instantaneous_obstruction=False,
        is_persistent_obstruction=False,
        temporal_persistence_streak=0,
        is_transient_occlusion=False,
        composite_quality_score=0.80,
        measurable_degradations=[]
    )
    return AgentFrameDecision(
        frame_index=frame_idx,
        timestamp_sec=frame_idx * 0.05,
        initial_state=state,
        final_state=state,
        initial_metrics=m,
        final_metrics=m,
        attempts_made=0,
        max_attempts=3,
        human_review_required=(state == ReliabilityState.HUMAN_REVIEW_REQUIRED)
    )


def test_audit_report_export():
    decisions = [
        _mock_decision(ReliabilityState.RELIABLE, 0),
        _mock_decision(ReliabilityState.HUMAN_REVIEW_REQUIRED, 1)
    ]
    gen = AuditReportGenerator(decisions, video_name="test_feed")
    summary = gen.generate_summary_stats()

    assert summary["total_frames_analyzed"] == 2
    assert summary["reliable_frames"] == 1
    assert summary["human_escalations"] == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        csv_path = os.path.join(tmpdir, "report.csv")

        gen.export_json(json_path)
        gen.export_csv(csv_path)

        assert os.path.exists(json_path)
        assert os.path.exists(csv_path)
        assert os.path.getsize(json_path) > 100
        assert os.path.getsize(csv_path) > 100


def test_aws_offline_graceful_mode():
    mgr = AWSVisionAuditManager(bucket_name="")
    assert not mgr.is_cloud_enabled()
    info = mgr.get_status_info()
    assert info["status"] == "OFFLINE_LOCAL_MODE"
    assert mgr.upload_file("nonexistent.mp4") is None
