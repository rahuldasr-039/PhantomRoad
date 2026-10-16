"""
Unit and integration tests for multi-frame video optical quality profiler.
Tests metric calculations, threshold flags, overlapping flags (occurrences vs unique frames),
blur threshold margin auditing, and frame sampling strides.
"""

import os
import cv2
import numpy as np
import pytest

from src.metrics.batch_profiler import (
    profile_single_frame,
    profile_video_batch,
    summarize_records,
    FrameTelemetryRecord,
    BatchAnalysisSummary
)
from src.video.frame_extractor import VideoLoader
from src.video.synthetic_generator import generate_synthetic_road_video


@pytest.fixture
def test_video_path(tmp_path) -> str:
    """Generates a small 20-frame synthetic road video for batch testing."""
    v_path = str(tmp_path / "test_batch_video.mp4")
    generate_synthetic_road_video(
        scenario="night_low_light",
        output_path=v_path,
        num_frames=20,
        fps=10
    )
    return v_path


# =====================================================================
# 1. METRIC CALCULATION & FLAG TESTS
# =====================================================================

def test_profile_single_frame_nominal():
    """Verifies a nominal, well-lit, high-contrast frame produces 0 flags."""
    # Balanced frame with sharp edges and good dynamic range
    frame = np.full((120, 160, 3), 120, dtype=np.uint8)
    # Draw high contrast sharp shapes
    cv2.rectangle(frame, (20, 20), (140, 100), (20, 220, 20), -1)
    cv2.circle(frame, (80, 60), 30, (220, 20, 20), -1)

    rec = profile_single_frame(
        frame_bgr=frame,
        frame_index=0,
        timestamp_sec=0.0,
        blur_threshold=50.0,
        low_light_threshold=40.0,
        overexposed_threshold=210.0,
        min_rms_contrast=0.10
    )

    assert isinstance(rec, FrameTelemetryRecord)
    assert rec.frame_index == 0
    assert rec.is_blurred is False
    assert rec.is_low_light is False
    assert rec.is_overexposed is False
    assert rec.is_low_contrast is False
    assert rec.flag_count == 0
    assert rec.is_degraded is False
    assert rec.status_summary == "Nominal"


def test_profile_single_frame_dark_and_low_contrast_overlap():
    """
    Verifies that a dark, flat frame triggers BOTH Low Light and Low Contrast flags,
    resulting in multiple flags while being counted as a single unique degraded frame.
    """
    # Uniform dark frame (mean intensity ~18, zero contrast, zero sharpness)
    dark_flat_frame = np.full((120, 160, 3), 18, dtype=np.uint8)

    rec = profile_single_frame(
        frame_bgr=dark_flat_frame,
        frame_index=5,
        timestamp_sec=0.5,
        blur_threshold=10.0,
        low_light_threshold=45.0,
        overexposed_threshold=200.0,
        min_rms_contrast=0.12
    )

    assert rec.is_low_light is True
    assert rec.is_low_contrast is True
    assert rec.flag_count >= 2
    assert "Low Light" in rec.active_flags
    assert "Low Contrast" in rec.active_flags
    assert rec.is_degraded is True
    assert "Low Light" in rec.status_summary
    assert "Low Contrast" in rec.status_summary


# =====================================================================
# 2. OCCURRENCES VS UNIQUE DEGRADED FRAMES TESTS
# =====================================================================

def test_unique_degraded_vs_total_flag_occurrences():
    """
    Directly tests the audit distinction between total flag occurrences and unique degraded frames:
    - 5 frames are Dark only (5 flags total)
    - 3 frames are BOTH Dark and Low-Contrast (6 flags total)
    - 2 frames are Nominal (0 flags total)
    Total frames: 10
    Total flag occurrences: 5 + 6 = 11
    Unique degraded frames: 5 + 3 = 8
    """
    records = []
    # 1. 5 Dark-only frames
    for i in range(5):
        records.append(FrameTelemetryRecord(
            frame_index=i, timestamp_sec=i * 0.1, laplacian_variance=200.0, sharpness_score=0.8,
            mean_brightness=30.0, rms_contrast=0.18, dynamic_range=60, underexposed_ratio=0.3,
            overexposed_ratio=0.0, entropy=4.0, is_blurred=False, is_low_light=True,
            is_overexposed=False, is_low_contrast=False, active_flags=["Low Light"], flag_count=1,
            is_degraded=True, status_summary="Degraded (Low Light)"
        ))

    # 2. 3 Dual-flag frames (Dark + Low-Contrast)
    for i in range(5, 8):
        records.append(FrameTelemetryRecord(
            frame_index=i, timestamp_sec=i * 0.1, laplacian_variance=200.0, sharpness_score=0.8,
            mean_brightness=20.0, rms_contrast=0.06, dynamic_range=20, underexposed_ratio=0.5,
            overexposed_ratio=0.0, entropy=3.0, is_blurred=False, is_low_light=True,
            is_overexposed=False, is_low_contrast=True, active_flags=["Low Light", "Low Contrast"], flag_count=2,
            is_degraded=True, status_summary="Degraded (Low Light + Low Contrast)"
        ))

    # 3. 2 Nominal frames
    for i in range(8, 10):
        records.append(FrameTelemetryRecord(
            frame_index=i, timestamp_sec=i * 0.1, laplacian_variance=250.0, sharpness_score=0.9,
            mean_brightness=110.0, rms_contrast=0.22, dynamic_range=120, underexposed_ratio=0.0,
            overexposed_ratio=0.0, entropy=5.5, is_blurred=False, is_low_light=False,
            is_overexposed=False, is_low_contrast=False, active_flags=[], flag_count=0,
            is_degraded=False, status_summary="Nominal"
        ))

    summary = summarize_records(
        records=records,
        total_video_frames=10,
        stride=1,
        elapsed_time_sec=0.5,
        throughput_fps=20.0,
        blur_threshold=85.0,
        low_light_threshold=45.0,
        overexposed_threshold=200.0,
        min_rms_contrast=0.12
    )

    assert summary.frames_analyzed == 10
    # Crucial Verification: 11 total occurrences vs 8 unique frames
    assert summary.total_flag_occurrences == 11
    assert summary.unique_degraded_frames == 8
    assert summary.unique_degraded_percentage == 80.0
    assert summary.nominal_frames_count == 2
    assert summary.nominal_percentage == 20.0
    assert summary.single_flag_frames == 5
    assert summary.dual_flag_frames == 3


# =====================================================================
# 3. BLUR THRESHOLD DISCREPANCY & AUDIT EXPLANATION TESTS
# =====================================================================

def test_blur_threshold_explanation_when_zero_crossings():
    """
    Verifies that when min sharpness > threshold (e.g., 337.2 > 85.0),
    blur_flag_count is 0 and the explanation provides the exact mathematical margin.
    """
    records = [
        FrameTelemetryRecord(
            frame_index=0, timestamp_sec=0.0, laplacian_variance=337.2, sharpness_score=0.9,
            mean_brightness=21.8, rms_contrast=0.118, dynamic_range=40, underexposed_ratio=0.45,
            overexposed_ratio=0.0, entropy=4.1, is_blurred=False, is_low_light=True,
            is_overexposed=False, is_low_contrast=True, active_flags=["Low Light", "Low Contrast"],
            flag_count=2, is_degraded=True, status_summary="Degraded (Low Light + Low Contrast)"
        ),
        FrameTelemetryRecord(
            frame_index=1, timestamp_sec=0.1, laplacian_variance=946.7, sharpness_score=0.95,
            mean_brightness=25.0, rms_contrast=0.125, dynamic_range=45, underexposed_ratio=0.40,
            overexposed_ratio=0.0, entropy=4.3, is_blurred=False, is_low_light=True,
            is_overexposed=False, is_low_contrast=False, active_flags=["Low Light"],
            flag_count=1, is_degraded=True, status_summary="Degraded (Low Light)"
        )
    ]

    summary = summarize_records(
        records=records,
        total_video_frames=2,
        stride=1,
        elapsed_time_sec=0.1,
        throughput_fps=20.0,
        blur_threshold=85.0,
        low_light_threshold=45.0,
        overexposed_threshold=200.0,
        min_rms_contrast=0.12
    )

    assert summary.blur_flag_count == 0
    assert summary.min_sharpness == 337.2
    assert summary.mean_sharpness == round((337.2 + 946.7) / 2, 1)
    # Must explain why 0 crossed with margin
    assert "Zero frames crossed" in summary.blur_explanation
    assert "337.2" in summary.blur_explanation
    assert "85.0" in summary.blur_explanation
    assert "+252.2" in summary.blur_explanation


def test_blur_threshold_triggered_when_threshold_reconfigured():
    """Verifies that raising the blur threshold above 337.2 accurately flags frames as blurred."""
    records = [
        FrameTelemetryRecord(
            frame_index=0, timestamp_sec=0.0, laplacian_variance=337.2, sharpness_score=0.9,
            mean_brightness=21.8, rms_contrast=0.118, dynamic_range=40, underexposed_ratio=0.45,
            overexposed_ratio=0.0, entropy=4.1, is_blurred=True, is_low_light=True,
            is_overexposed=False, is_low_contrast=True, active_flags=["Blur", "Low Light", "Low Contrast"],
            flag_count=3, is_degraded=True, status_summary="Degraded (Blur + Low Light + Low Contrast)"
        )
    ]

    summary = summarize_records(
        records=records,
        total_video_frames=1,
        stride=1,
        elapsed_time_sec=0.05,
        throughput_fps=20.0,
        blur_threshold=400.0,  # Reconfigured higher
        low_light_threshold=45.0,
        overexposed_threshold=200.0,
        min_rms_contrast=0.12
    )

    assert summary.blur_flag_count == 1
    assert "crossed the blur threshold" in summary.blur_explanation


# =====================================================================
# 4. SAMPLING STRIDE & PERFORMANCE TESTS
# =====================================================================

def test_profile_video_batch_strides(test_video_path):
    """Verifies configurable frame stride downsamples accurately without fabricating frames."""
    with VideoLoader(test_video_path) as loader:
        total = loader.total_frames
        assert total == 20

        # Stride 1: all 20 frames
        s1 = profile_video_batch(loader, stride=1)
        assert s1.frames_analyzed == 20
        assert s1.sampling_stride == 1
        assert len(s1.records) == 20

        # Stride 2: 10 frames
        s2 = profile_video_batch(loader, stride=2)
        assert s2.frames_analyzed == 10
        assert s2.sampling_stride == 2
        assert len(s2.records) == 10

        # Stride 5: 4 frames
        s5 = profile_video_batch(loader, stride=5)
        assert s5.frames_analyzed == 4
        assert s5.sampling_stride == 5
        assert len(s5.records) == 4

        # Max frames limit
        s_max = profile_video_batch(loader, stride=1, max_frames=8)
        assert s_max.frames_analyzed == 8


def test_frame_telemetry_table_dataframe_export(test_video_path):
    """Verifies that the telemetry dataframe contains all expected columns and valid data."""
    with VideoLoader(test_video_path) as loader:
        summary = profile_video_batch(loader, stride=2)
        df = summary.to_dataframe()

        expected_cols = [
            "Frame", "Timestamp (s)", "Laplacian Variance", "Sharpness Score",
            "Mean Brightness", "RMS Contrast", "Dynamic Range", "Underexposed (%)",
            "Overexposed (%)", "Entropy (bits)", "Blur Flag", "Lighting Flag",
            "Contrast Flag", "Active Flags", "Flag Count", "Is Degraded", "Status Summary"
        ]

        for col in expected_cols:
            assert col in df.columns, f"Missing expected column '{col}' in telemetry dataframe"

        assert len(df) == 10
        assert df["Frame"].iloc[0] == 0
        assert df["Frame"].iloc[1] == 2
