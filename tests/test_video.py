"""
Unit tests for video loading and synthetic scenario generation.
"""

import os
import tempfile
import pytest

from src.video.synthetic_generator import generate_synthetic_road_video, AVAILABLE_SCENARIOS
from src.video.frame_extractor import VideoLoader


@pytest.fixture
def temp_video_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_clip.mp4")


def test_generate_clear_video(temp_video_path):
    out_path = generate_synthetic_road_video(
        scenario="clear_day",
        output_path=temp_video_path,
        num_frames=15,
        fps=20,
        width=320,
        height=240
    )
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000

    with VideoLoader(out_path) as loader:
        meta = loader.metadata
        assert meta.total_frames == 15
        assert meta.width == 320
        assert meta.height == 240

        frame_0 = loader.get_frame(0)
        assert frame_0 is not None
        assert frame_0.shape == (240, 320, 3)

        # Test temporal neighborhood
        neighbors = loader.get_temporal_neighborhood(target_index=5, radius=2)
        assert len(neighbors) == 4
        assert [n[0] for n in neighbors] == [3, 4, 6, 7]


def test_generate_all_scenarios():
    with tempfile.TemporaryDirectory() as tmpdir:
        for scenario in AVAILABLE_SCENARIOS:
            clip_path = os.path.join(tmpdir, f"{scenario}.mp4")
            generate_synthetic_road_video(
                scenario=scenario,
                output_path=clip_path,
                num_frames=8,
                fps=20,
                width=320,
                height=240
            )
            assert os.path.exists(clip_path)
            assert os.path.getsize(clip_path) > 500
