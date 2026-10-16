"""
Video extraction, frame caching, and temporal neighborhood access.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator
import os
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """
    Metadata for a loaded video source.
    """
    filepath: str
    total_frames: int
    fps: float
    duration_sec: float
    width: int
    height: int


def validate_video_path(
    filepath: str,
    allowed_extensions: Optional[List[str]] = None,
    max_size_bytes: int = 500 * 1024 * 1024
) -> Tuple[bool, str, Optional[VideoMetadata]]:
    """
    Validates a video file for existence, format, file size, and OpenCV decodability.

    Args:
        filepath: Path to the target video file.
        allowed_extensions: List of allowed file extensions (default: mp4, avi, mov, mkv, webm).
        max_size_bytes: Maximum allowed file size in bytes (default 500MB).

    Returns:
        Tuple of (is_valid: bool, message: str, metadata: Optional[VideoMetadata])
    """
    if not isinstance(filepath, str) or not filepath.strip():
        return (False, "Filepath must be a non-empty string.", None)

    if not os.path.exists(filepath):
        return (False, f"Video file not found at: {filepath}", None)

    if not os.path.isfile(filepath):
        return (False, f"Target path is a directory, not a video file: {filepath}", None)

    allowed_exts = [e.lower() for e in (allowed_extensions or [".mp4", ".avi", ".mov", ".mkv", ".webm"])]
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in allowed_exts:
        return (False, f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(allowed_exts)}", None)

    size = os.path.getsize(filepath)
    if size == 0:
        return (False, "Video file is empty (0 bytes).", None)

    if size > max_size_bytes:
        return (False, f"Video file exceeds size limit ({size / (1024 * 1024):.1f}MB > {max_size_bytes / (1024 * 1024):.1f}MB).", None)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        cap.release()
        return (False, "Failed to open video stream. File may be corrupted or encoded in an unsupported codec.", None)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0 or total_frames <= 0:
        cap.release()
        return (False, "Video stream contains invalid dimensions or zero frames.", None)

    ret, test_frame = cap.read()
    cap.release()
    if not ret or test_frame is None:
        return (False, "Video stream could not be decoded into valid image frames.", None)

    duration = total_frames / fps if fps > 0 else 0.0
    meta = VideoMetadata(
        filepath=filepath,
        total_frames=total_frames,
        fps=round(fps, 2),
        duration_sec=round(duration, 2),
        width=width,
        height=height
    )
    return (True, "Video file is valid and decodable.", meta)


class VideoLoader:
    """
    Handles robust video decoding, random frame access, and temporal neighborhood buffers.
    """

    def __init__(self, filepath: str):
        if not isinstance(filepath, str) or not filepath.strip():
            raise TypeError("Video filepath must be a non-empty string.")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found at: {filepath}")

        if not os.path.isfile(filepath):
            raise ValueError(f"Target path is not a file: {filepath}")

        if os.path.getsize(filepath) == 0:
            raise ValueError(f"Video file is empty (0 bytes): {filepath}")

        self.filepath = filepath
        self._cap = cv2.VideoCapture(filepath)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {filepath}")

        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self._cap.release()
            raise ValueError(f"Video file contains zero decodable frames: {filepath}")

        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 25.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

        # Small LRU frame cache to optimize neighbor seeking
        self._cache = {}
        self._cache_capacity = 30

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            filepath=self.filepath,
            total_frames=self.total_frames,
            fps=round(self.fps, 2),
            duration_sec=round(self.duration_sec, 2),
            width=self.width,
            height=self.height
        )

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Retrieves a specific frame by index (0-indexed).
        """
        if not isinstance(frame_index, int):
            raise TypeError(f"frame_index must be an integer, got {type(frame_index).__name__}")

        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        if self._cap is None or not self._cap.isOpened():
            return None

        if frame_index in self._cache:
            return self._cache[frame_index].copy()

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        # Add to cache
        if len(self._cache) >= self._cache_capacity:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[frame_index] = frame.copy()

        return frame

    def get_temporal_neighborhood(
        self,
        target_index: int,
        radius: int = 2
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Retrieves neighboring frames within [target_index - radius, target_index + radius].
        Excludes the target frame itself.

        Returns:
            List of (frame_index, frame_bgr) tuples.
        """
        neighbors = []
        start_idx = max(0, target_index - radius)
        end_idx = min(self.total_frames - 1, target_index + radius)

        for idx in range(start_idx, end_idx + 1):
            if idx == target_index:
                continue
            f = self.get_frame(idx)
            if f is not None:
                neighbors.append((idx, f))

        return neighbors

    def iterate_frames(
        self,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (frame_index, timestamp_sec, frame_bgr) sequentially.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        curr_idx = 0
        yielded_count = 0

        while True:
            if max_frames is not None and yielded_count >= max_frames:
                break
            if curr_idx >= self.total_frames:
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            if curr_idx % stride == 0:
                timestamp = curr_idx / self.fps
                yield (curr_idx, timestamp, frame)
                yielded_count += 1

            curr_idx += 1

    def release(self) -> None:
        """Closes video capture handle."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
