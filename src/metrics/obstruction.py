"""
Multi-indicator camera obstruction detection and temporal persistence tracking.
Distinguishes transient occlusions (wipers, passing leaves) from persistent lens deposits.
"""

from typing import Dict, Any, List, Optional
from collections import deque
import cv2
import numpy as np


def compute_spatial_obstruction(
    image: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    edge_sparsity_threshold: float = 8.0,
    entropy_threshold: float = 3.2
) -> Dict[str, Any]:
    """
    Evaluates spatial camera obstruction indicators across a grid of local cells.

    Indicators measured:
    1. Local Sobel gradient density (edge sparsity).
    2. Local Shannon intensity entropy.
    3. Global frame darkness/whiteness or texture collapse.

    Args:
        image: Input image (BGR or Grayscale uint8 array).
        grid_rows: Number of grid cell rows.
        grid_cols: Number of grid cell columns.
        edge_sparsity_threshold: Mean gradient magnitude below which cell has no texture.
        entropy_threshold: Shannon entropy below which cell is uniform.

    Returns:
        dict containing:
            - obstructed_cell_ratio (float): Ratio of grid cells flagged as flat/obstructed.
            - obstructed_cell_mask (np.ndarray): Boolean mask of shape (grid_rows, grid_cols).
            - global_flatness_score (float): Overall flatness measure [0.0 - 1.0].
            - edge_density (float): Average edge magnitude across image.
            - is_instantaneous_obstruction (bool): True if candidate obstruction in this frame.
    """
    if image is None or image.size == 0:
        return {
            "obstructed_cell_ratio": 1.0,
            "obstructed_cell_mask": np.ones((grid_rows, grid_cols), dtype=bool),
            "global_flatness_score": 1.0,
            "edge_density": 0.0,
            "is_instantaneous_obstruction": True
        }

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)

    overall_edge_density = float(np.mean(grad_mag))

    cell_h = h // grid_rows
    cell_w = w // grid_cols

    obstructed_cells = np.zeros((grid_rows, grid_cols), dtype=bool)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h if r < grid_rows - 1 else h
            x1, x2 = c * cell_w, (c + 1) * cell_w if c < grid_cols - 1 else w

            cell_grad = grad_mag[y1:y2, x1:x2]
            cell_gray = gray[y1:y2, x1:x2]

            mean_cell_grad = float(np.mean(cell_grad))

            # Shannon entropy of cell
            hist, _ = np.histogram(cell_gray, bins=32, range=(0, 256), density=True)
            hist = hist[hist > 0]
            cell_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

            # Cell is considered obstructed if it has no edges AND minimal entropy.
            # Upper cells (sky horizon) with bright luminance are natural open sky, not lens deposits.
            mean_intensity = float(np.mean(cell_gray))
            is_open_sky = (r <= 1 and mean_intensity > 110)

            if mean_cell_grad < edge_sparsity_threshold and cell_entropy < entropy_threshold and not is_open_sky:
                obstructed_cells[r, c] = True

    total_cells = grid_rows * grid_cols
    obstructed_count = int(np.count_nonzero(obstructed_cells))
    obstructed_cell_ratio = float(obstructed_count / total_cells)

    # Global flatness: high when edge density is very low or vast majority of cells are blocked
    global_flatness_score = float(np.clip(
        0.6 * obstructed_cell_ratio + 0.4 * (1.0 - np.clip(overall_edge_density / 30.0, 0.0, 1.0)),
        0.0, 1.0
    ))

    # Instantaneous trigger: require >= 25% of valid non-sky cells to be blocked
    # or extreme global flatness collapse
    is_instantaneous = bool(obstructed_cell_ratio >= 0.25 or global_flatness_score > 0.75)

    return {
        "obstructed_cell_ratio": round(obstructed_cell_ratio, 4),
        "obstructed_cell_mask": obstructed_cells,
        "global_flatness_score": round(global_flatness_score, 4),
        "edge_density": round(overall_edge_density, 2),
        "is_instantaneous_obstruction": is_instantaneous
    }


class TemporalObstructionFilter:
    """
    Temporal tracker that accumulates spatial obstruction masks across sequential video frames.
    Distinguishes transient occlusions from persistent deposits or sensor failure.
    """

    def __init__(
        self,
        history_len: int = 5,
        min_persistence_frames: int = 3,
        grid_rows: int = 4,
        grid_cols: int = 4
    ):
        self.history_len = history_len
        self.min_persistence_frames = min_persistence_frames
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.history: deque = deque(maxlen=history_len)
        self.instantaneous_history: deque = deque(maxlen=history_len)

    def reset(self) -> None:
        """Resets the temporal history."""
        self.history.clear()
        self.instantaneous_history.clear()

    def update(self, spatial_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the tracker with current frame's spatial obstruction result.

        Returns:
            dict containing:
                - is_persistent_obstruction (bool): True if obstruction persists across frames.
                - persistent_cell_ratio (float): Ratio of cells persistently blocked.
                - persistence_streak (int): Number of consecutive frames flagged.
                - temporal_confidence (float): Confidence score [0.0 - 1.0].
                - is_transient (bool): True if flagged in 1 frame but not yet persistent.
        """
        mask = spatial_result.get("obstructed_cell_mask", np.zeros((self.grid_rows, self.grid_cols), dtype=bool))
        is_inst = spatial_result.get("is_instantaneous_obstruction", False)

        self.history.append(mask)
        self.instantaneous_history.append(is_inst)

        # Count consecutive instantaneous triggers from end of history
        streak = 0
        for val in reversed(self.instantaneous_history):
            if val:
                streak += 1
            else:
                break

        # Spatial consistency across history
        if len(self.history) > 0:
            stacked_masks = np.stack(list(self.history), axis=0)  # Shape: (N, R, C)
            cell_counts = np.sum(stacked_masks, axis=0)
            persistent_mask = cell_counts >= min(len(self.history), self.min_persistence_frames)
            persistent_cell_ratio = float(np.count_nonzero(persistent_mask) / (self.grid_rows * self.grid_cols))
        else:
            persistent_cell_ratio = 0.0

        # Persistent obstruction decision: requires current frame to be an instantaneous candidate
        # AND have sustained streak or persistent cell clustering
        is_persistent = bool(
            is_inst and (
                streak >= self.min_persistence_frames or
                (len(self.history) >= self.min_persistence_frames and persistent_cell_ratio >= 0.35)
            )
        )

        temporal_confidence = float(np.clip(
            0.5 * (streak / max(1, self.min_persistence_frames)) + 0.5 * persistent_cell_ratio,
            0.0, 1.0
        )) if is_inst else 0.0

        is_transient = bool(is_inst and not is_persistent)

        return {
            "is_persistent_obstruction": is_persistent,
            "persistent_cell_ratio": round(persistent_cell_ratio, 4),
            "persistence_streak": streak,
            "temporal_confidence": round(temporal_confidence, 4),
            "is_transient": is_transient,
            "history_depth": len(self.history)
        }
