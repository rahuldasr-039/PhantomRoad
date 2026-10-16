"""
Reproducible synthetic road scene generator for automated evaluation and local testing.
Generates realistic perspective road clips under clear and degraded conditions.
"""

from typing import Dict, List, Optional
import os
import cv2
import numpy as np

AVAILABLE_SCENARIOS = {
    "clear_day": "Nominal clear daylight highway driving with road lanes and lead vehicle.",
    "motion_blur": "Transient camera vibration and motion blur induced on middle frames.",
    "night_low_light": "Night driving scenario with headlight illumination cone and dark surroundings.",
    "dense_fog": "Atmospheric scattering (fog/haze) causing low contrast and high DCP attenuation.",
    "oncoming_glare": "Blinding oncoming vehicle high-beams producing localized specular saturation.",
    "lens_obstruction": "Persistent foreign mud deposit occluding a portion of the camera lens."
}


def _draw_base_road_frame(
    frame_idx: int,
    total_frames: int,
    width: int = 640,
    height: int = 360
) -> np.ndarray:
    """
    Renders a realistic base road scene with perspective road, lane markings,
    and a moving vehicle ahead.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    horizon_y = int(height * 0.45)

    # 1. Sky with subtle gradient
    for y in range(horizon_y):
        ratio = y / horizon_y
        b = int(220 - 40 * ratio)
        g = int(180 - 30 * ratio)
        r = int(140 - 20 * ratio)
        frame[y, :] = [b, g, r]

    # Mountains on horizon
    for x in range(0, width, 4):
        h_m = int(25 * np.sin(x * 0.02) + 15 * np.cos(x * 0.05))
        top_y = max(0, horizon_y - h_m)
        frame[top_y:horizon_y, x:x+4] = [120, 100, 80]

    # 2. Road surface (asphalt with subtle texture)
    frame[horizon_y:, :] = [65, 65, 65]

    # Add realistic asphalt road texture noise
    rng = np.random.RandomState(42 + frame_idx)
    noise = rng.normal(0, 3, (height - horizon_y, width, 3)).astype(np.int16)
    road_slice = frame[horizon_y:, :].astype(np.int16) + noise
    frame[horizon_y:, :] = np.clip(road_slice, 0, 255).astype(np.uint8)

    # 3. Perspective Road Borders
    vanish_x = width // 2
    vanish_y = horizon_y

    left_bottom = (int(width * 0.10), height)
    right_bottom = (int(width * 0.90), height)
    vanish_pt = (vanish_x, vanish_y)

    # Road edges (white solid lines)
    cv2.line(frame, vanish_pt, left_bottom, (240, 240, 240), 4, cv2.LINE_AA)
    cv2.line(frame, vanish_pt, right_bottom, (240, 240, 240), 4, cv2.LINE_AA)

    # 4. Animated Dashed Yellow Center Line
    offset = (frame_idx * 12) % 60
    for y in range(horizon_y + 10, height, 40):
        y_start = y + offset
        if y_start >= height:
            continue
        y_end = min(height - 1, y_start + 20)

        # Interpolate x along center
        t1 = (y_start - vanish_y) / (height - vanish_y)
        t2 = (y_end - vanish_y) / (height - vanish_y)
        x1 = int(vanish_x)
        x2 = int(vanish_x)
        thick = max(2, int(6 * t2))
        cv2.line(frame, (x1, y_start), (x2, y_end), (20, 210, 255), thick, cv2.LINE_AA)

    # 5. Lead Vehicle moving smoothly ahead
    car_progress = 0.55 + 0.15 * np.sin(frame_idx * 0.1)
    car_y = int(horizon_y + (height - horizon_y) * car_progress)
    car_w = int(60 * car_progress)
    car_h = int(35 * car_progress)
    car_x = int(vanish_x + 35 * car_progress)

    # Car body (Red sedan)
    cv2.rectangle(frame, (car_x - car_w // 2, car_y - car_h), (car_x + car_w // 2, car_y), (40, 40, 200), -1)
    # Car roof / cabin
    roof_w = int(car_w * 0.7)
    cv2.rectangle(frame, (car_x - roof_w // 2, car_y - int(car_h * 1.5)), (car_x + roof_w // 2, car_y - car_h), (30, 30, 160), -1)
    # Taillights
    light_r = max(2, int(4 * car_progress))
    cv2.circle(frame, (car_x - car_w // 2 + 6, car_y - car_h // 2), light_r, (30, 30, 240), -1)
    cv2.circle(frame, (car_x + car_w // 2 - 6, car_y - car_h // 2), light_r, (30, 30, 240), -1)

    return frame


def generate_synthetic_road_video(
    scenario: str,
    output_path: str,
    num_frames: int = 40,
    fps: int = 20,
    width: int = 640,
    height: int = 360
) -> str:
    """
    Renders an MP4 video clip tailored to a specific degradation scenario.

    Args:
        scenario: One of 'clear_day', 'motion_blur', 'night_low_light',
                  'dense_fog', 'oncoming_glare', 'lens_obstruction'.
        output_path: Path to save the MP4 video.
        num_frames: Total number of frames to generate.
        fps: Frames per second.
        width: Video width.
        height: Video height.

    Returns:
        Absolute path to generated video.
    """
    if scenario not in AVAILABLE_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(AVAILABLE_SCENARIOS.keys())}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    horizon_y = int(height * 0.45)

    for i in range(num_frames):
        base = _draw_base_road_frame(i, num_frames, width=width, height=height)

        if scenario == "clear_day":
            processed = base

        elif scenario == "motion_blur":
            # Induce sharp transient blur on frames 12 through 24
            if 12 <= i <= 24:
                k_size = 25
                kernel = np.zeros((k_size, k_size))
                kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
                kernel /= k_size
                processed = cv2.filter2D(base, -1, kernel)
            else:
                processed = base

        elif scenario == "night_low_light":
            # Global darkening with vehicle headlight trapezoid
            darkened = (base * 0.12).astype(np.uint8)
            # Headlight cone mask
            cone_mask = np.zeros((height, width), dtype=np.float32)
            vanish_x = width // 2
            pts = np.array([
                [vanish_x - 30, horizon_y + 40],
                [vanish_x + 30, horizon_y + 40],
                [width - 60, height],
                [60, height]
            ], dtype=np.int32)
            cv2.fillPoly(cone_mask, [pts], 1.0)
            cone_mask = cv2.GaussianBlur(cone_mask, (55, 55), 25.0)
            cone_3ch = np.repeat(cone_mask[:, :, np.newaxis], 3, axis=2)
            # Blend illuminated road with dark ambient
            illuminated = (base * 0.70).astype(np.uint8)
            processed = (illuminated.astype(np.float32) * cone_3ch +
                         darkened.astype(np.float32) * (1.0 - cone_3ch)).astype(np.uint8)

        elif scenario == "dense_fog":
            # Atmospheric scattering simulation
            fog_color = np.array([215, 210, 205], dtype=np.float32)
            # Depth transmission decreases towards horizon
            y_coords = np.arange(height)[:, np.newaxis]
            dist_factor = np.clip((height - y_coords) / (height - horizon_y + 1e-5), 0.15, 1.0)
            trans_map = np.clip(0.18 + 0.50 * (1.0 - dist_factor), 0.15, 0.65)
            trans_3ch = np.repeat(trans_map[:, :, np.newaxis], width, axis=1)

            processed = (base.astype(np.float32) * trans_3ch +
                         fog_color * (1.0 - trans_3ch)).astype(np.uint8)

        elif scenario == "oncoming_glare":
            processed = base.copy()
            # Oncoming car high beams approaching on left lane
            glare_x = int(width * 0.35 - i * 3)
            glare_y = int(horizon_y + 40 + i * 2)
            glare_radius = int(25 + i * 1.5)

            # High intensity white bloom
            overlay = processed.copy()
            cv2.circle(overlay, (glare_x, glare_y), glare_radius, (255, 255, 255), -1)
            cv2.circle(overlay, (glare_x + 35, glare_y), int(glare_radius * 0.9), (255, 255, 255), -1)
            overlay = cv2.GaussianBlur(overlay, (45, 45), 20.0)
            processed = cv2.addWeighted(processed, 0.4, overlay, 0.6, 0)

        elif scenario == "lens_obstruction":
            processed = base.copy()
            # Persistent mud splatter landing on lens starting at frame 6
            if i >= 6:
                mud_mask = np.zeros((height, width), dtype=np.uint8)
                # Irregular mud splat
                cv2.ellipse(mud_mask, (int(width * 0.60), int(height * 0.55)), (110, 75), 25, 0, 360, 255, -1)
                cv2.circle(mud_mask, (int(width * 0.68), int(height * 0.42)), 45, 255, -1)
                cv2.circle(mud_mask, (int(width * 0.52), int(height * 0.68)), 40, 255, -1)
                mud_mask = cv2.GaussianBlur(mud_mask, (15, 15), 5.0)

                mud_color = np.full_like(processed, [38, 48, 42], dtype=np.uint8)
                mask_norm = mud_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
                processed = (mud_color.astype(np.float32) * mask_norm +
                             processed.astype(np.float32) * (1.0 - mask_norm)).astype(np.uint8)

        out.write(processed)

    out.release()
    return os.path.abspath(output_path)
