"""
PhantomRoad: Adaptive Vision Reliability Agent
Stage 1 Operations Dashboard: Video Ingestion & Optical Quality Telemetry.
OpenCV AI Competition 2026.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui.styles import DARK_THEME_CSS
from src.ui.components import (
    render_stage1_header,
    render_brand_header,
    render_stage1_metrics_cards,
    build_stage1_histogram_chart,
    build_stage1_batch_timeline,
    render_decision_hud,
    render_detector_offline_alert,
    render_stage3_detector_view,
    render_perception_comparison_view
)
from src.video.frame_extractor import VideoLoader, VideoMetadata, validate_video_path
from src.video.synthetic_generator import generate_synthetic_road_video, AVAILABLE_SCENARIOS
from src.metrics.blur import compute_blur_metrics, compute_laplacian_map
from src.metrics.illumination import compute_illumination_metrics, compute_intensity_histogram
from src.agent.action_executor import AdaptiveReliabilityAgent
from src.agent.decision_engine import ReliabilityDecisionEngine
from src.detector import (
    OpenCVDNNDetector,
    ReliableRoadSceneAnalyzer,
    RoadScenePerceptionComparator,
    ROAD_CLASSES
)

# Streamlit Page Setup
st.set_page_config(
    page_title="PhantomRoad — Adaptive Vision Reliability Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom Dark Operations CSS
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_road_scene_detector() -> OpenCVDNNDetector:
    """Cached initialization of OpenCV DNN road scene object detector."""
    return OpenCVDNNDetector()


def main():
    # Session state initialization
    if "current_video_path" not in st.session_state:
        st.session_state.current_video_path = None
    if "current_video_name" not in st.session_state:
        st.session_state.current_video_name = None
    if "current_video_meta" not in st.session_state:
        st.session_state.current_video_meta = None
    if "batch_analysis_records" not in st.session_state:
        st.session_state.batch_analysis_records = None
    if "current_frame_idx" not in st.session_state:
        st.session_state.current_frame_idx = 0

    # Initialize Stage 3 detector and analyzers
    detector = get_road_scene_detector()
    analyzer = ReliableRoadSceneAnalyzer(detector)
    comparator = RoadScenePerceptionComparator(detector)

    # Top Brand Header with detector & cloud telemetry
    render_brand_header(
        cloud_status={"enabled": False},
        detector_available=detector.is_available()
    )

    # -------------------------------------------------------------
    # SIDEBAR: Video Ingestion & Threshold Controls
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ Video Ingestion (Stage 1)")

        source_mode = st.radio(
            "Video Source:",
            ["Preset Road Scenario", "Upload Video File"],
            help="Select an automated benchmark scenario or upload your own road video."
        )

        if source_mode == "Upload Video File":
            st.markdown("#### 📁 File Upload & Validation")
            uploaded_file = st.file_uploader(
                "Upload Road Video:",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Supports MP4, AVI, MOV, MKV, WEBM containers up to 500MB."
            )

            if uploaded_file is not None:
                # File size check
                file_bytes = uploaded_file.getvalue()
                if len(file_bytes) == 0:
                    st.error("Uploaded file is empty (0 bytes). Please upload a valid video file.")
                elif len(file_bytes) > 500 * 1024 * 1024:
                    st.error("Uploaded file exceeds 500MB limit.")
                else:
                    uploads_dir = os.path.join("data", "uploads")
                    os.makedirs(uploads_dir, exist_ok=True)
                    saved_path = os.path.join(uploads_dir, uploaded_file.name)

                    with open(saved_path, "wb") as f:
                        f.write(file_bytes)

                    # Validate video file stream using validate_video_path
                    is_valid, msg, meta = validate_video_path(saved_path)
                    if is_valid and meta is not None:
                        if st.session_state.current_video_path != saved_path:
                            st.session_state.current_video_path = saved_path
                            st.session_state.current_video_name = uploaded_file.name
                            st.session_state.current_video_meta = meta
                            st.session_state.batch_analysis_records = None
                            st.session_state.current_frame_idx = 0
                        st.success(f"✓ Video Validated: {uploaded_file.name}")
                    else:
                        st.error(f"Validation Failed: {msg}")

        else:
            st.markdown("#### 🧪 Benchmark Scenario Generator")
            scenario_choice = st.selectbox(
                "Select Road Scenario:",
                list(AVAILABLE_SCENARIOS.keys()),
                format_func=lambda x: f"{x.replace('_', ' ').title()}",
                help="Generates synthetic real road conditions for deterministic testing."
            )
            st.caption(f"_{AVAILABLE_SCENARIOS[scenario_choice]}_")

            num_frames = st.slider("Scenario Frames:", min_value=15, max_value=60, value=30, step=5)

            if st.button("Generate & Load Scenario", use_container_width=True, type="primary"):
                with st.spinner(f"Rendering {scenario_choice} video clip..."):
                    save_dir = os.path.join("data", "samples")
                    os.makedirs(save_dir, exist_ok=True)
                    target_path = os.path.join(save_dir, f"{scenario_choice}.mp4")
                    generate_synthetic_road_video(
                        scenario=scenario_choice,
                        output_path=target_path,
                        num_frames=num_frames,
                        fps=20
                    )
                    is_valid, msg, meta = validate_video_path(target_path)
                    if is_valid and meta is not None:
                        st.session_state.current_video_path = target_path
                        st.session_state.current_video_name = f"Scenario: {scenario_choice}"
                        st.session_state.current_video_meta = meta
                        st.session_state.batch_analysis_records = None
                        st.session_state.current_frame_idx = 0
                        st.success("Scenario clip loaded successfully!")
                    else:
                        st.error(f"Scenario generation error: {msg}")

        st.markdown("---")
        st.markdown("### ⚙️ Optical Quality Thresholds")
        blur_th = st.slider(
            "Blur Threshold (Laplacian Var):",
            min_value=30.0,
            max_value=200.0,
            value=85.0,
            step=5.0,
            help="Frames with Laplacian variance below this value are flagged as blurred."
        )
        low_light_th = st.slider(
            "Low-Light Threshold (Luminance):",
            min_value=15.0,
            max_value=80.0,
            value=45.0,
            step=5.0,
            help="Frames with mean perceptual luminance below this value are flagged as dark/low-light."
        )
        overexp_th = st.slider(
            "Overexposed Threshold (Luminance):",
            min_value=150.0,
            max_value=240.0,
            value=200.0,
            step=5.0,
            help="Frames with mean luminance above this value are flagged as overexposed."
        )
        min_contrast_th = st.slider(
            "Min RMS Contrast Threshold:",
            min_value=0.05,
            max_value=0.30,
            value=0.12,
            step=0.01,
            help="Frames with RMS contrast below this value are flagged as low contrast."
        )

        st.markdown("---")
        st.markdown("### 🚗 Road-Scene Object Detector (Stage 3)")
        if detector.is_available():
            st.success(f"✓ Detector Online: {detector.model_name}")
        else:
            st.warning("⚠️ Model Offline (Run `python models/download_models.py`)")

        conf_th = st.slider(
            "Detection Confidence Threshold:",
            min_value=0.15,
            max_value=0.85,
            value=0.35,
            step=0.05,
            help="Minimum confidence threshold for MobileNet-SSD bounding box detections."
        )

        st.markdown("---")
        st.markdown("### ☁️ Cloud Integration Status")
        st.info("Stage 1-3 Scope: 100% Offline Local Processing. AWS S3 synchronization is deferred to Stage 5.")

    # -------------------------------------------------------------
    # MAIN DASHBOARD AREA
    # -------------------------------------------------------------
    video_path = st.session_state.current_video_path
    video_name = st.session_state.current_video_name
    video_meta = st.session_state.current_video_meta

    if not video_path or not os.path.exists(video_path):
        st.info("👋 Select or generate a video scenario from the sidebar to begin Stage 1 optical telemetry analysis.")
        return

    # Tabs for Stages 1, 2, and 3
    tab_frame, tab_stage2, tab_stage3, tab_timeline, tab_docs = st.tabs([
        "🎬 Frame Extraction & Optical Metrics",
        "🛡️ Stage 2: Adaptive Decision Engine & Closed-Loop",
        "🚗 Stage 3: Road-Scene Detector & Reliability",
        "📈 Full Video Batch Analysis & Export",
        "ℹ️ Architecture & Documentation"
    ])

    # -------------------------------------------------------------
    # TAB 1: Frame Extraction & Optical Metrics
    # -------------------------------------------------------------
    with tab_frame:
        try:
            with VideoLoader(video_path) as loader:
                meta = loader.metadata

                # Video Stream Metadata Cards
                c_meta1, c_meta2, c_meta3, c_meta4, c_meta5 = st.columns(5)
                c_meta1.metric("Video Source", video_name)
                c_meta2.metric("Resolution", f"{meta.width} × {meta.height}")
                c_meta3.metric("Total Frames", meta.total_frames)
                c_meta4.metric("Frame Rate", f"{meta.fps:.1f} FPS")
                c_meta5.metric("Duration", f"{meta.duration_sec:.1f}s")

                st.markdown("---")

                # Frame Scrubber Controls
                sc_col1, sc_col2 = st.columns([3, 1])
                with sc_col1:
                    max_idx = max(0, meta.total_frames - 1)
                    frame_idx = st.slider(
                        "Scrub Video Frames:",
                        min_value=0,
                        max_value=max_idx,
                        value=min(st.session_state.current_frame_idx, max_idx),
                        key="frame_slider"
                    )
                    st.session_state.current_frame_idx = frame_idx

                with sc_col2:
                    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                    btn_prev, btn_next = st.columns(2)
                    if btn_prev.button("◀ Prev", use_container_width=True):
                        st.session_state.current_frame_idx = max(0, frame_idx - 1)
                        st.rerun()
                    if btn_next.button("Next ▶", use_container_width=True):
                        st.session_state.current_frame_idx = min(max_idx, frame_idx + 1)
                        st.rerun()

                # Extract Target Frame
                frame_bgr = loader.get_frame(frame_idx)
                timestamp = frame_idx / meta.fps if meta.fps > 0 else 0.0

                if frame_bgr is None:
                    st.error(f"Failed to decode frame #{frame_idx} from stream.")
                    return

                # Compute Real Metrics on Extracted Frame
                blur_metrics = compute_blur_metrics(frame_bgr, blur_threshold=blur_th)
                illum_metrics = compute_illumination_metrics(
                    frame_bgr,
                    low_light_threshold=low_light_th,
                    overexposed_threshold=overexp_th,
                    min_rms_contrast=min_contrast_th
                )
                laplacian_map = compute_laplacian_map(frame_bgr)
                hist_data = compute_intensity_histogram(frame_bgr)

                # Real Telemetry Cards
                st.markdown(f"#### 📊 Real-Time Telemetry — Frame #{frame_idx} `(t = {timestamp:.2f}s)`")
                render_stage1_metrics_cards(blur_metrics, illum_metrics)

                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

                # Visual Frame Display + Edge Map
                vis_col1, vis_col2 = st.columns(2)

                with vis_col1:
                    st.markdown(f"**Extracted Frame #{frame_idx} (RGB)**")
                    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    st.image(rgb_frame, use_container_width=True)

                with vis_col2:
                    st.markdown(f"**High-Frequency Laplacian Edge Map** (Var: `{blur_metrics['laplacian_variance']}`)")
                    st.image(laplacian_map, use_container_width=True, clamp=True)
                    st.caption("_Reveals structural high-frequency edge magnitudes used by the Pech-Pacheco variance formulation._")

                # Intensity Distribution Histogram
                st.markdown("#### 📉 Pixel Luminance & Color Channel Distribution")
                hist_chart = build_stage1_histogram_chart(hist_data)
                st.altair_chart(hist_chart, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading video stream: {e}")

    # -------------------------------------------------------------
    # TAB 2: Stage 2: Adaptive Decision Engine & Closed-Loop
    # -------------------------------------------------------------
    with tab_stage2:
        st.markdown("### 🛡️ Stage 2: Adaptive Reliability Decision Engine & Action Controller")
        st.caption(
            "Closed-loop decision engine that analyzes optical telemetry, dynamically selects corrective actions "
            "(neighbor inspection, unsharp mask, gamma correction, CLAHE, dehaze), recalculates metrics after each attempt, "
            "compares against the original frame, and escalates unresolved degradation to human review."
        )

        try:
            with VideoLoader(video_path) as loader:
                meta = loader.metadata
                frame_idx = st.session_state.current_frame_idx
                frame_bgr = loader.get_frame(frame_idx)
                timestamp = frame_idx / meta.fps if meta.fps > 0 else 0.0

                if frame_bgr is not None:
                    # Agent Configuration
                    c_cfg1, c_cfg2 = st.columns([1, 2])
                    with c_cfg1:
                        max_attempts = st.slider(
                            "Max Corrective Attempts:",
                            min_value=1,
                            max_value=5,
                            value=3,
                            key="s2_max_attempts",
                            help="Maximum number of verifiable actions the agent may attempt before escalating to human review."
                        )
                    with c_cfg2:
                        st.info(
                            f"**Active Operational Policy:** Frame #{frame_idx} `(t = {timestamp:.2f}s)`. "
                            "Stop condition triggers upon achieving composite quality score &ge; 0.65 or exhausting attempts."
                        )

                    # Initialize Stage 2 Decision Engine with active thresholds
                    engine = ReliabilityDecisionEngine(
                        blur_laplacian_threshold=blur_th,
                        low_light_threshold=low_light_th,
                        overexposed_threshold=overexp_th,
                        min_rms_contrast=min_contrast_th
                    )
                    agent = AdaptiveReliabilityAgent(
                        decision_engine=engine,
                        max_attempts=max_attempts
                    )

                    # Execute the closed-loop agent on the current frame
                    decision = agent.process_frame(
                        frame_bgr=frame_bgr,
                        frame_index=frame_idx,
                        timestamp_sec=timestamp,
                        video_loader=loader
                    )

                    # Render Stage 2 Decision HUD (verdict, stop condition, original comparison KPIs, action log)
                    render_decision_hud(decision)

                    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

                    # Visual Comparison: Original vs Final Adapted Frame
                    st.markdown("#### 🔄 Visual Frame Comparison: Original Input vs Adaptive Final Output")
                    col_orig, col_final = st.columns(2)
                    final_frame = decision.final_frame_bgr if decision.final_frame_bgr is not None else frame_bgr

                    with col_orig:
                        st.markdown(f"**Original Raw Frame #{frame_idx}**")
                        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.caption(
                            f"Initial State: `{decision.initial_state.value}` | "
                            f"Composite: `{decision.initial_metrics.composite_quality_score:.3f}` | "
                            f"Sharpness: `{decision.initial_metrics.laplacian_variance:.1f}`"
                        )

                    with col_final:
                        enhanced_label = "Accepted Enhancement" if decision.was_enhanced else "Original Retained (Unchanged)"
                        st.markdown(f"**Final Frame #{frame_idx}** ({enhanced_label})")
                        st.image(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.caption(
                            f"Final State: `{decision.final_state.value}` | "
                            f"Composite: `{decision.final_metrics.composite_quality_score:.3f}` | "
                            f"Stop: `{decision.stop_condition}`"
                        )

                    # Difference Heatmap
                    diff = cv2.absdiff(frame_bgr, final_frame)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    heatmap = cv2.applyColorMap(np.clip(gray_diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

                    st.markdown("#### 🔥 Enhancement Impact Heatmap")
                    st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption("_Displays localized pixel delta intensity (amplified 4x) showing precisely where the agent applied software corrections._")

        except Exception as e:
            st.error(f"Stage 2 Decision Engine error: {e}")

    # -------------------------------------------------------------
    # TAB 3: Stage 3: Road-Scene Detector & Reliability Integration
    # -------------------------------------------------------------
    with tab_stage3:
        st.markdown("### 🚗 Stage 3: Pretrained Road-Scene Object Detector & Reliability Annotation")
        st.caption(
            "Deep neural network object detection (MobileNet-SSD via OpenCV DNN) integrated with closed-loop optical reliability. "
            "Detects road obstacles (cars, buses, motorbikes, bicycles, pedestrians) on actual video frames, evaluates evidence quality, "
            "and explicitly marks candidates as UNVERIFIED_DEGRADED or SUPPRESSED when visual evidence is compromised. "
            "Never invents or fabricates obstacle detections."
        )

        if not detector.is_available():
            render_detector_offline_alert()
            st.info(
                "💡 **Graceful Fallback Mode:** Notice that all Stage 1 optical extraction, video batch processing, "
                "and Stage 2 adaptive reliability decision engine actions remain 100% operational above."
            )
        else:
            s3_subtabs = st.tabs([
                "🎯 Active Frame Perception & Reliability HUD",
                "⚖️ Clear vs. Degraded Perception Benchmark",
                "📚 Model Setup & Documented Limitations"
            ])

            # SUBTAB 1: Active Frame Inference
            with s3_subtabs[0]:
                try:
                    with VideoLoader(video_path) as loader:
                        meta = loader.metadata
                        frame_idx = st.session_state.current_frame_idx
                        frame_bgr = loader.get_frame(frame_idx)
                        timestamp = frame_idx / meta.fps if meta.fps > 0 else 0.0

                        if frame_bgr is not None:
                            # Run decision engine to get reliability state
                            engine = ReliabilityDecisionEngine(
                                blur_laplacian_threshold=blur_th,
                                low_light_threshold=low_light_th,
                                overexposed_threshold=overexp_th,
                                min_rms_contrast=min_contrast_th
                            )
                            agent = AdaptiveReliabilityAgent(decision_engine=engine, max_attempts=3)
                            decision = agent.process_frame(
                                frame_bgr=frame_bgr,
                                frame_index=frame_idx,
                                timestamp_sec=timestamp,
                                video_loader=loader
                            )

                            st.markdown(f"#### 🔍 Live Frame #{frame_idx} Inference `(t = {timestamp:.2f}s)`")
                            active_target_frame = decision.final_frame_bgr if decision.final_frame_bgr is not None else frame_bgr
                            render_stage3_detector_view(
                                analyzer=analyzer,
                                frame_bgr=active_target_frame,
                                decision=decision,
                                conf_threshold=conf_th
                            )
                except Exception as e:
                    st.error(f"Stage 3 Inference Error: {e}")

            # SUBTAB 2: Clear vs Degraded Perception Benchmark
            with s3_subtabs[1]:
                st.markdown("#### ⚖️ Optical Degradation vs. Perception Reliability Benchmark")
                st.caption(
                    "Directly demonstrates how optical noise, fog, blur, and lighting attenuation impact downstream detection accuracy. "
                    "Validates that PhantomRoad identifies degraded conditions and flags unverified detections rather than overconfidently assuming safety."
                )

                bench_col1, bench_col2 = st.columns([1, 2])
                with bench_col1:
                    degradation_type = st.selectbox(
                        "Simulate Environmental Degradation:",
                        [
                            "Heavy Atmospheric Fog (DCP Attenuation)",
                            "Severe Low-Light / Night Scene",
                            "High-Speed Motion / Defocus Blur",
                            "Overexposure / Direct Sun Glare",
                            "Physical Sensor Obstruction (Mud Splash)"
                        ]
                    )
                    degrade_intensity = st.slider(
                        "Degradation Severity:",
                        min_value=0.2,
                        max_value=0.9,
                        value=0.6,
                        step=0.1
                    )

                try:
                    with VideoLoader(video_path) as loader:
                        frame_idx = st.session_state.current_frame_idx
                        clear_frame = loader.get_frame(frame_idx)
                        if clear_frame is None:
                            clear_frame = np.zeros((360, 480, 3), dtype=np.uint8)

                        # Synthesize realistic degradation on the active frame
                        degraded_frame = clear_frame.copy()
                        h, w = degraded_frame.shape[:2]

                        if "Fog" in degradation_type:
                            fog_val = int(180 + 50 * degrade_intensity)
                            fog_layer = np.full_like(degraded_frame, fog_val, dtype=np.uint8)
                            alpha = max(0.1, 1.0 - degrade_intensity)
                            degraded_frame = cv2.addWeighted(degraded_frame, alpha, fog_layer, 1.0 - alpha, 0)
                        elif "Low-Light" in degradation_type:
                            gamma = max(0.15, 1.0 - degrade_intensity * 0.9)
                            inv_gamma = 1.0 / gamma
                            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                            degraded_frame = cv2.LUT(degraded_frame, table)
                            degraded_frame = (degraded_frame * max(0.1, 1.0 - degrade_intensity)).astype(np.uint8)
                        elif "Blur" in degradation_type:
                            k = int(degrade_intensity * 35) // 2 * 2 + 1
                            k = max(3, k)
                            degraded_frame = cv2.GaussianBlur(degraded_frame, (k, k), sigmaX=k * 0.3)
                        elif "Overexposure" in degradation_type:
                            degraded_frame = np.clip(degraded_frame.astype(np.float32) * (1.0 + degrade_intensity * 2.0) + degrade_intensity * 80, 0, 255).astype(np.uint8)
                        elif "Obstruction" in degradation_type:
                            center_x, center_y = w // 2, h // 2
                            radius = int(min(w, h) * 0.35 * degrade_intensity)
                            cv2.circle(degraded_frame, (center_x, center_y), radius, (30, 40, 50), -1)
                            cv2.circle(degraded_frame, (center_x - 30, center_y + 20), radius // 2, (35, 45, 55), -1)

                        # Evaluate clear frame through agent
                        engine = ReliabilityDecisionEngine(
                            blur_laplacian_threshold=blur_th,
                            low_light_threshold=low_light_th,
                            overexposed_threshold=overexp_th,
                            min_rms_contrast=min_contrast_th
                        )
                        agent = AdaptiveReliabilityAgent(decision_engine=engine, max_attempts=3)
                        clear_dec = agent.process_frame(clear_frame, frame_index=frame_idx, timestamp_sec=0.0)
                        deg_dec = agent.process_frame(degraded_frame, frame_index=frame_idx, timestamp_sec=0.0)

                        # Run comparative evaluation
                        render_perception_comparison_view(
                            comparator=comparator,
                            clear_frame=clear_frame,
                            degraded_frame=degraded_frame,
                            enhanced_frame=deg_dec.final_frame_bgr if deg_dec.was_enhanced else None,
                            clear_decision=clear_dec,
                            degraded_decision=deg_dec,
                            enhanced_decision=deg_dec if deg_dec.was_enhanced else None,
                            conf_threshold=conf_th
                        )
                except Exception as e:
                    st.error(f"Benchmark Execution Error: {e}")

            # SUBTAB 3: Model Setup & Documented Limitations
            with s3_subtabs[2]:
                st.markdown("#### 📚 MobileNet-SSD Architecture & Setup Guide")
                st.markdown("""
                **Model Selection Rationale:**
                MobileNet-SSD is a lightweight deep neural network combining Depthwise Separable Convolutions with a Single Shot MultiBox Detector head. It provides real-time local CPU execution (~25–35 FPS) without requiring CUDA hardware or cloud APIs.
                
                **How to Obtain & Configure the Model:**
                ```powershell
                # Automatic downloader
                python models/download_models.py
                ```
                In Python code:
                ```python
                from src.detector import OpenCVDNNDetector, ReliableRoadSceneAnalyzer
                detector = OpenCVDNNDetector(
                    prototxt_path="models/MobileNetSSD_deploy.prototxt",
                    weights_path="models/MobileNetSSD_deploy.caffemodel"
                )
                analyzer = ReliableRoadSceneAnalyzer(detector)
                ```

                ---
                #### ⚠️ Documented Architectural Limitations
                1. **Target Class Domain (VOC 20 Scope):**
                   Trained on PASCAL VOC. Filtered to road obstacles (`car`, `bus`, `motorbike`, `bicycle`, `person`, `train`). It does not output traffic signs, lane markers, or specialized road debris.
                2. **Fixed Input Resolution ($300 \\times 300$):**
                   Requires downsampling to $300\\times300$ square blob. Ultrawide automotive camera aspect ratios (e.g., 16:9, 21:9) suffer non-uniform aspect scaling, impacting distant vehicle bounding box aspect accuracy.
                3. **Small Obstacle & Atmospheric Attenuation Sensitivity:**
                   Distant vehicles smaller than approximately $25\\times25$ pixels experience severe recall dropoff in fog, heavy rain, or underexposure.
                4. **Single-Frame Inference Decoupling:**
                   Operates purely frame-by-frame without temporal tracking (e.g., Kalman filtering / SORT). Transient occlusions can cause flicker between adjacent video frames.
                
                ---
                #### 🛡️ Zero-Hallucination Integrity Policy
                Safety-critical autonomous systems must **never invent detections** to mask sensor degradation or pass tests. When optical evidence quality fails, PhantomRoad's core philosophy is to flag detections as `UNVERIFIED_DEGRADED` or `SUPPRESSED_OBSTRUCTED`, ensuring upstream planners can safely down-regulate speed or transfer authority to human safety drivers.
                """)

    # -------------------------------------------------------------
    # TAB 4: Full Video Batch Analysis & Export
    # -------------------------------------------------------------
    with tab_timeline:
        st.markdown("### 📊 Automated Multi-Frame Optical Analysis")
        st.caption("Sequentially extracts and profiles every frame in the video to compute full-timeline stability metrics.")

        run_col1, run_col2 = st.columns([1, 3])
        with run_col1:
            run_batch = st.button("Run Full Video Quality Profiling", type="primary", use_container_width=True)

        if run_batch:
            try:
                with VideoLoader(video_path) as loader:
                    total_f = loader.total_frames
                    progress_bar = st.progress(0, text="Analyzing video frames...")
                    records = []
                    t_start = time.perf_counter()

                    for i, (f_idx, ts, f_bgr) in enumerate(loader.iterate_frames()):
                        b_m = compute_blur_metrics(f_bgr, blur_threshold=blur_th)
                        i_m = compute_illumination_metrics(
                            f_bgr,
                            low_light_threshold=low_light_th,
                            overexposed_threshold=overexp_th,
                            min_rms_contrast=min_contrast_th
                        )
                        records.append({
                            "Frame": f_idx,
                            "Timestamp (s)": round(ts, 3),
                            "Laplacian Variance": b_m["laplacian_variance"],
                            "Sharpness Score": b_m["sharpness_score"],
                            "Status Blur": "Blurred" if b_m["is_blurred"] else "Sharp",
                            "Mean Brightness": i_m["mean_brightness"],
                            "RMS Contrast": i_m["rms_contrast"],
                            "Dynamic Range": i_m["dynamic_range"],
                            "Underexposed (%)": round(i_m["underexposed_ratio"] * 100.0, 2),
                            "Overexposed (%)": round(i_m["overexposed_ratio"] * 100.0, 2),
                            "Entropy (bits)": i_m["entropy"],
                            "Status Lighting": "Low Light" if i_m["is_low_light"] else ("Overexposed" if i_m["is_overexposed"] else "Balanced"),
                            "Status Contrast": "Low Contrast" if i_m["is_low_contrast"] else "Nominal"
                        })
                        progress_bar.progress((i + 1) / total_f, text=f"Processed frame {i + 1} of {total_f}...")

                    t_elapsed = time.perf_counter() - t_start
                    st.session_state.batch_analysis_records = records
                    st.success(f"Analysis complete! Profiled {total_f} frames in {t_elapsed:.2f}s ({total_f / max(1e-3, t_elapsed):.1f} FPS).")
            except Exception as e:
                st.error(f"Batch processing error: {e}")

        if st.session_state.batch_analysis_records:
            records = st.session_state.batch_analysis_records
            df_records = pd.DataFrame(records)

            # KPI Summary Metrics
            total_n = len(df_records)
            blurred_n = len(df_records[df_records["Status Blur"] == "Blurred"])
            low_light_n = len(df_records[df_records["Status Lighting"] == "Low Light"])
            low_contrast_n = len(df_records[df_records["Status Contrast"] == "Low Contrast"])

            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            kpi_c1.metric(
                "Mean Sharpness",
                f"{df_records['Laplacian Variance'].mean():.1f}",
                f"Min: {df_records['Laplacian Variance'].min():.1f}"
            )
            kpi_c2.metric(
                "Mean Brightness",
                f"{df_records['Mean Brightness'].mean():.1f}",
                f"Range: {df_records['Mean Brightness'].min():.1f} - {df_records['Mean Brightness'].max():.1f}"
            )
            kpi_c3.metric(
                "Mean RMS Contrast",
                f"{df_records['RMS Contrast'].mean():.3f}"
            )
            kpi_c4.metric(
                "Degradation Flags",
                f"{blurred_n + low_light_n + low_contrast_n} frames",
                f"Blur: {blurred_n} | Dark: {low_light_n} | LowCont: {low_contrast_n}"
            )

            st.markdown("---")
            st.markdown("#### 📉 Multi-Metric Temporal Timeline")
            timeline_chart = build_stage1_batch_timeline(records)
            st.altair_chart(timeline_chart, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📥 Export Stage 1 Telemetry Data")
            csv_data = df_records.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Telemetry (CSV)",
                data=csv_data,
                file_name=f"phantomroad_stage1_telemetry_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # -------------------------------------------------------------
    # TAB 3: Stage 1 Architecture & CV Formulations
    # -------------------------------------------------------------
    with tab_docs:
        st.markdown("""
        ### 📐 Stage 1 Computer Vision Mathematical Formulations

        #### 1. Pech-Pacheco Laplacian Variance (Sharpness / Blur Estimation)
        Measures the second spatial derivative of the grayscale frame intensity $I$:
        $$\\text{Var}(\\nabla^2 I) = \\frac{1}{N} \\sum_{x, y} (\\nabla^2 I(x, y) - \\mu)^2$$
        - High variance indicates steep transitions and sharp edges.
        - Low variance ($< 85.0$) signifies optical defocus or camera motion blur.

        #### 2. 2D-FFT High-Frequency Ratio
        Evaluates the two-dimensional Fourier spectrum of the frame:
        $$\\text{Ratio}_{\\text{high}} = \\frac{\\sum_{r > r_0} |F(u, v)|}{\\sum |F(u, v)|}$$
        Provides frequency-domain corroboration of spatial edge sharpness.

        #### 3. Perceptual Luminance & Illumination Profiling
        Extracts perceptual luminance $Y$ via the $YCrCb$ color space:
        $$Y = 0.299 R + 0.587 G + 0.114 B$$
        Frames with mean $Y < 45.0$ are classified as low-light; frames with mean $Y > 200.0$ are classified as overexposed.

        #### 4. Root Mean Square (RMS) Contrast
        Standard deviation of pixel luminance normalized to $[0.0, 1.0]$:
        $$C_{\\text{RMS}} = \\frac{1}{255} \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (Y_i - \\bar{Y})^2}$$
        Measures dynamic tone separation essential for road-boundary and lane detection.

        #### 5. Shannon Intensity Entropy
        Quantifies information density in the intensity distribution:
        $$H = -\\sum_{k=0}^{255} p_k \\log_2(p_k)$$

        ---

        ### 🛡️ Stage 2 Closed-Loop Reliability & Adaptive Action Formulations

        #### 1. Composite Optical Reliability Score
        Aggregates multi-spectral optical telemetry into a unified $[0.0, 1.0]$ confidence index:
        $$Q(I) = w_s \\cdot S(I) + w_c \\cdot C(I) + w_l \\cdot L(I) + w_t \\cdot T(I)$$
        where $S(I)$ is normalized sigmoid sharpness, $C(I)$ is dynamic tone contrast, $L(I)$ is exposure balance, and $T(I)$ is dark channel prior transmission. Operational reliability requires $Q(I) \\ge 0.65$.

        #### 2. Quantitative Verification & Rollback Rule
        Every candidate action $A_k$ is verified against preceding metrics and original unenhanced measurements:
        $$\\text{Accept}(A_k) = \\begin{cases} \\text{True} & \\text{if } \\Delta Q = Q(I_k) - Q(I_{k-1}) \\ge \\epsilon \\text{ and } \\text{CritCheck}(I_k) \\\\ \\text{False} & \\text{otherwise (Trigger Automatic Rollback)} \\end{cases}$$

        #### 3. Stop Conditions
        - `STOP_ALREADY_RELIABLE`: Input frame natively meets $Q(I) \\ge 0.65$ without degradations.
        - `STOP_RELIABLE_ACHIEVED`: Software action raised composite quality to nominal operating threshold.
        - `STOP_CRITICAL_OBSTRUCTION`: Physical sensor lens blockage or total blackout/whiteout detected.
        - `STOP_MAX_ATTEMPTS_EXHAUSTED`: Reached configured attempt cap without achieving reliable threshold; escalates to human review.
        - `STOP_NO_VIABLE_ACTIONS`: All valid software enhancement strategies exhausted without sufficient gain.

        ---

        ### 🚗 Stage 3 Road-Scene Detector & Perception Reliability Formulations

        #### 1. Perception Degradation Failure Modes
        Standard road-scene object detectors optimize Average Precision ($AP$) assuming nominal visibility distributions. Under atmospheric fog, lens water droplets, and low illumination:
        - **False Negatives (Missed Obstacles):** Contrast attenuation causes obstacle bounding box scores to drop below confidence threshold $\\tau_{\\text{conf}}$.
        - **False Positives (Spurious Activations):** High-frequency glare spikes or particulate noise trigger phantom activations.

        #### 2. Closed-Loop Reliability State Mapping
        PhantomRoad couples neural network detection outputs $D = \\{b_i, c_i, s_i\\}_{i=1}^M$ with the agent's verified optical reliability state $\\mathcal{S}$:
        $$\\text{Status}(b_i) = \\begin{cases} \\text{TRUSTED} & \\text{if } \\mathcal{S} = \\text{RELIABLE} \\\\ \\text{UNVERIFIED\\_DEGRADED} & \\text{if } \\mathcal{S} \\in \\{\\text{DEGRADED\\_CORRECTABLE}, \\text{DEGRADED\\_UNCERTAIN}\\} \\\\ \\text{SUPPRESSED\\_OBSTRUCTED} & \\text{if } \\mathcal{S} \\in \\{\\text{CRITICAL\\_OBSTRUCTION}, \\text{HUMAN\\_REVIEW\\_REQUIRED}\\} \\end{cases}$$

        #### 3. Zero-Hallucination Integrity
        Autonomous robotics safety mandates that perception systems must never synthesize fake bounding boxes to pass validation suites. When sensory evidence quality drops below reliable thresholds, flagging candidates as `UNVERIFIED_DEGRADED` provides upstream trajectory planners with actionable epistemic uncertainty.
        """)



if __name__ == "__main__":
    main()
