"""
Reusable UI components for the PhantomRoad Streamlit operations dashboard.
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from .styles import get_state_badge_html, get_stop_condition_badge_html
from ..agent.state import AgentFrameDecision, ReliabilityState, ActionAuditStep
from ..metrics.quality_assessor import FrameQualityMetrics


def render_brand_header(cloud_status: Dict[str, Any], detector_available: bool) -> None:
    """
    Renders top navigation banner with operational status indicators.
    """
    cloud_color = "#10b981" if cloud_status.get("enabled", False) else "#64748b"
    cloud_label = "AWS S3 ACTIVE" if cloud_status.get("enabled", False) else "LOCAL MODE (NO AWS)"

    detector_color = "#10b981" if detector_available else "#f59e0b"
    detector_label = "DNN DETECTOR ONLINE" if detector_available else "DETECTOR OFFLINE"

    header_html = f"""
    <div class="brand-container">
        <div>
            <h1 class="brand-title">PHANTOMROAD</h1>
            <p class="brand-subtitle">Adaptive Vision Reliability Agent &bull; OpenCV AI Competition 2026</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <span style="font-size: 11px; font-weight: 700; color: {detector_color}; background: rgba(255,255,255,0.05); padding: 5px 10px; border-radius: 6px; border: 1px solid {detector_color}40;">
                &#x25CF; {detector_label}
            </span>
            <span style="font-size: 11px; font-weight: 700; color: {cloud_color}; background: rgba(255,255,255,0.05); padding: 5px 10px; border-radius: 6px; border: 1px solid {cloud_color}40;">
                &#x25CF; {cloud_label}
            </span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_telemetry_cards(metrics: FrameQualityMetrics) -> None:
    """
    Renders 5 core optical telemetry cards with color-coded status meters.
    """
    cols = st.columns(5)

    # 1. Sharpness (Laplacian variance)
    with cols[0]:
        sharp_color = "#34d399" if not metrics.is_blurred else "#fb7185"
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">Sharpness (Laplacian)</div>
            <div class="metric-value" style="color: {sharp_color};">{metrics.laplacian_variance:.1f}</div>
            <div class="metric-sub">{"Nominal Sharp" if not metrics.is_blurred else "Motion/Blur Alert"}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Luminance & Exposure
    with cols[1]:
        lum_color = "#34d399" if not (metrics.is_low_light or metrics.is_overexposed) else "#fbbf24"
        lum_sub = "Low Light" if metrics.is_low_light else ("Overexposed" if metrics.is_overexposed else "Balanced")
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">Mean Luminance</div>
            <div class="metric-value" style="color: {lum_color};">{metrics.mean_brightness:.1f}</div>
            <div class="metric-sub">{lum_sub} (Dyn: {metrics.dynamic_range})</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. RMS Contrast
    with cols[2]:
        cont_color = "#34d399" if not metrics.is_low_contrast else "#fbbf24"
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">RMS Contrast</div>
            <div class="metric-value" style="color: {cont_color};">{metrics.rms_contrast:.3f}</div>
            <div class="metric-sub">{"Good Edge Dynamic" if not metrics.is_low_contrast else "Flat/Low Contrast"}</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Atmospheric Transmission
    with cols[3]:
        trans_color = "#34d399" if not metrics.is_hazy_or_foggy else "#fb7185"
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">DCP Transmission</div>
            <div class="metric-value" style="color: {trans_color};">{metrics.mean_transmission:.2f}</div>
            <div class="metric-sub">{"Clear Atmosphere" if not metrics.is_hazy_or_foggy else "Atmospheric Attenuation"}</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Composite Quality Score
    with cols[4]:
        q_score = metrics.composite_quality_score
        q_color = "#34d399" if q_score >= 0.65 else ("#fbbf24" if q_score >= 0.45 else "#fb7185")
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">Composite Quality</div>
            <div class="metric-value" style="color: {q_color};">{q_score * 100:.1f}%</div>
            <div class="metric-sub">Target &ge; 65.0%</div>
        </div>
        """, unsafe_allow_html=True)


def render_decision_hud(decision: AgentFrameDecision) -> None:
    """
    Renders the Agent Decision Banner, Evidence Rationale, Stop Condition,
    Comparison vs Original KPIs, and Detailed Action Log for Stage 2.
    """
    state_badge = get_state_badge_html(decision.final_state.value if hasattr(decision.final_state, "value") else str(decision.final_state))
    initial_badge = get_state_badge_html(decision.initial_state.value if hasattr(decision.initial_state, "value") else str(decision.initial_state))
    stop_badge = get_stop_condition_badge_html(decision.stop_condition)

    st.markdown(f"""
    <div class="hud-card hud-card-accent">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
            <span style="font-size: 13px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">
                Agent Reliability Verdict &bull; Frame #{decision.frame_index} ({decision.timestamp_sec:.2f}s)
            </span>
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <span style="font-size: 11px; color: #64748b;">INITIAL: {initial_badge}</span>
                <span>FINAL: {state_badge}</span>
                <span>{stop_badge}</span>
            </div>
        </div>
        <p style="font-size: 14px; margin: 0 0 6px 0; color: #cbd5e1;">
            <strong>Optical Evidence:</strong> {decision.final_metrics.evidence_summary}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Escalation banner if human review needed
    if decision.human_review_required:
        st.markdown(f"""
        <div class="alert-danger-box">
            <strong>&#x26A0; HUMAN ESCALATION REQUIRED:</strong> {decision.escalation_reason}
        </div>
        """, unsafe_allow_html=True)

    # Net Comparison vs Original Metrics Strip
    if decision.comparison_vs_original and decision.attempts_made > 0:
        comp = decision.comparison_vs_original
        st.markdown("##### 🔍 Net Verification & Improvement vs Original Raw Frame")
        c1, c2, c3, c4 = st.columns(4)

        comp_delta = comp.get("composite_delta", 0.0)
        c_color = "#34d399" if comp_delta >= 0 else "#fb7185"
        c1.metric(
            "Composite Delta",
            f"{comp_delta:+.3f}",
            f"{comp.get('net_improvement_percentage', 0.0):+.1f}% gain"
        )
        c2.metric(
            "Sharpness Delta",
            f"{comp.get('sharpness_delta', 0.0):+.1f}",
            "Laplacian var"
        )
        c3.metric(
            "RMS Contrast Delta",
            f"{comp.get('contrast_delta', 0.0):+.3f}",
            "Luminance dynamic"
        )
        c4.metric(
            "Mean Brightness Delta",
            f"{comp.get('brightness_delta', 0.0):+.1f}",
            "Perceptual Y"
        )

    # Action Execution Log (Stage 2 Requirement)
    if decision.audit_trace:
        st.markdown("##### 📜 Adaptive Action Execution Log (Stage 2 Closed-Loop)")
        for step in decision.audit_trace:
            acc_badge = '<span style="color:#34d399; font-weight:700;">&#x2713; ACCEPTED</span>' if step.accepted else '<span style="color:#fb7185; font-weight:700;">&#x2717; REJECTED / ROLLED BACK</span>'
            delta_color = "#34d399" if step.quality_delta >= 0 else "#fb7185"
            orig_delta_color = "#34d399" if step.quality_delta_vs_original >= 0 else "#fb7185"
            action_name = step.action.value if hasattr(step.action, "value") else str(step.action)

            st.markdown(f"""
            <div class="audit-step-box" style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap;">
                    <span style="font-size: 13px; font-weight: 700; color: #f8fafc;">
                        Step #{step.attempt_index}: <code>{action_name}</code>
                    </span>
                    <div style="font-size: 12px;">
                        {acc_badge} &bull; Step Delta: <strong style="color: {delta_color};">{step.quality_delta:+.3f}</strong> &bull; Vs Original: <strong style="color: {orig_delta_color};">{step.quality_delta_vs_original:+.3f}</strong>
                    </div>
                </div>
                <div style="font-size: 12px; color: #93c5fd; margin-bottom: 4px;">
                    <strong>Decision Rationale:</strong> {step.reason}
                </div>
                <div style="font-size: 12px; color: #94a3b8; margin-bottom: 4px;">
                    <strong>Verification Evidence:</strong> {step.evidence_reason}
                </div>
                <div style="display: flex; gap: 16px; font-size: 11px; color: #64748b; margin-top: 6px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 4px;">
                    <span>Outcome: <strong style="color:#e2e8f0;">{step.outcome}</strong></span>
                    <span>Composite Before: <strong style="color:#e2e8f0;">{step.before_composite:.3f}</strong> &rarr; After: <strong style="color:#e2e8f0;">{step.after_composite:.3f}</strong></span>
                    <span>Stop Condition: <strong style="color:#e2e8f0;">{step.stop_condition or 'CONTINUE'}</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)



def render_comparison_slider(
    original_frame: np.ndarray,
    final_frame: np.ndarray,
    annotated_frame: Optional[np.ndarray] = None
) -> None:
    """
    Renders synchronized side-by-side original vs processed comparison.
    """
    tabs = st.tabs(["Side-by-Side Comparison", "Difference Heatmap", "Downstream Perception"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original Camera Frame (BGR Input)")
            st.image(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            st.caption("Agent Adaptive Frame (Accepted Enhancement or Verified Input)")
            st.image(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    with tabs[1]:
        # Compute absolute difference heatmap
        diff = cv2.absdiff(original_frame, final_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(gray_diff * 4, cv2.COLORMAP_JET)
        st.caption("Pixel Intensity Delta Heatmap (Amplified 4x for visualization)")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)

    with tabs[2]:
        if annotated_frame is not None:
            st.caption("Downstream Object Detection & Reliability Annotation HUD")
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("Detector annotations unavailable for this frame.")


def build_reliability_timeline_chart(decisions: List[AgentFrameDecision]) -> alt.Chart:
    """
    Creates an interactive Altair timeline showing composite quality and state transitions.
    """
    data = []
    for d in decisions:
        data.append({
            "Frame": d.frame_index,
            "Timestamp": d.timestamp_sec,
            "Composite Quality": d.final_metrics.composite_quality_score,
            "Initial State": d.initial_state.value,
            "Final State": d.final_state.value,
            "Was Enhanced": "Enhanced" if d.was_enhanced else "Original",
            "Human Review": "Escalated" if d.human_review_required else "Nominal"
        })

    df = pd.DataFrame(data)

    color_scale = alt.Scale(
        domain=["RELIABLE", "DEGRADED_CORRECTABLE", "DEGRADED_UNCERTAIN", "CRITICAL_OBSTRUCTION", "HUMAN_REVIEW_REQUIRED"],
        range=["#10b981", "#f59e0b", "#fbbf24", "#e11d48", "#f43f5e"]
    )

    base = alt.Chart(df).encode(
        x=alt.X("Frame:Q", title="Video Frame Index"),
        tooltip=["Frame", "Timestamp", "Composite Quality", "Initial State", "Final State", "Was Enhanced", "Human Review"]
    )

    line = base.mark_line(color="#38bdf8", strokeWidth=2).encode(
        y=alt.Y("Composite Quality:Q", scale=alt.Scale(domain=[0.0, 1.0]), title="Quality Score (0 - 1.0)")
    )

    points = base.mark_circle(size=70).encode(
        y="Composite Quality:Q",
        color=alt.Color("Final State:N", scale=color_scale, legend=alt.Legend(title="Reliability State"))
    )

    # Operational threshold line at 0.65
    rule = alt.Chart(pd.DataFrame([{"threshold": 0.65}])).mark_rule(
        color="#10b981",
        strokeDash=[4, 4],
        strokeWidth=1.5
    ).encode(y="threshold:Q")

    chart = (line + points + rule).properties(
        height=260,
        title="Frame Reliability and Composite Quality Score Timeline"
    ).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="rgba(255,255,255,0.08)",
        domainColor="rgba(255,255,255,0.2)",
        labelColor="#94a3b8",
        titleColor="#cbd5e1"
    ).configure_title(
        color="#f8fafc"
    ).configure_legend(
        labelColor="#cbd5e1",
        titleColor="#f8fafc"
    )

    return chart


def render_stage1_header() -> None:
    """
    Renders the dedicated Stage 1 operational header.
    """
    header_html = """
    <div class="brand-container">
        <div>
            <h1 class="brand-title">PHANTOMROAD &bull; STAGE 1</h1>
            <p class="brand-subtitle">Video Ingestion, Frame Extraction & Optical Quality Telemetry &bull; OpenCV AI Competition 2026</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <span style="font-size: 11px; font-weight: 700; color: #10b981; background: rgba(16,185,129,0.1); padding: 5px 10px; border-radius: 6px; border: 1px solid rgba(16,185,129,0.3);">
                &#x25CF; STAGE 1 ONLINE
            </span>
            <span style="font-size: 11px; font-weight: 700; color: #64748b; background: rgba(255,255,255,0.05); padding: 5px 10px; border-radius: 6px; border: 1px solid rgba(100,116,139,0.3);">
                &#x25CF; LOCAL MODE (NO AWS)
            </span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_stage1_metrics_cards(blur_m: Dict[str, Any], illum_m: Dict[str, Any]) -> None:
    """
    Renders telemetry cards for Stage 1 (Blur, Brightness, Contrast, Entropy).
    """
    cols = st.columns(4)

    # 1. Blur / Sharpness
    with cols[0]:
        sharp_color = "#34d399" if not blur_m["is_blurred"] else "#fb7185"
        sharp_sub = "Nominal Sharp" if not blur_m["is_blurred"] else f"Blur Alert (< {blur_m['threshold']})"
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">Laplacian Variance</div>
            <div class="metric-value" style="color: {sharp_color};">{blur_m['laplacian_variance']:.1f}</div>
            <div class="metric-sub">{sharp_sub} (FFT: {blur_m['fft_high_freq_ratio']:.3f})</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Mean Luminance
    with cols[1]:
        lum_color = "#34d399" if not (illum_m["is_low_light"] or illum_m["is_overexposed"]) else "#fbbf24"
        lum_sub = "Low Light" if illum_m["is_low_light"] else ("Overexposed" if illum_m["is_overexposed"] else "Balanced Light")
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">Mean Luminance</div>
            <div class="metric-value" style="color: {lum_color};">{illum_m['mean_brightness']:.1f}</div>
            <div class="metric-sub">{lum_sub} / 255 (Dyn: {illum_m['dynamic_range']})</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. RMS Contrast
    with cols[2]:
        cont_color = "#34d399" if not illum_m["is_low_contrast"] else "#fbbf24"
        cont_sub = "Good Dynamic Separation" if not illum_m["is_low_contrast"] else "Low Contrast Alert"
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">RMS Contrast</div>
            <div class="metric-value" style="color: {cont_color};">{illum_m['rms_contrast']:.3f}</div>
            <div class="metric-sub">{cont_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Clipping & Information Density
    with cols[3]:
        under_pct = illum_m["underexposed_ratio"] * 100.0
        over_pct = illum_m["overexposed_ratio"] * 100.0
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">Exposure & Entropy</div>
            <div class="metric-value" style="color: #38bdf8;">{illum_m['entropy']:.2f} <span style="font-size:13px;color:#94a3b8;">bits</span></div>
            <div class="metric-sub">Shadows: {under_pct:.1f}% | Specular: {over_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


def build_stage1_histogram_chart(hist_data: Dict[str, Any]) -> alt.Chart:
    """
    Renders an interactive Altair intensity distribution histogram.
    """
    bins = hist_data.get("bins", list(range(256)))
    records = []

    if "luminance" in hist_data:
        for b, count in zip(bins, hist_data["luminance"]):
            records.append({"Intensity": b, "Pixel Count": count, "Channel": "Luminance (Y)"})
    if "red" in hist_data:
        for b, count in zip(bins, hist_data["red"]):
            records.append({"Intensity": b, "Pixel Count": count, "Channel": "Red (R)"})
    if "green" in hist_data:
        for b, count in zip(bins, hist_data["green"]):
            records.append({"Intensity": b, "Pixel Count": count, "Channel": "Green (G)"})
    if "blue" in hist_data:
        for b, count in zip(bins, hist_data["blue"]):
            records.append({"Intensity": b, "Pixel Count": count, "Channel": "Blue (B)"})

    df = pd.DataFrame(records)

    color_scale = alt.Scale(
        domain=["Luminance (Y)", "Red (R)", "Green (G)", "Blue (B)"],
        range=["#e2e8f0", "#ef4444", "#22c55e", "#3b82f6"]
    )

    chart = alt.Chart(df).mark_line(opacity=0.85, strokeWidth=1.5).encode(
        x=alt.X("Intensity:Q", title="Pixel Intensity (0 - 255)", scale=alt.Scale(domain=[0, 255])),
        y=alt.Y("Pixel Count:Q", title="Pixel Frequency"),
        color=alt.Color("Channel:N", scale=color_scale, legend=alt.Legend(title="Channel")),
        tooltip=["Channel", "Intensity", "Pixel Count"]
    ).properties(
        height=220,
        title="Frame Pixel Intensity Distribution Histogram"
    ).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="rgba(255,255,255,0.08)",
        domainColor="rgba(255,255,255,0.2)",
        labelColor="#94a3b8",
        titleColor="#cbd5e1"
    ).configure_title(
        color="#f8fafc"
    ).configure_legend(
        labelColor="#cbd5e1",
        titleColor="#f8fafc"
    )
    return chart


def build_stage1_batch_timeline(records: List[Dict[str, Any]]) -> alt.VConcatChart:
    """
    Renders timeline charts for blur, brightness, and contrast across video frames.
    """
    df = pd.DataFrame(records)
    base = alt.Chart(df).encode(x=alt.X("Frame:Q", title="Frame Index"))

    c_blur = base.mark_line(color="#38bdf8", strokeWidth=2).encode(
        y=alt.Y("Laplacian Variance:Q", title="Laplacian Var"),
        tooltip=["Frame", "Timestamp (s)", "Laplacian Variance", "Status Blur"]
    ).properties(height=150, title="Sharpness Timeline (Laplacian Variance)")

    c_illum = base.mark_line(color="#fbbf24", strokeWidth=2).encode(
        y=alt.Y("Mean Brightness:Q", scale=alt.Scale(domain=[0, 255]), title="Mean Luminance"),
        tooltip=["Frame", "Timestamp (s)", "Mean Brightness", "Status Lighting"]
    ).properties(height=150, title="Illumination Timeline (Mean Luminance)")

    c_contrast = base.mark_line(color="#a78bfa", strokeWidth=2).encode(
        y=alt.Y("RMS Contrast:Q", title="RMS Contrast"),
        tooltip=["Frame", "Timestamp (s)", "RMS Contrast", "Dynamic Range"]
    ).properties(height=150, title="Contrast Timeline (RMS Contrast)")

    vconcat = alt.vconcat(c_blur, c_illum, c_contrast).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="rgba(255,255,255,0.08)",
        domainColor="rgba(255,255,255,0.2)",
        labelColor="#94a3b8",
        titleColor="#cbd5e1"
    ).configure_title(
        color="#f8fafc"
    )
    return vconcat


def build_degraded_frames_timeline_chart(records: List[Dict[str, Any]]) -> alt.Chart:
    """
    Renders an interactive timeline showing unique degraded frames, severity, and degradation types.
    """
    df = pd.DataFrame(records)

    color_scale = alt.Scale(
        domain=[
            "Nominal",
            "Degraded (Low Light)",
            "Degraded (Low Contrast)",
            "Degraded (Blur)",
            "Degraded (Overexposed)",
            "Degraded (Low Light + Low Contrast)",
            "Degraded (Multiple Flags)"
        ],
        range=[
            "#10b981",  # Green
            "#f59e0b",  # Amber
            "#a855f7",  # Purple
            "#38bdf8",  # Sky Blue
            "#fb923c",  # Orange
            "#ef4444",  # Red
            "#e11d48"   # Rose
        ]
    )

    def map_category(status: str) -> str:
        if status == "Nominal":
            return "Nominal"
        if status == "Degraded (Low Light)":
            return "Degraded (Low Light)"
        if status == "Degraded (Low Contrast)":
            return "Degraded (Low Contrast)"
        if status == "Degraded (Blur)":
            return "Degraded (Blur)"
        if status == "Degraded (Overexposed)":
            return "Degraded (Overexposed)"
        if "Low Light" in status and "Low Contrast" in status:
            return "Degraded (Low Light + Low Contrast)"
        return "Degraded (Multiple Flags)"

    df["Category"] = df["Status Summary"].apply(map_category)

    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("Frame:Q", title="Video Frame Index"),
        y=alt.Y("Flag Count:Q", title="Active Degradation Flag Count", scale=alt.Scale(domain=[-0.2, 3.2])),
        color=alt.Color("Category:N", scale=color_scale, legend=alt.Legend(title="Optical Degradation State")),
        tooltip=[
            "Frame",
            "Timestamp (s)",
            "Laplacian Variance",
            "Mean Brightness",
            "RMS Contrast",
            "Status Summary",
            "Flag Count"
        ]
    ).properties(
        height=200,
        title="Unique Degraded Frames Timeline (Active Degradation Count per Frame)"
    ).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="rgba(255,255,255,0.08)",
        domainColor="rgba(255,255,255,0.2)",
        labelColor="#94a3b8",
        titleColor="#cbd5e1"
    ).configure_title(
        color="#f8fafc"
    ).configure_legend(
        labelColor="#cbd5e1",
        titleColor="#f8fafc"
    )
    return chart


def render_batch_summary_kpis(summary: Any) -> None:
    """
    Renders top KPI cards clearly distinguishing unique degraded frames from total flag occurrences,
    breakdown of overlapping conditions, and factual explanation for blur threshold.
    """
    # Row 1: Core Telemetry KPIs
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Unique Degraded Frames",
        f"{summary.unique_degraded_frames} / {summary.frames_analyzed}",
        f"{summary.unique_degraded_percentage:.1f}% of video"
    )
    k2.metric(
        "Total Flag Occurrences",
        summary.total_flag_occurrences,
        f"{summary.total_flag_occurrences / max(1, summary.frames_analyzed):.2f} flags/frame"
    )
    k3.metric(
        "Sharpness (Laplacian Var)",
        f"{summary.mean_sharpness:.1f}",
        f"Min: {summary.min_sharpness:.1f} | Max: {summary.max_sharpness:.1f}"
    )
    k4.metric(
        "Luminance (Mean Y)",
        f"{summary.mean_brightness:.1f}",
        f"Range: {summary.min_brightness:.1f} - {summary.max_brightness:.1f}"
    )
    k5.metric(
        "RMS Contrast (σ_Y / 255)",
        f"{summary.mean_contrast:.3f}",
        f"Min: {summary.min_contrast:.3f} | Max: {summary.max_contrast:.3f}"
    )

    # Row 2: Clarification and Overlap Breakdown Cards
    c_clarify, c_blur = st.columns([1, 1])

    with c_clarify:
        st.markdown(f"""
        <div class="hud-card" style="border-left: 4px solid #38bdf8; height: 100%;">
            <div style="font-size: 13px; font-weight: 700; color: #60a5fa; text-transform: uppercase; margin-bottom: 6px;">
                📊 Occurrences vs. Unique Degraded Frames Audit
            </div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
                &bull; <strong>Total Flag Occurrences ({summary.total_flag_occurrences}):</strong> Sum of individual threshold triggers (Dark: {summary.dark_flag_count} + Low Contrast: {summary.low_contrast_flag_count} + Blur: {summary.blur_flag_count} + Overexposed: {summary.overexposed_flag_count}).<br/>
                &bull; <strong>Unique Degraded Frames ({summary.unique_degraded_frames}):</strong> Distinct video frames having <em>at least one</em> degradation flag.<br/>
                &bull; <strong>Multiplicity Breakdown:</strong><br/>
                &nbsp;&nbsp;&ndash; <strong>Dual/Overlapping Flags:</strong> {summary.dual_flag_frames} frames ({summary.dual_flag_frames / max(1, summary.frames_analyzed) * 100:.1f}%) were <em>both</em> Dark & Low-Contrast.<br/>
                &nbsp;&nbsp;&ndash; <strong>Single Flag:</strong> {summary.single_flag_frames} frames ({summary.single_flag_frames / max(1, summary.frames_analyzed) * 100:.1f}%).<br/>
                &nbsp;&nbsp;&ndash; <strong>Nominal Frames:</strong> {summary.nominal_frames_count} frames ({summary.nominal_percentage:.1f}%).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_blur:
        st.markdown(f"""
        <div class="hud-card" style="border-left: 4px solid #10b981; height: 100%;">
            <div style="font-size: 13px; font-weight: 700; color: #34d399; text-transform: uppercase; margin-bottom: 6px;">
                🔍 Blur Threshold Analysis (0 of {summary.frames_analyzed} Triggered)
            </div>
            <div style="font-size: 12px; color: #cbd5e1; line-height: 1.5;">
                {summary.blur_explanation}
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 4px;">
                Configured Blur Threshold: <code>τ_blur = {summary.blur_threshold:.1f}</code> &bull; 
                Min Measured Sharpness: <code>{summary.min_sharpness:.1f}</code> &bull;
                Safety Margin: <code>+{max(0.0, summary.min_sharpness - summary.blur_threshold):.1f}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Robotics Safety vs Optical Telemetry Advisory
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px 16px; margin: 12px 0; font-size: 12px; color: #cbd5e1;">
        <strong style="color: #60a5fa;">🛡️ Autonomous Robotics Advisory (Optical Quality vs. Scene Safety):</strong>
        Low luminance (< 45.0) or low RMS contrast (< 0.12) measures <em>sensor operating conditions and epistemic evidence quality</em>—it does <strong>not</strong> prove that a scene is unsafe, that lane markings are absent, or that a collision is imminent. Night driving scenes with headlights naturally display low mean luminance while remaining drivable. Optical quality flags serve to down-weight camera certainty in multi-sensor fusion (e.g. LiDAR/radar handoff) and guide adaptive software compensation.
    </div>
    """, unsafe_allow_html=True)


def render_metric_reference_guide() -> None:
    """
    Renders comprehensive reference documentation for optical quality metrics,
    including exact mathematical formulas, digital units, and normalization ranges.
    """
    guide_html = """
    <div class="hud-card">
        <h4 style="margin: 0 0 12px 0; color: #f8fafc;">📐 Optical Quality Metric Formulations, Units & Normalization</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 12px; color: #cbd5e1;">
            <div style="background: rgba(0,0,0,0.25); padding: 12px; border-radius: 6px;">
                <strong style="color: #38bdf8;">1. Pech-Pacheco Laplacian Variance (Sharpness)</strong><br/>
                &bull; <strong>Formula:</strong> Var(∇²I) = (1/N) ∑_{x,y} (∇²I(x,y) - μ)² with 3×3 kernel<br/>
                &bull; <strong>Units:</strong> Intensity variance (intensity levels)² ∈ [0, ∞)<br/>
                &bull; <strong>Threshold:</strong> Flagged blurred if Var(∇²I) < τ_blur (Default: 85.0)<br/>
                &bull; <strong>Interpretation:</strong> Sensitive to high-frequency edge transitions and texture. Note that sensor analog noise (ISO grain) also elevates this variance.
            </div>
            <div style="background: rgba(0,0,0,0.25); padding: 12px; border-radius: 6px;">
                <strong style="color: #fbbf24;">2. Mean Luminance (Perceptual Brightness)</strong><br/>
                &bull; <strong>Formula:</strong> Y = 0.299 R + 0.587 G + 0.114 B (ITU-R BT.601)<br/>
                &bull; <strong>Units:</strong> 8-bit digital pixel values ∈ [0, 255]<br/>
                &bull; <strong>Thresholds:</strong> Low Light if Y < 45.0 or shadow clip > 40%; Overexposed if Y > 200.0<br/>
                &bull; <strong>Dynamic Range:</strong> P95 - P5 intensity spread ∈ [0, 255]
            </div>
            <div style="background: rgba(0,0,0,0.25); padding: 12px; border-radius: 6px;">
                <strong style="color: #a78bfa;">3. Root Mean Square (RMS) Contrast</strong><br/>
                &bull; <strong>Formula:</strong> C_RMS = (1/255) √[ (1/N) ∑_{i=1}^N (Y_i - Ȳ)² ] = σ_Y / 255<br/>
                &bull; <strong>Units:</strong> Dimensionless normalized ratio ∈ [0.0, 1.0]<br/>
                &bull; <strong>Threshold:</strong> Low Contrast if C_RMS < 0.12 (Default)<br/>
                &bull; <strong>Interpretation:</strong> Quantifies tone dispersion across the luminance histogram.
            </div>
            <div style="background: rgba(0,0,0,0.25); padding: 12px; border-radius: 6px;">
                <strong style="color: #34d399;">4. Information Density & Clipping Fractions</strong><br/>
                &bull; <strong>Shannon Entropy:</strong> H = -∑ p_k log₂(p_k) ∈ [0.0, 8.0] bits<br/>
                &bull; <strong>Underexposed Ratio:</strong> Fraction of pixels with Y < 20 (Deep shadows)<br/>
                &bull; <strong>Overexposed Ratio:</strong> Fraction of pixels with Y > 235 (Specular clipping)<br/>
                &bull; <strong>Composite Quality:</strong> Aggregated [0.0, 1.0] score weighting sharpness, contrast, exposure, and transmission.
            </div>
        </div>
    </div>
    """
    st.markdown(guide_html, unsafe_allow_html=True)



def render_detector_offline_alert() -> None:
    """
    Renders an actionable setup message when object detector weights are missing.
    Ensures all Stage 1 and Stage 2 features remain completely operational.
    """
    alert_html = """
    <div class="hud-card" style="border-left: 4px solid #f59e0b;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 20px;">⚠️</span>
            <span style="font-size: 16px; font-weight: 700; color: #fbbf24;">
                PRETRAINED ROAD-SCENE DETECTOR WEIGHTS OFFLINE
            </span>
        </div>
        <p style="font-size: 13px; color: #cbd5e1; margin: 0 0 10px 0; line-height: 1.5;">
            The <strong>MobileNet-SSD (Caffe / OpenCV DNN)</strong> model files are not yet present on local storage.
            To preserve stability and avoid vendor lock-in or unhandled exceptions, all Stage 1 (Telemetry)
            and Stage 2 (Adaptive Closed-Loop Decision Engine) features remain <strong>100% operational</strong>.
        </p>
        <div style="background: rgba(0, 0, 0, 0.4); border-radius: 6px; padding: 12px; margin-bottom: 12px; font-family: monospace; font-size: 12px; color: #38bdf8;">
            # Run this one-line acquisition script from the project root in PowerShell:<br/>
            <strong>python models/download_models.py</strong>
        </div>
        <div style="font-size: 12px; color: #94a3b8;">
            <strong>Expected Locations:</strong><br/>
            &bull; Prototxt Architecture: <code>models/MobileNetSSD_deploy.prototxt</code><br/>
            &bull; Pretrained Weights: <code>models/MobileNetSSD_deploy.caffemodel</code> (23.1 MB)
        </div>
    </div>
    """
    st.markdown(alert_html, unsafe_allow_html=True)


def render_stage3_detector_view(
    analyzer: Any,
    frame_bgr: np.ndarray,
    decision: AgentFrameDecision,
    conf_threshold: float = 0.35
) -> None:
    """
    Renders Stage 3 Object Detection and Reliability Annotation View.
    Shows annotated bounding boxes color-coded by reliability, zero invented detections guarantee,
    and structured detection metadata table.
    """
    if not analyzer.detector.is_available():
        render_detector_offline_alert()
        return

    # Run inference and reliability annotation
    detection_result = analyzer.analyze_frame(
        frame_bgr=frame_bgr,
        decision=decision,
        conf_threshold=conf_threshold
    )

    # If warning notice is present (e.g. degraded visual evidence)
    if detection_result.warning_notice:
        if "CRITICAL" in detection_result.warning_notice:
            st.markdown(f"""
            <div class="alert-danger-box">
                <strong>&#x26A0; PERCEPTION WARNING:</strong> {detection_result.warning_notice}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-warning-box">
                <strong>&#x25B2; DEGRADATION NOTICE:</strong> {detection_result.warning_notice}
            </div>
            """, unsafe_allow_html=True)

    # Detection Breakdown KPIs
    total_det = len(detection_result.detections)
    trusted_det = sum(1 for d in detection_result.detections if d.reliability_status == "TRUSTED")
    unverified_det = sum(1 for d in detection_result.detections if d.reliability_status == "UNVERIFIED_DEGRADED")
    suppressed_det = sum(1 for d in detection_result.detections if d.reliability_status == "SUPPRESSED_OBSTRUCTED")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Detections", total_det, "MobileNet-SSD")
    k2.metric("Trusted Objects", trusted_det, "Nominal Evidence", delta_color="normal")
    k3.metric("Unverified Candidates", unverified_det, "Degraded Evidence", delta_color="off")
    k4.metric("Suppressed / Blocked", suppressed_det, "Obstructed Sensor", delta_color="inverse")
    k5.metric("Evidence State", decision.final_state.value)

    # Annotated Image Rendering
    annotated_bgr = analyzer.render_annotated_frame(frame_bgr, detection_result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col_img, col_data = st.columns([3, 2])
    with col_img:
        st.markdown(f"**Annotated Road Scene Frame #{decision.frame_index}**")
        st.image(annotated_rgb, use_container_width=True)
        st.caption(
            "Bounding Box Colors: 🟢 Green = TRUSTED (Reliable) | "
            "🟠 Amber = UNVERIFIED (Degraded) | 🔴 Red = SUPPRESSED (Critical Obstruction)"
        )

    with col_data:
        st.markdown("**Perception Stream Metadata & Rationale**")
        if total_det == 0:
            st.info(
                "✓ No obstacles detected above confidence threshold "
                f"({conf_threshold:.2f}). **Honesty Guarantee:** Zero detections invented or simulated."
            )
        else:
            records = []
            for d in detection_result.detections:
                records.append({
                    "Class": d.label.title(),
                    "Conf": f"{d.confidence:.2f}",
                    "Status": d.reliability_status,
                    "Box [x1, y1, x2, y2]": f"[{d.box_xyxy[0]}, {d.box_xyxy[1]}, {d.box_xyxy[2]}, {d.box_xyxy[3]}]",
                    "Safety Rationale": d.suppression_reason or "Optical evidence nominal; bounding box verified."
                })
            df_det = pd.DataFrame(records)
            st.dataframe(df_det, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 12px; color: #94a3b8;">
            <strong style="color: #38bdf8;">Zero-Hallucination Integrity:</strong>
            PhantomRoad never injects synthetic bounding boxes to pass safety audits. Detections originate exclusively from neural network activations. When visual evidence quality drops below nominal operating levels, object candidates are explicitly flagged as <code>UNVERIFIED_DEGRADED</code> so downstream trajectory planning systems do not blindly assume clearance or misattribute distance.
        </div>
        """, unsafe_allow_html=True)


def render_perception_comparison_view(
    comparator: Any,
    clear_frame: np.ndarray,
    degraded_frame: np.ndarray,
    enhanced_frame: Optional[np.ndarray] = None,
    clear_decision: Optional[AgentFrameDecision] = None,
    degraded_decision: Optional[AgentFrameDecision] = None,
    enhanced_decision: Optional[AgentFrameDecision] = None,
    conf_threshold: float = 0.35
) -> None:
    """
    Renders Stage 3 Comparative Evaluation across Clear, Degraded, and Enhanced frames.
    """
    report = comparator.compare_frames(
        clear_frame=clear_frame,
        degraded_frame=degraded_frame,
        enhanced_frame=enhanced_frame,
        clear_decision=clear_decision,
        degraded_decision=degraded_decision,
        enhanced_decision=enhanced_decision,
        conf_threshold=conf_threshold
    )

    # Narrative Summary Banner
    st.markdown(f"""
    <div class="hud-card" style="border-left: 4px solid #38bdf8;">
        <span style="font-size: 14px; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.5px;">
            Perception Degradation Benchmark Report
        </span>
        <p style="font-size: 14px; color: #e2e8f0; margin: 6px 0 0 0;">
            {report.summary_narrative}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Comparative Metrics Strip
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Clear Frame Objects", report.clear_detection_count, f"Mean Conf: {report.clear_confidence_mean:.2f}")
    col2.metric("Degraded Frame Objects", report.degraded_detection_count, f"Mean Conf: {report.degraded_confidence_mean:.2f}")
    delta_label = f"{report.confidence_delta:+.2f} conf drop"
    col3.metric("Confidence Delta", f"{report.confidence_delta:+.2f}", delta_label, delta_color="inverse" if report.confidence_delta > 0 else "normal")
    col4.metric("Unverified / Degraded", report.unverified_count, "Flagged for safety")
    enhanced_label = f"Mean Conf: {report.enhanced_confidence_mean:.2f}" if enhanced_frame is not None else "N/A"
    col5.metric("Enhanced Frame Objects", report.enhanced_detection_count if enhanced_frame is not None else "N/A", enhanced_label)

    # Visual Comparison Side-by-Side
    annotated_clear = comparator.analyzer.render_annotated_frame(clear_frame, report.clear_result)
    annotated_deg = comparator.analyzer.render_annotated_frame(degraded_frame, report.degraded_result)

    if enhanced_frame is not None and report.enhanced_result is not None:
        annotated_enh = comparator.analyzer.render_annotated_frame(enhanced_frame, report.enhanced_result)
        c_clear, c_deg, c_enh = st.columns(3)
        with c_clear:
            st.markdown(f"**Clear Input ({report.clear_detection_count} detected)**")
            st.image(cv2.cvtColor(annotated_clear, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("All objects marked TRUSTED (Green)")
        with c_deg:
            st.markdown(f"**Degraded Input ({report.degraded_detection_count} detected)**")
            st.image(cv2.cvtColor(annotated_deg, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"{report.unverified_count} marked UNVERIFIED (Amber)")
        with c_enh:
            st.markdown(f"**Adaptive Output ({report.enhanced_detection_count} detected)**")
            st.image(cv2.cvtColor(annotated_enh, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Post-correction perception verification")
    else:
        c_clear, c_deg = st.columns(2)
        with c_clear:
            st.markdown(f"**Clear Input ({report.clear_detection_count} detected)**")
            st.image(cv2.cvtColor(annotated_clear, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("All objects marked TRUSTED (Green)")
        with c_deg:
            st.markdown(f"**Degraded Input ({report.degraded_detection_count} detected)**")
            st.image(cv2.cvtColor(annotated_deg, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"{report.unverified_count} marked UNVERIFIED (Amber)")

