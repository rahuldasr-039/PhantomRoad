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
