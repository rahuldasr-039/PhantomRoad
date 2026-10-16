"""
UI package for PhantomRoad Streamlit operations dashboard.
"""

from .styles import DARK_THEME_CSS, get_state_badge_html, get_stop_condition_badge_html
from .components import (
    render_brand_header,
    render_stage1_header,
    render_telemetry_cards,
    render_stage1_metrics_cards,
    render_decision_hud,
    render_comparison_slider,
    render_detector_offline_alert,
    render_stage3_detector_view,
    render_perception_comparison_view,
    render_batch_summary_kpis,
    render_metric_reference_guide,
    build_stage1_histogram_chart,
    build_stage1_batch_timeline,
    build_degraded_frames_timeline_chart,
    build_reliability_timeline_chart,
)

__all__ = [
    "DARK_THEME_CSS",
    "get_state_badge_html",
    "get_stop_condition_badge_html",
    "render_brand_header",
    "render_stage1_header",
    "render_telemetry_cards",
    "render_stage1_metrics_cards",
    "render_decision_hud",
    "render_comparison_slider",
    "render_detector_offline_alert",
    "render_stage3_detector_view",
    "render_perception_comparison_view",
    "render_batch_summary_kpis",
    "render_metric_reference_guide",
    "build_stage1_histogram_chart",
    "build_stage1_batch_timeline",
    "build_degraded_frames_timeline_chart",
    "build_reliability_timeline_chart",
]
