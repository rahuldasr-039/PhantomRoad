"""
Professional Dark Computer Vision Operations Theme for PhantomRoad.
Custom CSS with glassmorphism, tactical HUD badges, and telemetry card styling.
"""

DARK_THEME_CSS = """
<style>
/* Base Dark Operations Theme */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Glassmorphism Cards */
.hud-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(30, 41, 59, 0.70) 100%);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
    margin-bottom: 16px;
}

.hud-card-accent {
    border-left: 4px solid #3b82f6;
}

/* Top Navigation / Brand Title */
.brand-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 20px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 24px;
}

.brand-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #60a5fa 0%, #38bdf8 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.brand-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin: 2px 0 0 0;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Status Badges */
.badge-reliable {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background-color: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
}

.badge-correctable {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background-color: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
}

.badge-escalation {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background-color: rgba(244, 63, 94, 0.18);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.4);
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
}

.badge-obstruction {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background-color: rgba(225, 29, 72, 0.25);
    color: #fda4af;
    border: 1px solid #e11d48;
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 0.5px;
    animation: pulse 2s infinite;
}

/* Metric Display Numbers */
.metric-value {
    font-size: 26px;
    font-weight: 700;
    font-family: "JetBrains Mono", "Courier New", monospace;
    color: #f8fafc;
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-bottom: 4px;
}

.metric-sub {
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
}

/* Audit Step Row */
.audit-step-box {
    background-color: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
}

/* Warnings and Notices */
.alert-warning-box {
    background-color: rgba(245, 158, 11, 0.12);
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 12px 0;
    color: #fef3c7;
    font-size: 13px;
}

.alert-danger-box {
    background-color: rgba(244, 63, 94, 0.12);
    border-left: 4px solid #f43f5e;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 12px 0;
    color: #ffe4e6;
    font-size: 13px;
}

/* Table styling */
.dataframe {
    background-color: #111827 !important;
    color: #e2e8f0 !important;
    border-color: #374151 !important;
}
</style>
"""


def get_state_badge_html(state_str: str) -> str:
    """Returns HTML for color-coded status badge."""
    if state_str == "RELIABLE":
        return '<span class="badge-reliable">&#x25CF; NOMINAL RELIABLE</span>'
    elif state_str in ("DEGRADED_CORRECTABLE", "DEGRADED_UNCERTAIN"):
        return f'<span class="badge-correctable">&#x25B2; {state_str.replace("_", " ")}</span>'
    elif state_str == "CRITICAL_OBSTRUCTION":
        return '<span class="badge-obstruction">&#x26A0; CRITICAL OBSTRUCTION</span>'
    else:
        return '<span class="badge-escalation">&#x26A0; HUMAN REVIEW REQUIRED</span>'


def get_stop_condition_badge_html(stop_cond_str: str) -> str:
    """Returns HTML for color-coded Stop Condition badge."""
    if stop_cond_str == "STOP_ALREADY_RELIABLE":
        return '<span style="display:inline-block; padding:3px 10px; border-radius:9999px; background:rgba(59,130,246,0.18); color:#60a5fa; border:1px solid rgba(59,130,246,0.4); font-size:11px; font-weight:700;">&#x2713; STOP: ALREADY RELIABLE</span>'
    elif stop_cond_str == "STOP_RELIABLE_ACHIEVED":
        return '<span style="display:inline-block; padding:3px 10px; border-radius:9999px; background:rgba(16,185,129,0.18); color:#34d399; border:1px solid rgba(16,185,129,0.4); font-size:11px; font-weight:700;">&#x2713; STOP: RELIABLE ACHIEVED</span>'
    elif stop_cond_str == "STOP_CRITICAL_OBSTRUCTION":
        return '<span style="display:inline-block; padding:3px 10px; border-radius:9999px; background:rgba(225,29,72,0.25); color:#fda4af; border:1px solid #e11d48; font-size:11px; font-weight:800;">&#x26A0; STOP: CRITICAL OBSTRUCTION</span>'
    elif stop_cond_str == "STOP_MAX_ATTEMPTS_EXHAUSTED":
        return '<span style="display:inline-block; padding:3px 10px; border-radius:9999px; background:rgba(245,158,11,0.18); color:#fbbf24; border:1px solid rgba(245,158,11,0.4); font-size:11px; font-weight:700;">&#x25B2; STOP: MAX ATTEMPTS EXHAUSTED</span>'
    elif stop_cond_str == "STOP_NO_VIABLE_ACTIONS":
        return '<span style="display:inline-block; padding:3px 10px; border-radius:9999px; background:rgba(244,63,94,0.18); color:#fb7185; border:1px solid rgba(244,63,94,0.4); font-size:11px; font-weight:700;">&#x26A0; STOP: NO VIABLE ACTIONS</span>'
    else:
        return f'<span style="font-size:11px; color:#94a3b8;">{stop_cond_str}</span>'

