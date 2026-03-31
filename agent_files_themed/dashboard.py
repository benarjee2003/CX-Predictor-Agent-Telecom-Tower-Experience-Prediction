"""
dashboard.py
Streamlit dashboard for CX Predictor Agent.
Watches a local Excel file — auto-reruns predictions when file changes or new rows are added.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from model_trainer import load_model, predict_batch
from agent_engine  import RiskEngine, RecommendationEngine, AlertEngine

# ─────────────────────────────────────────────
# EXCEL FILE WATCHER CONFIG
# Change this path to wherever your Excel file is on your Mac
# ─────────────────────────────────────────────
DEFAULT_EXCEL_PATH = str(Path.home() / "Desktop" / "tower_kpi.xlsx")

WATCH_INTERVAL = 3  # seconds between file checks

# Required columns in the Excel file
REQUIRED_COLS = [
    "tower_id", "region", "call_drop_rate", "cssr", "packet_loss",
    "latency_ms", "throughput_mbps", "dl_speed_mbps", "ul_speed_mbps",
    "session_failures", "customer_complaints", "weather"
]

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="CX Predictor Agent",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling — Dark Neon Tech Theme ────────────
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
        border-right: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: #fff !important; border: none !important;
        border-radius: 8px !important; font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950) !important;
        box-shadow: 0 0 12px rgba(46,160,67,0.5) !important;
    }
    [data-testid="block-container"] { background-color: #0d1117 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #1c2128) !important;
        border: 1px solid #30363d !important; border-radius: 12px !important;
        padding: 16px !important; box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700 !important; color: #58a6ff !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.82rem !important; }
    .risk-high   { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid #f85149;
                   padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .risk-medium { background: rgba(210,153,34,0.15); color: #d2992a; border: 1px solid #d2992a;
                   padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .risk-low    { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950;
                   padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    h1, h2, h3, .stSubheader { color: #58a6ff !important; }
    h1 { text-shadow: 0 0 20px rgba(88,166,255,0.4); }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #161b22 !important; border-radius: 10px 10px 0 0 !important;
        border-bottom: 1px solid #30363d !important; gap: 4px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent !important; color: #8b949e !important;
        border-radius: 8px 8px 0 0 !important; padding: 8px 18px !important; font-weight: 500 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(180deg, #1f6feb22, #1f6feb44) !important;
        color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important;
    }
    [data-testid="stExpander"] {
        background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important;
    }
    hr { border-color: #30363d !important; }
    [data-testid="stAlert"] {
        background: #161b22 !important; border: 1px solid #30363d !important;
        border-radius: 8px !important; color: #c9d1d9 !important;
    }
    [data-testid="stDataFrame"] { background: #161b22 !important; border-radius: 10px !important; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #6e7681 !important; font-size: 0.78rem !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
    .neon-title {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; letter-spacing: -0.5px;
    }
    .status-watching {
        background: rgba(63,185,80,0.1); border: 1px solid #3fb950;
        border-radius: 8px; padding: 8px 14px; font-size: 13px; color: #3fb950;
        display: inline-block; margin-bottom: 8px;
    }
    .status-error {
        background: rgba(248,81,73,0.1); border: 1px solid #f85149;
        border-radius: 8px; padding: 8px 14px; font-size: 13px; color: #f85149;
        display: inline-block; margin-bottom: 8px;
    }
    .status-waiting {
        background: rgba(88,166,255,0.1); border: 1px solid #58a6ff;
        border-radius: 8px; padding: 8px 14px; font-size: 13px; color: #58a6ff;
        display: inline-block; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load ML Model ─────────────────────────────
@st.cache_resource
def load_agent():
    models, meta = load_model()
    return models, meta


# ── Excel File Watcher Functions ──────────────
def get_file_signature(filepath):
    """Returns (last_modified_time, row_count) as a change signature."""
    try:
        mtime = os.path.getmtime(filepath)
        df    = pd.read_excel(filepath)
        return (mtime, len(df))
    except Exception:
        return (None, 0)


def load_excel_data(filepath):
    """Load and validate Excel file, fill optional columns."""
    df = pd.read_excel(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in your Excel: {missing}")

    now = datetime.now()
    if "hour"         not in df.columns: df["hour"]         = now.hour
    if "day_of_week"  not in df.columns: df["day_of_week"]  = now.weekday()
    if "is_peak_hour" not in df.columns:
        h = df.get("hour", now.hour)
        df["is_peak_hour"] = (((h >= 8) & (h <= 11)) | ((h >= 18) & (h <= 22))).astype(int)

    return df


def run_predictions(df):
    """Run the full prediction + risk pipeline."""
    models, meta  = load_agent()
    risk_engine   = RiskEngine()
    alert_engine  = AlertEngine(cooldown_minutes=0)
    result_df     = predict_batch(df, models, meta)
    result_df     = risk_engine.score_batch(result_df)
    new_alerts    = alert_engine.evaluate_batch(result_df)
    return result_df, new_alerts


# ── Session State ─────────────────────────────
defaults = {
    "alerts_log":     [],
    "result_df":      pd.DataFrame(),
    "last_signature": (None, 0),
    "last_checked":   None,
    "last_changed":   None,
    "excel_path":     DEFAULT_EXCEL_PATH,
    "watching":       False,
    "error_msg":      None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cell-tower.png", width=72)
    st.title("CX Predictor Agent")
    st.caption("Telecom KPI Intelligence")
    st.divider()

    st.subheader("📂 Excel File Source")
    excel_path = st.text_input(
        "File path on your Mac",
        value=st.session_state.excel_path,
        help="Full path e.g. /Users/yourname/Desktop/tower_kpi.xlsx"
    )
    st.session_state.excel_path = excel_path

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            st.session_state.watching     = True
            st.session_state.error_msg    = None
            st.session_state.alerts_log   = []
            st.session_state.result_df    = pd.DataFrame()
            st.session_state.last_signature = (None, 0)
    with col_b:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.watching = False

    st.divider()
    st.subheader("🔍 Agent Status")

    if st.session_state.watching:
        st.markdown('<div class="status-watching">🟢 Watching for changes...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-waiting">🔵 Not watching — press Start</div>', unsafe_allow_html=True)

    if st.session_state.last_checked:
        st.caption(f"Last checked: {st.session_state.last_checked}")
    if st.session_state.last_changed:
        st.caption(f"Last change detected: {st.session_state.last_changed}")
    if st.session_state.error_msg:
        st.markdown(f'<div class="status-error">❌ {st.session_state.error_msg}</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Force Re-run Now", use_container_width=True):
        try:
            df_excel = load_excel_data(st.session_state.excel_path)
            result, alerts = run_predictions(df_excel)
            st.session_state.result_df    = result
            st.session_state.alerts_log   = alerts
            st.session_state.last_changed = datetime.now().strftime("%H:%M:%S")
            st.session_state.error_msg    = None
            st.rerun()
        except Exception as e:
            st.session_state.error_msg = str(e)

    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.result_df      = pd.DataFrame()
        st.session_state.alerts_log     = []
        st.session_state.last_signature = (None, 0)
        st.rerun()

    st.divider()
    st.subheader("📋 Required Excel Columns")
    for c in REQUIRED_COLS:
        st.caption(f"• {c}")


# ── Main Header ───────────────────────────────
st.markdown('<p class="neon-title">📡 CX Predictor Agent — Excel Watcher</p>', unsafe_allow_html=True)
st.caption(f"Monitoring: `{st.session_state.excel_path}`  |  Checks every {WATCH_INTERVAL}s")

# ── File Watcher Logic ────────────────────────
if st.session_state.watching:
    filepath = st.session_state.excel_path
    st.session_state.last_checked = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(filepath):
        st.session_state.error_msg = f"File not found: {filepath}"
    else:
        try:
            current_sig = get_file_signature(filepath)
            prev_sig    = st.session_state.last_signature
            changed     = current_sig != prev_sig

            if changed:
                prev_rows = prev_sig[1]
                curr_rows = current_sig[1]
                change_type = "new rows added" if curr_rows > prev_rows else "data edited"

                df_excel = load_excel_data(filepath)
                result, alerts = run_predictions(df_excel)

                st.session_state.result_df      = result
                st.session_state.alerts_log     = alerts
                st.session_state.last_signature = current_sig
                st.session_state.last_changed   = datetime.now().strftime("%H:%M:%S")
                st.session_state.error_msg      = None

                st.toast(f"⚡ Change detected ({change_type}) — predictions updated!", icon="🔄")

        except Exception as e:
            st.session_state.error_msg = str(e)

# ── Change notification banner ────────────────
if st.session_state.last_changed and st.session_state.watching:
    st.markdown(
        f'<div style="background:rgba(63,185,80,0.08);border:1px solid #3fb950;border-radius:8px;'
        f'padding:10px 16px;margin-bottom:16px;font-size:13px;color:#3fb950">'
        f'⚡ <strong>Last prediction run:</strong> {st.session_state.last_changed} — '
        f'dashboard reflects your latest Excel data automatically.</div>',
        unsafe_allow_html=True
    )

# ── No data yet ───────────────────────────────
df     = st.session_state.result_df
alerts = st.session_state.alerts_log

if df.empty:
    st.divider()
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#8b949e">
        <div style="font-size:3rem;margin-bottom:16px">📂</div>
        <div style="font-size:1.2rem;font-weight:600;color:#58a6ff;margin-bottom:8px">No data yet</div>
        <div>Set your Excel file path in the sidebar and press <strong>▶ Start</strong></div>
        <div style="margin-top:12px;font-size:0.85rem">
            The agent will automatically re-run predictions every time you save changes to your Excel file.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📥 See sample Excel format to get started"):
        sample = pd.DataFrame([{
            "tower_id": "TWR_001", "region": "North", "call_drop_rate": 1.2,
            "cssr": 96.5, "packet_loss": 0.4, "latency_ms": 45,
            "throughput_mbps": 55.0, "dl_speed_mbps": 32.0, "ul_speed_mbps": 12.0,
            "session_failures": 2, "customer_complaints": 0, "weather": "Clear"
        }, {
            "tower_id": "TWR_002", "region": "South", "call_drop_rate": 5.8,
            "cssr": 78.0, "packet_loss": 3.2, "latency_ms": 210,
            "throughput_mbps": 8.0, "dl_speed_mbps": 5.0, "ul_speed_mbps": 2.0,
            "session_failures": 18, "customer_complaints": 7, "weather": "Thunderstorm"
        }])
        st.dataframe(sample, use_container_width=True)
        st.caption("Weather options: Clear, Cloudy, Rainy, Heavy_Rain, Thunderstorm, Foggy")
    st.stop()


# ── KPI Summary Cards ─────────────────────────
st.subheader("📊 Fleet Summary")
total  = len(df)
n_high = (df["risk_level"] == "High").sum()
n_med  = (df["risk_level"] == "Medium").sum()
n_low  = (df["risk_level"] == "Low").sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Towers",   total)
c2.metric("🔴 High Risk",   int(n_high), delta=f"{n_high/total*100:.1f}%", delta_color="inverse")
c3.metric("🟡 Medium Risk", int(n_med),  delta=f"{n_med/total*100:.1f}%",  delta_color="off")
c4.metric("🟢 Low Risk",    int(n_low),  delta=f"{n_low/total*100:.1f}%")
c5.metric("🚨 Alerts",      len(alerts), delta_color="inverse")

st.divider()

# ── Charts Row ────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Risk Distribution")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]
    fig = px.pie(risk_counts, values="Count", names="Risk", color="Risk",
                  color_discrete_map={"High":"#f85149","Medium":"#d2992a","Low":"#3fb950"}, hole=0.45)
    fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Experience Distribution")
    exp_counts = df["predicted_experience"].value_counts().reset_index()
    exp_counts.columns = ["Experience", "Count"]
    fig2 = px.bar(exp_counts, x="Experience", y="Count", color="Experience",
                   color_discrete_map={"Good":"#3fb950","Moderate":"#d2992a","Poor":"#f85149"})
    fig2.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0), showlegend=False,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
                       font=dict(color="#c9d1d9"),
                       xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.subheader("Avg Latency by Region")
    lat_region = df.groupby("region")["latency_ms"].mean().sort_values(ascending=False).reset_index().head(8)
    fig3 = px.bar(lat_region, x="latency_ms", y="region", orientation="h",
                   color="latency_ms", color_continuous_scale=["#3fb950","#d2992a","#f85149"])
    fig3.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0), showlegend=False, coloraxis_showscale=False,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
                       font=dict(color="#c9d1d9"),
                       xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
    st.plotly_chart(fig3, use_container_width=True)

# ── Region Risk Heatmap ───────────────────────
st.subheader("🗺️ Region Risk Heatmap")
region_risk = df.groupby("region").agg(
    high_risk=("risk_level", lambda x: (x=="High").sum()),
    total=("risk_level", "count"),
    avg_latency=("latency_ms", "mean"),
    avg_cdr=("call_drop_rate", "mean"),
).reset_index()
region_risk["risk_pct"] = (region_risk["high_risk"] / region_risk["total"] * 100).round(1)
region_risk = region_risk.sort_values("risk_pct", ascending=False)

fig_heat = px.bar(region_risk, x="region", y="risk_pct", color="risk_pct",
    color_continuous_scale=["#3fb950","#d2992a","#f85149"],
    labels={"risk_pct":"High Risk %","region":"Region"}, text="risk_pct")
fig_heat.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_color="#c9d1d9")
fig_heat.update_layout(height=320, margin=dict(t=10,b=80,l=0,r=0), coloraxis_showscale=False,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
                        font=dict(color="#c9d1d9"),
                        xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"))
st.plotly_chart(fig_heat, use_container_width=True)

# ── KPI Charts per Tower ──────────────────────
st.subheader("📈 KPI View by Tower")
tab1, tab2, tab3, tab4 = st.tabs(["Call Drop Rate", "Latency", "Packet Loss", "Throughput"])

chart_layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
                    font=dict(color="#c9d1d9"), xaxis=dict(gridcolor="#30363d"),
                    yaxis=dict(gridcolor="#30363d"), height=280, margin=dict(t=10,b=10))

with tab1:
    f = px.bar(df, x="tower_id", y="call_drop_rate", color="region",
               labels={"call_drop_rate":"CDR (%)","tower_id":"Tower"})
    f.add_hline(y=3.5, line_dash="dash", line_color="#f85149", annotation_text="High threshold")
    f.update_layout(**chart_layout)
    st.plotly_chart(f, use_container_width=True)

with tab2:
    f = px.bar(df, x="tower_id", y="latency_ms", color="region",
               labels={"latency_ms":"Latency (ms)","tower_id":"Tower"})
    f.add_hline(y=120, line_dash="dash", line_color="#d2992a", annotation_text="Medium threshold")
    f.add_hline(y=200, line_dash="dash", line_color="#f85149", annotation_text="High threshold")
    f.update_layout(**chart_layout)
    st.plotly_chart(f, use_container_width=True)

with tab3:
    f = px.bar(df, x="tower_id", y="packet_loss", color="region",
               labels={"packet_loss":"Packet Loss (%)","tower_id":"Tower"})
    f.add_hline(y=2.5, line_dash="dash", line_color="#f85149", annotation_text="High threshold")
    f.update_layout(**chart_layout)
    st.plotly_chart(f, use_container_width=True)

with tab4:
    f = px.bar(df, x="tower_id", y="throughput_mbps", color="region",
               labels={"throughput_mbps":"Throughput (Mbps)","tower_id":"Tower"})
    f.add_hline(y=5, line_dash="dash", line_color="#f85149", annotation_text="Critical threshold")
    f.update_layout(**chart_layout)
    st.plotly_chart(f, use_container_width=True)

# ── Alerts Panel ─────────────────────────────
st.subheader(f"🚨 Alerts Panel ({len(alerts)} total)")
if alerts:
    for alert in reversed(alerts[-15:]):
        sev      = alert["severity"]
        color    = "rgba(248,81,73,0.08)" if sev == "High" else "rgba(210,153,34,0.08)"
        bdr      = "#f85149" if sev == "High" else "#d2992a"
        txt      = "#ffa198" if sev == "High" else "#e3b341"
        recs_str = alert["recommendations"][0]["message"][:90]+"..." if alert["recommendations"] else "No specific recommendation"
        st.markdown(f"""
        <div style="background:{color};border-left:4px solid {bdr};padding:10px 16px;border-radius:4px;margin:4px 0;font-size:13px;color:{txt}">
          <strong>[{sev}]</strong> {alert['region']} / {alert['tower_id']} —
          {alert['experience']} (conf: {alert['confidence']:.0%}) |
          CDR: {alert['kpis']['call_drop_rate']:.1f}% | Latency: {alert['kpis']['latency_ms']:.0f}ms |
          Pkt Loss: {alert['kpis']['packet_loss']:.1f}%<br>
          <em>→ {recs_str}</em>
        </div>""", unsafe_allow_html=True)
else:
    st.success("✅ No alerts — all towers look healthy.")

# ── Raw Data Table ────────────────────────────
with st.expander("📋 Full Prediction Results (all rows from Excel)"):
    display_cols = [c for c in ["region","tower_id","risk_level","predicted_experience","confidence",
                    "call_drop_rate","cssr","latency_ms","packet_loss","throughput_mbps",
                    "session_failures","customer_complaints"] if c in df.columns]
    show_df = df[display_cols].copy()

    def color_risk(val):
        colors = {"High":"background-color:#3d1515","Medium":"background-color:#3d2e00","Low":"background-color:#0d2d1a"}
        return colors.get(val, "")

    st.dataframe(
        show_df.style.applymap(color_risk, subset=["risk_level"]),
        use_container_width=True, height=400
    )
    csv = show_df.to_csv(index=False)
    st.download_button("⬇️ Download Results as CSV", csv, "predictions.csv", "text/csv")

# ── Auto-recheck watcher ──────────────────────
if st.session_state.watching:
    time.sleep(WATCH_INTERVAL)
    st.rerun()
