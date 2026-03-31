import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import sys

# Fix imports
sys.path.insert(0, os.path.dirname(__file__))

from model_trainer import load_model, predict_batch
from agent_engine import RiskEngine, AlertEngine

# Page config
st.set_page_config(page_title="CX Predictor Agent", layout="wide")

st.title("📡 CX Predictor Agent (Cloud Version)")

# ─────────────────────────────────────────────
# Upload File (WORKS IN STREAMLIT CLOUD)
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload KPI Excel/CSV", type=["xlsx", "csv"])

REQUIRED_COLS = [
    "tower_id", "region", "call_drop_rate", "cssr", "packet_loss",
    "latency_ms", "throughput_mbps", "dl_speed_mbps", "ul_speed_mbps",
    "session_failures", "customer_complaints", "weather"
]

# Load model
@st.cache_resource
def load_agent():
    return load_model()

# Run predictions
def run_predictions(df):
    models, meta = load_agent()
    risk_engine = RiskEngine()
    alert_engine = AlertEngine(cooldown_minutes=0)

    df = predict_batch(df, models, meta)
    df = risk_engine.score_batch(df)
    alerts = alert_engine.evaluate_batch(df)

    return df, alerts

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Clean columns
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Validate columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
            st.stop()

        # Add time features
        now = datetime.now()
        df["hour"] = now.hour
        df["day_of_week"] = now.weekday()
        df["is_peak_hour"] = (((df["hour"] >= 8) & (df["hour"] <= 11)) |
                              ((df["hour"] >= 18) & (df["hour"] <= 22))).astype(int)

        # Run predictions
        df, alerts = run_predictions(df)

        st.success("✅ Predictions updated successfully!")

        # ─────────────────────────────────────────────
        # KPIs
        # ─────────────────────────────────────────────
        st.subheader("📊 Summary")

        total = len(df)
        high = (df["risk_level"] == "High").sum()
        med = (df["risk_level"] == "Medium").sum()
        low = (df["risk_level"] == "Low").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Towers", total)
        c2.metric("🔴 High Risk", int(high))
        c3.metric("🟡 Medium Risk", int(med))
        c4.metric("🟢 Low Risk", int(low))

        # ─────────────────────────────────────────────
        # Charts
        # ─────────────────────────────────────────────
        st.subheader("📈 Risk Distribution")

        fig = px.pie(df, names="risk_level", title="Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 Latency by Region")
        fig2 = px.bar(df, x="region", y="latency_ms", color="risk_level")
        st.plotly_chart(fig2, use_container_width=True)

        # ─────────────────────────────────────────────
        # Alerts
        # ─────────────────────────────────────────────
        st.subheader("🚨 Alerts")

        if alerts:
            for a in alerts:
                st.warning(
                    f"{a['tower_id']} | {a['region']} → {a['experience']} "
                    f"(Confidence: {a['confidence']:.0%})"
                )
        else:
            st.success("No alerts 🚀")

        # ─────────────────────────────────────────────
        # Data Table
        # ─────────────────────────────────────────────
        st.subheader("📋 Full Data")
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Upload an Excel/CSV file to start")
