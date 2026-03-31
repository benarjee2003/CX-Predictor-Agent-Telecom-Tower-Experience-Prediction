"""
cx_agent.py
Main CX Predictor Agent — orchestrates all modules.
Run directly: python cx_agent.py
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ── local modules ──
from data_simulator import generate_dataset, generate_live_batch
from model_trainer   import train, load_model, predict_batch, LABEL_ORDER
from agent_engine    import RiskEngine, RecommendationEngine, AlertEngine


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     CX PREDICTOR AGENT  —  Telecom KPI Intelligence         ║
║     Predict  |  Risk-Score  |  Alert  |  Recommend           ║
╚══════════════════════════════════════════════════════════════╝
"""

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
def setup(force_retrain=False):
    """Generate data and train model if not already done."""
    print(BANNER)

    data_path  = "data/telecom_kpi.csv"
    model_path = "models/cx_model.pkl"

    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(data_path):
        print("[Setup] No dataset found. Generating synthetic telecom KPI data...")
        generate_dataset(5000, data_path)
    else:
        print(f"[Setup] Dataset found: {data_path}")

    if not os.path.exists(model_path) or force_retrain:
        print("[Setup] Training model...")
        train(data_path)
    else:
        print(f"[Setup] Model found: {model_path}")

    models, meta = load_model()
    print(f"\n[Setup] Model loaded — Accuracy: {meta['accuracy']:.2%} | F1: {meta['f1_score']:.2%}")
    return models, meta


# ─────────────────────────────────────────────
# SINGLE PREDICTION
# ─────────────────────────────────────────────
def predict_single(kpi_dict: dict, models: dict, meta: dict) -> dict:
    """Predict CX for a single KPI reading dict."""
    df = pd.DataFrame([kpi_dict])
    risk_engine = RiskEngine()
    rec_engine  = RecommendationEngine()

    df = predict_batch(df, models, meta)
    df = risk_engine.score_batch(df)

    row = df.iloc[0]
    recs = rec_engine.get_recommendations(row)

    return {
        "prediction":      row["predicted_experience"],
        "confidence":      round(float(row["confidence"]), 4),
        "prob_good":       round(float(row["prob_good"]), 4),
        "prob_moderate":   round(float(row["prob_moderate"]), 4),
        "prob_poor":       round(float(row["prob_poor"]), 4),
        "risk_level":      row["risk_level"],
        "recommendations": recs
    }


# ─────────────────────────────────────────────
# BATCH ANALYSIS
# ─────────────────────────────────────────────
def analyze_batch(df: pd.DataFrame, models: dict, meta: dict,
                  alert_engine: AlertEngine) -> pd.DataFrame:
    """Run full pipeline on a batch of KPI readings."""
    risk_engine = RiskEngine()

    df = predict_batch(df, models, meta)
    df = risk_engine.score_batch(df)

    alerts = alert_engine.evaluate_batch(df)
    for alert in alerts:
        print(alert_engine.format_alert_text(alert))

    return df, alerts


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────
def generate_report(df: pd.DataFrame, alerts: list, alert_engine: AlertEngine) -> str:
    """Generate a text summary report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    total       = len(df)
    risk_counts = df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {}
    pred_counts = df["predicted_experience"].value_counts().to_dict() if "predicted_experience" in df.columns else {}

    high_risk_regions = (
        df[df["risk_level"] == "High"]
        .groupby("region")["risk_level"]
        .count()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    ) if "risk_level" in df.columns else {}

    high_risk_towers = (
        df[df["risk_level"] == "High"]
        .groupby("tower_id")["risk_level"]
        .count()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    ) if "risk_level" in df.columns else {}

    report_lines = [
        "=" * 65,
        f"  CX PREDICTOR AGENT — ANALYSIS REPORT",
        f"  Generated: {now}",
        "=" * 65,
        "",
        "  SUMMARY",
        f"  ─────────────────────────────────────",
        f"  Total readings analyzed : {total}",
        f"  High-risk readings      : {risk_counts.get('High', 0)}  ({risk_counts.get('High', 0)/max(total,1)*100:.1f}%)",
        f"  Medium-risk readings    : {risk_counts.get('Medium', 0)}  ({risk_counts.get('Medium', 0)/max(total,1)*100:.1f}%)",
        f"  Low-risk readings       : {risk_counts.get('Low', 0)}  ({risk_counts.get('Low', 0)/max(total,1)*100:.1f}%)",
        f"  Alerts triggered        : {len(alerts)}",
        "",
        "  EXPERIENCE DISTRIBUTION",
        f"  ─────────────────────────────────────",
        *[f"  {k:12s} : {v}" for k, v in pred_counts.items()],
        "",
        "  HIGH-RISK REGIONS",
        f"  ─────────────────────────────────────",
        *[f"  {r:25s} : {c} high-risk readings" for r, c in high_risk_regions.items()],
        "",
        "  HIGH-RISK TOWERS",
        f"  ─────────────────────────────────────",
        *[f"  {t:15s} : {c} high-risk readings" for t, c in high_risk_towers.items()],
        "",
        "  ALERTS",
        f"  ─────────────────────────────────────",
    ]

    for alert in alerts[:10]:
        recs_str = alert["recommendations"][0]["code"] if alert["recommendations"] else "N/A"
        report_lines.append(
            f"  [{alert['severity']:6s}] {alert['region']:20s} | {alert['tower_id']:12s} | {recs_str}"
        )

    report_lines += ["", "=" * 65]
    report_text = "\n".join(report_lines)

    # Save report
    report_path = f"{REPORT_DIR}/report_{ts}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save alerts JSON
    alerts_path = f"{REPORT_DIR}/alerts_{ts}.json"
    with open(alerts_path, "w") as f:
        json.dump(alerts, f, indent=2, default=str)

    print(report_text)
    print(f"\n[Agent] Report saved → {report_path}")
    print(f"[Agent] Alerts saved → {alerts_path}")
    return report_path


# ─────────────────────────────────────────────
# MONITORING LOOP
# ─────────────────────────────────────────────
def monitoring_loop(models: dict, meta: dict, interval_seconds=10,
                    max_cycles=5, batch_size=20):
    """Continuous monitoring loop — simulates live KPI ingestion."""
    print(f"\n[Monitor] Starting live monitoring loop (interval={interval_seconds}s, cycles={max_cycles})")
    print("[Monitor] Press Ctrl+C to stop.\n")

    alert_engine = AlertEngine(cooldown_minutes=5)
    cycle = 0

    try:
        while cycle < max_cycles:
            cycle += 1
            now = datetime.now().strftime("%H:%M:%S")
            print(f"\n[Monitor] ── Cycle {cycle}/{max_cycles} at {now} ──")

            live_df = generate_live_batch(n=batch_size)
            result_df, alerts = analyze_batch(live_df, models, meta, alert_engine)

            high  = (result_df["risk_level"] == "High").sum()
            med   = (result_df["risk_level"] == "Medium").sum()
            low   = (result_df["risk_level"] == "Low").sum()
            print(f"[Monitor] Processed {len(result_df)} readings → "
                  f"High: {high} | Medium: {med} | Low: {low} | Alerts: {len(alerts)}")

            if cycle < max_cycles:
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n[Monitor] Stopped by user.")

    print(f"\n[Monitor] Total alerts generated: {len(alert_engine.alerts_log)}")
    return alert_engine


# ─────────────────────────────────────────────
# DEMO MODE
# ─────────────────────────────────────────────
def run_demo(models: dict, meta: dict):
    """Run a full demo: single prediction + batch analysis + report."""
    print("\n" + "─"*60)
    print("  DEMO MODE")
    print("─"*60)

    # 1. Single prediction example
    print("\n[Demo] Single Prediction — Healthy Tower:")
    healthy = {
        "call_drop_rate": 0.5, "cssr": 98.2, "packet_loss": 0.3,
        "latency_ms": 25, "throughput_mbps": 80, "dl_speed_mbps": 45,
        "ul_speed_mbps": 18, "session_failures": 2, "customer_complaints": 1,
        "hour": 10, "day_of_week": 1, "is_peak_hour": 1, "weather": "Clear"
    }
    result = predict_single(healthy, models, meta)
    print(f"  → Prediction  : {result['prediction']}")
    print(f"  → Risk Level  : {result['risk_level']}")
    print(f"  → Confidence  : {result['confidence']:.1%}")
    print(f"  → P(Good/Mod/Poor): {result['prob_good']:.1%} / {result['prob_moderate']:.1%} / {result['prob_poor']:.1%}")

    print("\n[Demo] Single Prediction — Degraded Tower:")
    degraded = {
        "call_drop_rate": 7.2, "cssr": 74.5, "packet_loss": 6.1,
        "latency_ms": 310, "throughput_mbps": 3.2, "dl_speed_mbps": 1.8,
        "ul_speed_mbps": 0.4, "session_failures": 65, "customer_complaints": 35,
        "hour": 20, "day_of_week": 4, "is_peak_hour": 1, "weather": "Heavy_Rain"
    }
    result = predict_single(degraded, models, meta)
    print(f"  → Prediction  : {result['prediction']}")
    print(f"  → Risk Level  : {result['risk_level']}")
    print(f"  → Confidence  : {result['confidence']:.1%}")
    print(f"  → P(Good/Mod/Poor): {result['prob_good']:.1%} / {result['prob_moderate']:.1%} / {result['prob_poor']:.1%}")
    if result["recommendations"]:
        print(f"  → Top Rec     : [{result['recommendations'][0]['code']}] {result['recommendations'][0]['message'][:80]}...")

    # 2. Batch analysis on live-simulated data
    print("\n[Demo] Batch Analysis (50 readings)...")
    alert_engine = AlertEngine(cooldown_minutes=0)
    batch_df = generate_live_batch(n=50)
    result_df, alerts = analyze_batch(batch_df, models, meta, alert_engine)

    # 3. Report
    generate_report(result_df, alerts, alert_engine)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CX Predictor Agent")
    parser.add_argument("--mode",    choices=["demo", "monitor", "train"], default="demo")
    parser.add_argument("--retrain", action="store_true", help="Force retrain model")
    parser.add_argument("--cycles",  type=int, default=3, help="Monitor cycles (default 3)")
    parser.add_argument("--interval",type=int, default=5, help="Seconds between cycles (default 5)")
    args = parser.parse_args()

    models, meta = setup(force_retrain=args.retrain)

    if args.mode == "train":
        print("[Agent] Retraining complete.")

    elif args.mode == "monitor":
        alert_engine = monitoring_loop(
            models, meta,
            interval_seconds=args.interval,
            max_cycles=args.cycles,
            batch_size=20
        )

    else:  # demo
        run_demo(models, meta)
