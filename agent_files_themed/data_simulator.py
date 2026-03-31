"""
data_simulator.py
Generates realistic synthetic telecom KPI data for training and live simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

REGIONS = [
    "Chennai_North", "Chennai_South", "Chennai_Central",
    "Mumbai_East", "Mumbai_West", "Delhi_NCR",
    "Bangalore_Tech", "Bangalore_Outer", "Hyderabad_City",
    "Pune_Central", "Kolkata_North", "Ahmedabad_West"
]

TOWER_IDS = [f"TWR_{r[:3].upper()}_{i:03d}" for r in REGIONS for i in range(1, 6)]

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rainy", "Heavy_Rain", "Thunderstorm", "Foggy"]


def generate_kpi_row(region, tower_id, timestamp, scenario="normal"):
    """Generate one row of KPI data based on scenario."""
    hour = timestamp.hour
    is_peak = 8 <= hour <= 11 or 18 <= hour <= 22

    # Base KPI ranges by scenario
    if scenario == "good":
        cdr         = np.random.uniform(0.1, 1.0)
        cssr        = np.random.uniform(95, 99.9)
        packet_loss = np.random.uniform(0.1, 0.8)
        latency     = np.random.uniform(15, 50)
        throughput  = np.random.uniform(60, 150)
        dl_speed    = np.random.uniform(30, 80)
        ul_speed    = np.random.uniform(10, 30)
        session_failures = np.random.randint(0, 5)
        complaints  = np.random.randint(0, 3)

    elif scenario == "moderate":
        cdr         = np.random.uniform(1.0, 3.5)
        cssr        = np.random.uniform(85, 95)
        packet_loss = np.random.uniform(0.8, 2.5)
        latency     = np.random.uniform(50, 120)
        throughput  = np.random.uniform(20, 60)
        dl_speed    = np.random.uniform(10, 30)
        ul_speed    = np.random.uniform(3, 10)
        session_failures = np.random.randint(5, 20)
        complaints  = np.random.randint(3, 12)

    elif scenario == "poor":
        cdr         = np.random.uniform(3.5, 12.0)
        cssr        = np.random.uniform(60, 85)
        packet_loss = np.random.uniform(2.5, 10.0)
        latency     = np.random.uniform(120, 400)
        throughput  = np.random.uniform(1, 20)
        dl_speed    = np.random.uniform(0.5, 10)
        ul_speed    = np.random.uniform(0.1, 3)
        session_failures = np.random.randint(20, 80)
        complaints  = np.random.randint(12, 50)

    else:  # normal - mixed distribution
        scenario = np.random.choice(["good", "moderate", "poor"], p=[0.55, 0.30, 0.15])
        return generate_kpi_row(region, tower_id, timestamp, scenario)

    # Peak hour degradation
    if is_peak:
        cdr         *= np.random.uniform(1.1, 1.5)
        latency     *= np.random.uniform(1.1, 1.4)
        session_failures = int(session_failures * np.random.uniform(1.1, 1.6))

    weather = np.random.choice(WEATHER_CONDITIONS, p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05])
    if weather in ["Heavy_Rain", "Thunderstorm"]:
        packet_loss *= np.random.uniform(1.2, 2.0)
        latency     *= np.random.uniform(1.1, 1.5)

    # Derive label from thresholds
    if cdr >= 3.5 or cssr < 85 or packet_loss >= 2.5 or latency >= 120 or session_failures >= 20:
        label = "Poor"
    elif cdr >= 1.0 or cssr < 95 or packet_loss >= 0.8 or latency >= 50 or session_failures >= 5:
        label = "Moderate"
    else:
        label = "Good"

    return {
        "timestamp":        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "region":           region,
        "tower_id":         tower_id,
        "hour":             hour,
        "day_of_week":      timestamp.weekday(),
        "is_peak_hour":     int(is_peak),
        "weather":          weather,
        "call_drop_rate":   round(cdr, 3),
        "cssr":             round(cssr, 3),
        "packet_loss":      round(packet_loss, 3),
        "latency_ms":       round(latency, 2),
        "throughput_mbps":  round(throughput, 2),
        "dl_speed_mbps":    round(dl_speed, 2),
        "ul_speed_mbps":    round(ul_speed, 2),
        "session_failures": int(session_failures),
        "customer_complaints": int(complaints),
        "experience_label": label
    }


def generate_dataset(n_records=5000, output_path="data/telecom_kpi.csv"):
    """Generate full training dataset."""
    print(f"[DataSimulator] Generating {n_records} records...")
    records = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(n_records):
        region   = random.choice(REGIONS)
        tower_id = random.choice([t for t in TOWER_IDS if t.startswith(f"TWR_{region[:3].upper()}")] or TOWER_IDS)
        timestamp = base_time + timedelta(minutes=random.randint(0, 43200))
        records.append(generate_kpi_row(region, tower_id, timestamp))

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"[DataSimulator] Saved to {output_path}")
    print(f"[DataSimulator] Label distribution:\n{df['experience_label'].value_counts().to_string()}")
    return df


def generate_live_batch(n=10):
    """Generate a live batch of current readings (for agent monitoring loop)."""
    now = datetime.now()
    batch = []
    for _ in range(n):
        region   = random.choice(REGIONS)
        tower_id = random.choice([t for t in TOWER_IDS if t.startswith(f"TWR_{region[:3].upper()}")] or TOWER_IDS)
        # Inject some high-risk readings
        scenario = np.random.choice(["normal", "poor"], p=[0.75, 0.25])
        batch.append(generate_kpi_row(region, tower_id, now, scenario))
    return pd.DataFrame(batch)


if __name__ == "__main__":
    generate_dataset(5000, "data/telecom_kpi.csv")
