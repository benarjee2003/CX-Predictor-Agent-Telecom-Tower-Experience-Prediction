"""
model_trainer.py
Handles preprocessing, feature engineering, model training, evaluation and persistence.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)

# ─────────────────────────────────────────────
# PATH SETUP (FIXED FOR CLOUD)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "cx_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "cx_meta.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "telecom_kpi.csv")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "call_drop_rate", "cssr", "packet_loss", "latency_ms",
    "throughput_mbps", "dl_speed_mbps", "ul_speed_mbps",
    "session_failures", "customer_complaints",
    "hour", "day_of_week", "is_peak_hour"
]

LABEL_COL = "experience_label"
LABEL_ORDER = ["Good", "Moderate", "Poor"]


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    weather_map = {
        "Clear": 0, "Cloudy": 1, "Foggy": 2,
        "Rainy": 3, "Heavy_Rain": 4, "Thunderstorm": 5
    }
    df["weather_enc"] = df["weather"].map(weather_map).fillna(0)

    df["network_stress"] = (
        df["call_drop_rate"] * 2 +
        (100 - df["cssr"]) * 0.5 +
        df["packet_loss"] * 3 +
        df["latency_ms"] / 50
    )

    df["speed_ratio"] = df["dl_speed_mbps"] / (df["ul_speed_mbps"] + 0.001)

    df["failure_load"] = df["session_failures"] + df["customer_complaints"] * 2

    df["high_latency_flag"] = (df["latency_ms"] > 100).astype(int)
    df["cssr_degraded"] = (df["cssr"] < 90).astype(int)

    return df


def get_feature_columns():
    return NUMERIC_FEATURES + [
        "weather_enc", "network_stress", "speed_ratio",
        "failure_load", "high_latency_flag", "cssr_degraded"
    ]


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(data_path=DATA_PATH):
    print("\n[Trainer] Loading data...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at: {data_path}")

    df = pd.read_csv(data_path)

    print(f"[Trainer] Records: {len(df)}")
    print(f"[Trainer] Label distribution:\n{df[LABEL_COL].value_counts()}")

    df = engineer_features(df)
    feature_cols = get_feature_columns()

    X = df[feature_cols].fillna(0)

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df[LABEL_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[Trainer] Training models...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        random_state=42
    )

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Ensemble
    rf_probs = rf.predict_proba(X_test)
    gb_probs = gb.predict_proba(X_test)
    probs = rf_probs * 0.6 + gb_probs * 0.4

    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n[Trainer] ── Evaluation ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\n", classification_report(y_test, y_pred, target_names=LABEL_ORDER))

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump({"rf": rf, "gb": gb}, MODEL_PATH)

    meta = {
        "label_encoder": le,
        "feature_cols": feature_cols,
        "label_order": LABEL_ORDER,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4)
    }

    joblib.dump(meta, META_PATH)

    print(f"\n[Trainer] Model saved → {MODEL_PATH}")
    print(f"[Trainer] Meta saved → {META_PATH}")

    return {"rf": rf, "gb": gb}, meta


# ─────────────────────────────────────────────
# LOAD MODEL (FIXED 🔥)
# ─────────────────────────────────────────────
def load_model():
    # AUTO-FIX: train if missing
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        print("[Model] Not found. Training new model...")
        train()

    models = joblib.load(MODEL_PATH)
    meta   = joblib.load(META_PATH)

    return models, meta


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_batch(df: pd.DataFrame, models: dict, meta: dict) -> pd.DataFrame:
    df = engineer_features(df)

    X = df[meta["feature_cols"]].fillna(0)
    le = meta["label_encoder"]

    rf_probs = models["rf"].predict_proba(X)
    gb_probs = models["gb"].predict_proba(X)
    probs = rf_probs * 0.6 + gb_probs * 0.4

    predicted_idx = np.argmax(probs, axis=1)
    predicted_labels = le.inverse_transform(predicted_idx)
    confidence = np.max(probs, axis=1)

    result = df.copy()
    result["predicted_experience"] = predicted_labels
    result["confidence"] = np.round(confidence, 4)

    return result


# ─────────────────────────────────────────────
# RUN TRAINING MANUALLY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train()
