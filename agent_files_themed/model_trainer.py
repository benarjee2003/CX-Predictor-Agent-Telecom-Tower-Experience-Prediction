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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "call_drop_rate", "cssr", "packet_loss", "latency_ms",
    "throughput_mbps", "dl_speed_mbps", "ul_speed_mbps",
    "session_failures", "customer_complaints",
    "hour", "day_of_week", "is_peak_hour"
]

CATEGORICAL_FEATURES = ["weather"]
LABEL_COL = "experience_label"
LABEL_ORDER = ["Good", "Moderate", "Poor"]
MODEL_PATH = "models/cx_model.pkl"
META_PATH  = "models/cx_meta.pkl"


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode weather
    weather_map = {
        "Clear": 0, "Cloudy": 1, "Foggy": 2,
        "Rainy": 3, "Heavy_Rain": 4, "Thunderstorm": 5
    }
    df["weather_enc"] = df["weather"].map(weather_map).fillna(0)

    # Composite stress score
    df["network_stress"] = (
        df["call_drop_rate"] * 2 +
        (100 - df["cssr"]) * 0.5 +
        df["packet_loss"] * 3 +
        df["latency_ms"] / 50
    )

    # Speed ratio (asymmetry indicator)
    df["speed_ratio"] = df["dl_speed_mbps"] / (df["ul_speed_mbps"] + 0.001)

    # Combined failure load
    df["failure_load"] = df["session_failures"] + df["customer_complaints"] * 2

    # High latency flag
    df["high_latency_flag"] = (df["latency_ms"] > 100).astype(int)

    # CSSR degradation flag
    df["cssr_degraded"] = (df["cssr"] < 90).astype(int)

    return df


def get_feature_columns():
    return NUMERIC_FEATURES + [
        "weather_enc", "network_stress", "speed_ratio",
        "failure_load", "high_latency_flag", "cssr_degraded"
    ]


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(data_path="data/telecom_kpi.csv"):
    print("\n[Trainer] Loading data...")
    df = pd.read_csv(data_path)
    print(f"[Trainer] Records: {len(df)} | Labels: {df[LABEL_COL].value_counts().to_dict()}")

    df = engineer_features(df)
    feature_cols = get_feature_columns()

    X = df[feature_cols].fillna(0)
    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df[LABEL_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[Trainer] Training RandomForest + GradientBoosting ensemble...")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced", random_state=42
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=5,
        learning_rate=0.08, random_state=42
    )

    # Train both
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Soft-vote ensemble
    rf_probs = rf.predict_proba(X_test)
    gb_probs = gb.predict_proba(X_test)
    ensemble_probs = (rf_probs * 0.6 + gb_probs * 0.4)
    y_pred = np.argmax(ensemble_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[Trainer] ── Evaluation ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=LABEL_ORDER)}")

    # Cross-validation on RF alone
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1_weighted")
    print(f"[Trainer] 5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n[Trainer] Top-10 Feature Importances:\n{fi.head(10).to_string()}")

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump({"rf": rf, "gb": gb}, MODEL_PATH)
    meta = {
        "label_encoder": le,
        "feature_cols":  feature_cols,
        "label_order":   LABEL_ORDER,
        "accuracy":      round(acc, 4),
        "f1_score":      round(f1, 4),
        "feature_importance": fi.to_dict()
    }
    joblib.dump(meta, META_PATH)

    print(f"\n[Trainer] Model saved → {MODEL_PATH}")
    print(f"[Trainer] Meta  saved → {META_PATH}")
    return rf, gb, meta


# ─────────────────────────────────────────────
# Prediction interface
# ─────────────────────────────────────────────
def load_model():
    models = joblib.load(MODEL_PATH)
    meta   = joblib.load(META_PATH)
    return models, meta


def predict_batch(df: pd.DataFrame, models: dict, meta: dict) -> pd.DataFrame:
    df = engineer_features(df)
    X  = df[meta["feature_cols"]].fillna(0)
    le = meta["label_encoder"]

    rf_probs = models["rf"].predict_proba(X)
    gb_probs = models["gb"].predict_proba(X)
    probs    = rf_probs * 0.6 + gb_probs * 0.4

    predicted_idx = np.argmax(probs, axis=1)
    predicted_labels = le.inverse_transform(predicted_idx)
    confidence = np.max(probs, axis=1)

    result = df.copy()
    result["predicted_experience"] = predicted_labels
    result["confidence"]           = np.round(confidence, 4)
    result["prob_good"]     = np.round(probs[:, le.transform(["Good"])[0]], 4)
    result["prob_moderate"] = np.round(probs[:, le.transform(["Moderate"])[0]], 4)
    result["prob_poor"]     = np.round(probs[:, le.transform(["Poor"])[0]], 4)
    return result


if __name__ == "__main__":
    train("data/telecom_kpi.csv")
