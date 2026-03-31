# 📡 CX Predictor Agent
### AI-Powered Telecom Customer Experience Prediction, Risk Identification & Proactive Alerting

---

## 🗂️ Project Structure

```
cx_predictor_agent/
├── data_simulator.py    # Synthetic KPI data generator (training + live simulation)
├── model_trainer.py     # ML pipeline: preprocessing, feature engineering, RF + GB ensemble
├── agent_engine.py      # Risk Engine + Recommendation Engine + Alert Engine
├── cx_agent.py          # Main orchestrator — run this
├── dashboard.py         # Streamlit live dashboard
├── requirements.txt     # Dependencies
├── data/                # Generated dataset (created on first run)
├── models/              # Saved model + metadata
└── reports/             # Auto-generated alert reports + JSON logs
```

---

## ⚡ Quick Start (3 steps)

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy joblib scipy plotly streamlit
```

### 2. Run the agent (demo mode)
```bash
python cx_agent.py --mode demo
```
This will:
- Auto-generate 5,000 synthetic KPI records
- Train a Random Forest + Gradient Boosting ensemble
- Run single tower predictions (healthy + degraded)
- Analyze a live batch of 50 readings
- Fire alerts with recommendations
- Save a full report to `reports/`

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```

---

## 🎮 Agent Modes

```bash
# Demo mode (default) — full end-to-end run
python cx_agent.py --mode demo

# Live monitoring loop (3 cycles, 5s apart)
python cx_agent.py --mode monitor --cycles 5 --interval 10

# Force model retrain
python cx_agent.py --mode demo --retrain
```

---

## 🏗️ Architecture

```
Input KPIs (CSV/API/Live)
        │
        ▼
  Data Simulator / Ingestion
        │
        ▼
  Feature Engineering
  (rolling stats, composite scores, encoding)
        │
        ▼
  ML Model (RF + GB Ensemble)
  → Predicts: Good / Moderate / Poor
  → Outputs: class probabilities
        │
        ▼
  Risk Engine
  → prob_poor > 0.65 → High
  → prob_poor > 0.35 → Medium
  → Rule-based overrides (CDR > 6%, Latency > 250ms, etc.)
        │
        ▼
  Alert Engine  ←→  Recommendation Engine
  → Deduplication     → 13 rule-based KPI checks
  → Cooldown logic    → Top-3 recommendations per alert
  → Alert log/JSON    → Prioritized by severity
        │
        ▼
  Streamlit Dashboard + Reports
```

---

## 📊 KPI Features Used

| KPI                  | Category     | Weight in Model |
|----------------------|--------------|-----------------|
| Latency (ms)         | Network      | ⭐ Highest       |
| Call Drop Rate (%)   | Network      | ⭐⭐             |
| Session Failures     | UX           | ⭐⭐             |
| Packet Loss (%)      | Network      | High            |
| CSSR (%)             | Network      | High            |
| Throughput (Mbps)    | Network      | Medium          |
| DL / UL Speed        | UX           | Medium          |
| Customer Complaints  | UX           | Medium          |
| Weather              | Contextual   | Low             |
| Hour / Peak Flag     | Contextual   | Low             |

---

## 🚨 Alert Recommendation Codes

| Code                 | Trigger                        |
|----------------------|--------------------------------|
| CDR_CRITICAL         | Call Drop Rate > 5%            |
| CDR_HIGH             | Call Drop Rate > 3%            |
| CSSR_CRITICAL        | CSSR < 80%                     |
| LATENCY_CRITICAL     | Latency > 200ms                |
| PKT_LOSS_CRITICAL    | Packet Loss > 4%               |
| THROUGHPUT_CRITICAL  | Throughput < 5 Mbps            |
| SESSION_FAIL_CRITICAL| Session Failures > 40          |
| COMPLAINTS_SURGE     | Complaints > 20                |

---

## 🔌 API Usage (embed in your system)

```python
from model_trainer import load_model, predict_batch
from agent_engine  import RiskEngine, RecommendationEngine, AlertEngine
import pandas as pd

# Load once
models, meta = load_model()
risk_eng = RiskEngine()
rec_eng  = RecommendationEngine()

# Predict on live data
df = pd.read_csv("your_live_kpis.csv")
df = predict_batch(df, models, meta)
df = risk_eng.score_batch(df)

# Get recommendations for a row
recs = rec_eng.get_recommendations(df.iloc[0])
```

---

## 🔮 Upgrade Path

| Feature              | How to add                          |
|----------------------|-------------------------------------|
| XGBoost / LightGBM   | Replace RF in model_trainer.py      |
| Email alerts         | Add smtplib in AlertEngine          |
| SMS alerts           | Add Twilio in AlertEngine           |
| Real DB ingestion    | Replace generate_live_batch()       |
| LLM explanations     | Pipe alert dict to Claude/GPT API   |
| SHAP explainability  | Add shap.TreeExplainer in trainer   |
| MLflow tracking      | Add mlflow.log_metric in trainer    |

---

## 📈 Model Performance

- **Algorithm**: RandomForest (200 trees) + GradientBoosting ensemble (soft vote 60/40)
- **Classes**: Good / Moderate / Poor
- **CV F1**: ~0.9996 ± 0.0005 (5-fold stratified)
- **Top Features**: latency_ms, call_drop_rate, session_failures, packet_loss
