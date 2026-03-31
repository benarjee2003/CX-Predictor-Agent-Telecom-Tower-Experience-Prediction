"""
agent_engine.py
Risk Engine + Recommendation Engine + Alert Engine
The core intelligence layer of the CX Predictor Agent.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────────
# RISK ENGINE
# ─────────────────────────────────────────────
class RiskEngine:
    """Converts model probabilities + KPI values into risk levels."""

    HIGH_THRESHOLD   = 0.65
    MEDIUM_THRESHOLD = 0.35

    # Hard threshold overrides (rule-based safety net)
    CRITICAL_RULES = [
        ("call_drop_rate",   ">",  6.0,  "High"),
        ("cssr",             "<",  75.0, "High"),
        ("packet_loss",      ">",  5.0,  "High"),
        ("latency_ms",       ">",  250,  "High"),
        ("session_failures", ">",  50,   "High"),
        ("call_drop_rate",   ">",  3.5,  "Medium"),
        ("latency_ms",       ">",  120,  "Medium"),
        ("packet_loss",      ">",  2.5,  "Medium"),
    ]

    def score_row(self, row: pd.Series) -> dict:
        prob_poor = row.get("prob_poor", 0.0)

        # Rule-based override first
        rule_risk = "Low"
        triggered_rule = None
        for col, op, threshold, risk_level in self.CRITICAL_RULES:
            val = row.get(col, 0)
            if op == ">" and val > threshold:
                rule_risk = risk_level
                triggered_rule = f"{col} {op} {threshold} (actual: {val:.2f})"
                break
            elif op == "<" and val < threshold:
                rule_risk = risk_level
                triggered_rule = f"{col} {op} {threshold} (actual: {val:.2f})"
                break

        # Probability-based risk
        if prob_poor >= self.HIGH_THRESHOLD:
            prob_risk = "High"
        elif prob_poor >= self.MEDIUM_THRESHOLD:
            prob_risk = "Medium"
        else:
            prob_risk = "Low"

        # Take the more severe of the two
        risk_priority = {"Low": 0, "Medium": 1, "High": 2}
        final_risk = rule_risk if risk_priority[rule_risk] >= risk_priority[prob_risk] else prob_risk

        return {
            "risk_level":     final_risk,
            "prob_poor":      prob_poor,
            "rule_triggered": triggered_rule
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df.apply(self.score_row, axis=1, result_type="expand")
        df = df.copy()
        df["risk_level"]     = results["risk_level"]
        df["rule_triggered"] = results["rule_triggered"]
        return df


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
class RecommendationEngine:
    """Rule-based engine that maps KPI degradations to actionable recommendations."""

    RULES = [
        # (kpi, operator, threshold, priority, short_code, recommendation)
        ("call_drop_rate",   ">",  5.0,  1, "CDR_CRITICAL",
         "Critical CDR detected. Immediately check handover parameters, antenna tilt, and RF interference. Escalate to RF team."),

        ("call_drop_rate",   ">",  3.0,  2, "CDR_HIGH",
         "High call drop rate. Audit handover success rates, check neighbor cell list, verify RACH configuration."),

        ("cssr",             "<",  80.0, 1, "CSSR_CRITICAL",
         "Call setup success is critically low. Check RACH/PRACH failures, backhaul link status, and core network connectivity."),

        ("cssr",             "<",  90.0, 2, "CSSR_LOW",
         "CSSR below SLA. Review congestion on RACH channels and check signaling load on the core."),

        ("latency_ms",       ">",  200,  1, "LATENCY_CRITICAL",
         "Severe latency detected. Check backhaul transmission link, routing table anomalies, and transport network congestion."),

        ("latency_ms",       ">",  100,  2, "LATENCY_HIGH",
         "High latency. Optimize routing path, check QoS policies, and verify backhaul utilization."),

        ("packet_loss",      ">",  4.0,  1, "PKT_LOSS_CRITICAL",
         "Critical packet loss. Inspect physical layer errors, check fiber integrity, and review transmission alarms."),

        ("packet_loss",      ">",  2.0,  2, "PKT_LOSS_HIGH",
         "Elevated packet loss. Check for interference sources, verify RLC retransmissions, and audit transport layer."),

        ("throughput_mbps",  "<",  5.0,  1, "THROUGHPUT_CRITICAL",
         "Throughput critically low. Check PRB utilization, scheduler settings, and verify bearer configuration."),

        ("throughput_mbps",  "<",  15.0, 2, "THROUGHPUT_LOW",
         "Low throughput. Review cell load, check MIMO configuration, and inspect downlink scheduler efficiency."),

        ("session_failures", ">",  40,   1, "SESSION_FAIL_CRITICAL",
         "Very high session failures. Check PDN gateway, inspect S1/X2 interface alarms, review bearer setup failures."),

        ("session_failures", ">",  15,   2, "SESSION_FAIL_HIGH",
         "Elevated session failures. Review attach/detach procedures, check authentication server load."),

        ("customer_complaints", ">", 20, 1, "COMPLAINTS_SURGE",
         "Customer complaint surge. Cross-check with field team, initiate emergency network audit, notify NOC."),
    ]

    def get_recommendations(self, row: pd.Series) -> list:
        recommendations = []
        for col, op, threshold, priority, code, message in self.RULES:
            val = row.get(col, None)
            if val is None:
                continue
            triggered = (op == ">" and val > threshold) or (op == "<" and val < threshold)
            if triggered:
                recommendations.append({
                    "code":     code,
                    "priority": priority,
                    "kpi":      col,
                    "value":    round(float(val), 3),
                    "threshold": threshold,
                    "message":  message
                })

        # Sort by priority (1=highest)
        recommendations.sort(key=lambda x: x["priority"])
        return recommendations[:3]  # Top 3 recommendations

    def get_summary_recommendation(self, row: pd.Series) -> str:
        recs = self.get_recommendations(row)
        if not recs:
            return "All KPIs within acceptable range. Continue monitoring."
        return " | ".join([r["message"][:80] + "..." for r in recs[:2]])


# ─────────────────────────────────────────────
# ALERT ENGINE
# ─────────────────────────────────────────────
class AlertEngine:
    """Generates, deduplicates and manages alerts."""

    def __init__(self, cooldown_minutes=15):
        self.cooldown_minutes = cooldown_minutes
        self._alert_history   = defaultdict(list)  # tower_id -> [timestamps]
        self.alerts_log       = []
        self._rec_engine      = RecommendationEngine()

    def _is_cooldown_active(self, tower_id: str) -> bool:
        now = datetime.now()
        recent = [t for t in self._alert_history[tower_id]
                  if (now - t).seconds < self.cooldown_minutes * 60]
        self._alert_history[tower_id] = recent
        return len(recent) > 0

    def evaluate_row(self, row: pd.Series) -> dict | None:
        risk = row.get("risk_level", "Low")
        if risk not in ("High", "Medium"):
            return None

        tower_id = row.get("tower_id", "UNKNOWN")
        if risk == "Medium" and self._is_cooldown_active(tower_id):
            return None  # Suppress duplicate medium alerts

        recs = self._rec_engine.get_recommendations(row)

        alert = {
            "alert_id":    f"ALT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{tower_id[-6:]}",
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "severity":    risk,
            "region":      row.get("region", "N/A"),
            "tower_id":    tower_id,
            "experience":  row.get("predicted_experience", "N/A"),
            "confidence":  round(float(row.get("confidence", 0)), 3),
            "prob_poor":   round(float(row.get("prob_poor", 0)), 3),
            "kpis": {
                "call_drop_rate":    round(float(row.get("call_drop_rate", 0)), 3),
                "cssr":              round(float(row.get("cssr", 0)), 3),
                "packet_loss":       round(float(row.get("packet_loss", 0)), 3),
                "latency_ms":        round(float(row.get("latency_ms", 0)), 2),
                "throughput_mbps":   round(float(row.get("throughput_mbps", 0)), 2),
                "session_failures":  int(row.get("session_failures", 0)),
                "customer_complaints": int(row.get("customer_complaints", 0)),
            },
            "recommendations": recs,
            "rule_triggered": row.get("rule_triggered", None)
        }

        self._alert_history[tower_id].append(datetime.now())
        self.alerts_log.append(alert)
        return alert

    def evaluate_batch(self, df: pd.DataFrame) -> list:
        alerts = []
        for _, row in df.iterrows():
            alert = self.evaluate_row(row)
            if alert:
                alerts.append(alert)
        return alerts

    def format_alert_text(self, alert: dict) -> str:
        sep = "═" * 60
        recs_text = "\n".join(
            [f"   [{i+1}] {r['code']}: {r['message'][:100]}"
             for i, r in enumerate(alert["recommendations"])]
        ) or "   No specific recommendation triggered."

        kpis = alert["kpis"]
        return f"""
{sep}
🚨  ALERT [{alert['severity'].upper()}]  |  {alert['timestamp']}
    Alert ID  : {alert['alert_id']}
    Region    : {alert['region']}
    Tower     : {alert['tower_id']}
    Prediction: {alert['experience']}  (conf: {alert['confidence']:.0%}, P(Poor): {alert['prob_poor']:.0%})
─────────────────────────────────────────────────────────────
    KPI Snapshot:
    • Call Drop Rate  : {kpis['call_drop_rate']:.2f}%
    • CSSR            : {kpis['cssr']:.2f}%
    • Packet Loss     : {kpis['packet_loss']:.2f}%
    • Latency         : {kpis['latency_ms']:.0f} ms
    • Throughput      : {kpis['throughput_mbps']:.1f} Mbps
    • Session Failures: {kpis['session_failures']}
    • Complaints      : {kpis['customer_complaints']}
─────────────────────────────────────────────────────────────
    Recommendations:
{recs_text}
{sep}"""

    def get_alerts_dataframe(self) -> pd.DataFrame:
        if not self.alerts_log:
            return pd.DataFrame()
        rows = []
        for a in self.alerts_log:
            row = {
                "alert_id":   a["alert_id"],
                "timestamp":  a["timestamp"],
                "severity":   a["severity"],
                "region":     a["region"],
                "tower_id":   a["tower_id"],
                "experience": a["experience"],
                "confidence": a["confidence"],
                "top_rec":    a["recommendations"][0]["message"][:80] if a["recommendations"] else "",
            }
            row.update(a["kpis"])
            rows.append(row)
        return pd.DataFrame(rows)
