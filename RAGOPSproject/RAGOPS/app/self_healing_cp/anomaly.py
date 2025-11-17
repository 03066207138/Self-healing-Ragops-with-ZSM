"""
Stable Anomaly Detection for Self-Healing RAGOps
------------------------------------------------
This version:
- fixes indentation errors
- prevents repeated hallucination_spike spam
- raises thresholds to avoid false positives
- applies priority-order detection
"""

from collections import deque
from typing import Dict, Any, Optional
import numpy as np


# ============================================================
# Rolling Stats
# ============================================================
class RollingStats:
    def __init__(self, window: int = 100):
        self.lat = deque(maxlen=window)
        self.cov = deque(maxlen=window)
        self.faith = deque(maxlen=window)
        self.cost = deque(maxlen=window)
        self.halluc = deque(maxlen=window)
        self.drift = deque(maxlen=window)

    def update(self, latency, faith, cov, cost, halluc, drift):
        self.lat.append(float(latency))
        self.cov.append(float(cov))
        self.faith.append(float(faith))
        self.cost.append(float(cost))
        self.halluc.append(float(halluc))
        self.drift.append(float(drift))

    def p95_latency(self):
        if not self.lat:
            return None
        arr = sorted(self.lat)
        idx = int(0.95 * (len(arr) - 1))
        return arr[idx]


# ============================================================
# Main Detection
# ============================================================
def detect(row: Dict[str, Any], stats: RollingStats, thresholds: Dict[str, float]):

    info = {"anomaly_detected": False, "anomaly_type": "—", "confidence": 0.0}

    latency = float(row.get("latency_ms", 0))
    cov = float(row.get("coverage_at_k", 1))
    faith = float(row.get("faithfulness", 1))
    halluc = float(row.get("hallucination_rate", 0))
    drift = float(row.get("semantic_drift", 0))
    cost = float(row.get("cost_usd", 0))
    gov = float(row.get("governance_score", 100))

    stats.update(latency, faith, cov, cost, halluc, drift)

    # Priority 1 — Critical hallucination
    if halluc >= 0.20:   # raised threshold
        return {
            "anomaly_detected": True,
            "anomaly_type": "hallucination_spike",
            "confidence": 0.85
        }

    # Priority 2 — Retrieval drift
    if drift > 0.10:  # previously 0.06
        return {
            "anomaly_detected": True,
            "anomaly_type": "retrieval_drift",
            "confidence": 0.80
        }

    # Priority 3 — Faithfulness drop
    if faith < thresholds.get("faithfulness_min", 0.85):
        return {
            "anomaly_detected": True,
            "anomaly_type": "faithfulness_drop",
            "confidence": 0.75
        }

    # Priority 4 — Coverage low
    if cov < thresholds.get("coverage_min", 0.95):
        return {
            "anomaly_detected": True,
            "anomaly_type": "coverage_drop",
            "confidence": 0.70
        }

    # Priority 5 — Latency
    if len(stats.lat) >= 15:
        p95 = stats.p95_latency()
        if p95 and latency > p95 * 1.50:
            return {
                "anomaly_detected": True,
                "anomaly_type": "latency_spike",
                "confidence": 0.75
            }

    # Priority 6 — Cost
    if cost > thresholds.get("cost_max", 0.03):
        return {
            "anomaly_detected": True,
            "anomaly_type": "cost_overrun",
            "confidence": 0.72
        }

    # Priority 7 — Governance
    if gov < 50:  # lower threshold
        return {
            "anomaly_detected": True,
            "anomaly_type": "trust_degradation",
            "confidence": 0.65
        }

    return info


# ============================================================
# Rule-Based Type Detection (Optional)
# ============================================================
def detect_anomaly_type(metrics: Dict[str, Any]) -> Optional[str]:
    rules = {
        "hallucination_spike": lambda m: m.get("hallucination_rate", 0) > 0.15,
        "retrieval_drift": lambda m: m.get("semantic_drift", 0) > 0.10,
        "faithfulness_drop": lambda m: m.get("faithfulness", 1) < 0.88,
        "coverage_drop": lambda m: m.get("coverage_at_k", 1) < 0.95,
        "latency_spike": lambda m: m.get("latency_ms", 0) > 3000,
    }

    for atype, cond in rules.items():
        if cond(metrics):
            return atype

    return None
