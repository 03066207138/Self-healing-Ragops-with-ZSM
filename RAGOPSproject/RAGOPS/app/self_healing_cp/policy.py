"""
Adaptive Policy Management & TRiSM Enforcement
----------------------------------------------
Enhanced version for Agentic Self-Healing RAGOps 2.0
Integrates multi-agent orchestration (HealingAgent, GovernanceAgent, LearningAgent)
with explainable and adaptive policy governance.
"""

import re
import random
import math
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================
# 🔐 Prompt & Output Safety Filters (Enhanced)
# ============================================================

INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"<\s*system\s*>",
    r"@tool",
    r"run\s+shell",
    r"delete\s+all",
    r"sudo\s+|rm\s+-rf",
    r"base64\s+decode",
    r"curl\s+|wget\s+",
    r"powershell|cmd\.exe|os\.system",
    r"import\s+os|eval\(|exec\(",
]


def prompt_is_safe(user_text: str) -> bool:
    """Detect prompt injections or malicious instructions."""
    for pat in INJECTION_PATTERNS:
        if re.search(pat, user_text, flags=re.IGNORECASE):
            return False
    return True


def sanitize_output(model_output: str) -> str:
    """Cleans and redacts unsafe tokens or system injections from LLM responses."""
    cleaned = re.sub(r"<\s*/?\s*system\s*>", "", model_output)
    cleaned = re.sub(r"(sudo|@tool|run\s+shell|os\.system)", "[REDACTED]", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# ============================================================
# ⚙️ Adaptive Healing Policy Logic
# ============================================================

BASE_POLICIES: Dict[str, List[str]] = {
    "latency_spike": ["scale_out_cb"],
    "coverage_drop": ["increase_k_cb", "reindex_cb"],
    "faithfulness_drop": ["tighten_prompt_cb"],
    "retrieval_drift": ["switch_retriever_cb"],
    "cost_overrun": ["fallback_llm_cb"],
    "trust_degradation": ["shrink_context_cb", "fallback_llm_cb"],
}

CONFIDENCE_FACTORS = {
    "latency_spike": 0.8,
    "coverage_drop": 0.9,
    "faithfulness_drop": 0.85,
    "retrieval_drift": 0.75,
    "cost_overrun": 0.7,
    "trust_degradation": 0.65,
}


def get_policy_for_anomaly(anomaly_type: str, learning_agent: Optional[Any] = None) -> Dict[str, Any]:
    """
    Chooses best policy action using reinforcement-style scoring if LearningAgent is available.
    Returns policy with confidence and explainability metadata.
    """
    available = BASE_POLICIES.get(anomaly_type, [])
    if not available:
        return {
            "anomaly_type": anomaly_type,
            "actions": [],
            "confidence": 0.0,
            "explanation": "No predefined policy for this anomaly type."
        }

    # LearningAgent chooses adaptively; fallback to random
    chosen_action = (
        learning_agent.choose_action(anomaly_type, available)
        if learning_agent
        else random.choice(available)
    )

    confidence = CONFIDENCE_FACTORS.get(anomaly_type, 0.7)
    dynamic_boost = random.uniform(-0.05, 0.1)
    final_conf = min(1.0, max(0.0, confidence + dynamic_boost))

    return {
        "anomaly_type": anomaly_type,
        "actions": [chosen_action],
        "confidence": round(final_conf, 2),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# 🧠 TRiSM-Aware Governance & Risk Enforcement
# ============================================================

def action_allowed(action: str, governance_score: float) -> bool:
    """
    Restricts or delays risky actions if governance (trust) is low.
    """
    restricted_actions = {"scale_out_cb", "fallback_llm_cb"}
    if governance_score < 50 and action in restricted_actions:
        return False
    return True


def dynamic_trust_weight(governance_score: float, faithfulness: float, drift: float) -> float:
    """
    Adjusts confidence or learning weights based on TRiSM factors.
    Higher faithfulness and lower drift yield higher trust multiplier.
    """
    base_weight = max(0.3, min(1.0, governance_score / 100))
    stability_factor = (faithfulness * (1 - drift))
    weighted = base_weight * stability_factor
    return round(weighted, 2)


def evaluate_risk_policy(governance_score: float, faithfulness: float, halluc_rate: float) -> str:
    """Categorizes risk for dashboard visualization."""
    if governance_score < 40 or halluc_rate > 0.4:
        return "high_risk"
    elif faithfulness < 0.7 or governance_score < 70:
        return "moderate_risk"
    return "low_risk"


# ============================================================
# 🧩 Explainable Policy Summary (Human + Machine Readable)
# ============================================================

EXPLANATIONS = {
    "latency_spike": "Scaling compute resources to handle high latency or degraded response time.",
    "coverage_drop": "Increasing retriever depth (k) and refreshing document index for better coverage.",
    "faithfulness_drop": "Tightening prompt alignment and grounding for faithful responses.",
    "retrieval_drift": "Switching retriever or updating embeddings to reduce semantic drift.",
    "cost_overrun": "Switching to cost-efficient model or adaptive batching for optimization.",
    "trust_degradation": "Reducing risky context windows and fallback to verified data sources.",
}


def summarize_policy(
    anomaly_type: str,
    action: str,
    confidence: float,
    governance_score: float,
    faithfulness: float = 1.0,
    halluc_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Returns an explainable, TRiSM-compliant summary for the dashboard or healing log.
    """
    risk_level = evaluate_risk_policy(governance_score, faithfulness, halluc_rate)
    trust_weight = dynamic_trust_weight(governance_score, faithfulness, drift=0.05)
    explain_text = EXPLANATIONS.get(anomaly_type, "Generic adaptive policy action executed.")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "anomaly_type": anomaly_type,
        "selected_action": action,
        "confidence": round(confidence, 2),
        "governance_score": round(governance_score, 2),
        "trust_weight": trust_weight,
        "risk_level": risk_level,
        "explanation": explain_text,
        "allowed": action_allowed(action, governance_score),
        "policy_signature": f"{anomaly_type}:{action}:{round(confidence,2)}",
    }

    return summary


# ============================================================
# 🧾 Policy Audit Trail Generator
# ============================================================

def generate_audit_entry(policy_summary: Dict[str, Any]) -> str:
    """
    Converts policy summary into a log-friendly audit entry (JSON line).
    Example: Stored into healing_audit.jsonl
    """
    entry = (
        f"[{policy_summary['timestamp']}] "
        f"Policy={policy_summary['policy_signature']} | "
        f"Risk={policy_summary['risk_level']} | "
        f"Trust={policy_summary['trust_weight']} | "
        f"Allowed={policy_summary['allowed']} | "
        f"Explanation={policy_summary['explanation']}"
    )
    return entry
