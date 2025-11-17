"""
Agentic Self-Healing Orchestrator (Unified Version)
---------------------------------------------------
Compatible with new main.py + anomaly.py
Correctly triggers actions and rewards.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)

# ============================================================
# 🔧 Healing Policies (MUST MATCH anomaly.anomaly_type)
# ============================================================
HEALING_POLICIES: Dict[str, List[str]] = {
    "latency_spike": ["increase_k_cb"],
    "coverage_drop": ["increase_k_cb", "reindex_cb"],
    "faithfulness_drop": ["tighten_prompt_cb"],
    "hallucination_spike": ["tighten_prompt_cb"],
    "retrieval_drift": ["switch_retriever_cb"],
    "cost_overrun": ["fallback_llm_cb"],
    "trust_degradation": ["tighten_prompt_cb"],   # FIXED 💡
}

# ============================================================
# 🤖 Reinforcement Learning Agent
# ============================================================
class LearningAgent:
    def __init__(self):
        self.q_table: Dict[str, float] = {}

    def choose_action(self, anomaly_type: str, possible_actions: List[str]) -> str:
        if not possible_actions:
            return None

        # Exploration
        if random.random() < 0.1:
            return random.choice(possible_actions)

        # Exploitation (Q-learning)
        scores = {a: self.q_table.get(f"{anomaly_type}:{a}", 0.0) for a in possible_actions}
        return max(scores, key=scores.get)

    def update_reward(self, anomaly_type: str, action: str, reward: float):
        key = f"{anomaly_type}:{action}"
        old = self.q_table.get(key, 0.0)
        new = old + 0.5 * (reward - old)
        self.q_table[key] = new
        logger.info(f"[RL] Updated Q({key}) = {new:.3f}")

# ============================================================
# 🛡 Governance Agent
# ============================================================
class GovernanceAgent:
    RESTRICTED = ["scale_out_cb", "fallback_llm_cb"]

    def is_allowed(self, action: str):
        return action not in self.RESTRICTED

# ============================================================
# 🔧 Safe Callback Executor
# ============================================================
def run_callback(name: str, cb: Callable):
    try:
        logger.info(f"[heal] Running callback '{name}'")
        result = cb()
        logger.info(f"[heal] SUCCESS '{name}' → {result}")
        return True, result
    except Exception as e:
        logger.error(f"[heal] FAILED '{name}' → {e}")
        return False, str(e)

# ============================================================
# 🧠 Healing Agent
# ============================================================
class HealingAgent:
    def __init__(self):
        self.learning = LearningAgent()
        self.gov = GovernanceAgent()
        self.history: List[Dict] = []

    def heal(self, before_metrics: Dict[str, Any], callbacks: Dict[str, Callable]):

        anomaly_type = before_metrics.get("anomaly_type")

        # If no anomaly → skip
        if anomaly_type in [None, "—"]:
            return {"status": "healthy", "actions": []}

        logger.warning(f"[detect] Anomaly detected → {anomaly_type}")

        # Plan actions based on policy
        possible_actions = HEALING_POLICIES.get(anomaly_type, [])

        if not possible_actions:
            return {
                "status": "healthy",
                "actions": [],
                "message": f"No policy for anomaly '{anomaly_type}'"
            }

        # RL selects best action
        action = self.learning.choose_action(anomaly_type, possible_actions)

        # TRiSM restricts unsafe actions
        if not self.gov.is_allowed(action):
            logger.warning(f"[TRiSM] Action '{action}' blocked → fallback to 'increase_k_cb'")
            action = "increase_k_cb"

        # Execute action
        cb = callbacks.get(action)
        if not cb:
            return {
                "status": "failed",
                "actions": [action],
                "message": "Callback not implemented"
            }

        ok, result = run_callback(action, cb)

        # Reward logic
        reward = 1.0 if ok else -0.3
        self.learning.update_reward(anomaly_type, action, reward)

        summary = {
            "anomaly": anomaly_type,
            "actions": [action],
            "healing_action": action,
            "reward": round(reward, 3),
            "status_after_heal": "healthy" if ok else "failed",
            "timestamp": datetime.now().isoformat()
        }

        self.history.append(summary)
        return summary
