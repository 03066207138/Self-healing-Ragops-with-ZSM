import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for models, pricing, and healing thresholds.
    Supports both Groq and OpenAI automatically.
    """

    # -----------------------------
    # 🔑 API Keys
    # -----------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # -----------------------------
    # 🤖 Model Settings
    # -----------------------------
    GEN_MODEL: str = os.getenv(
        "GEN_MODEL",
        "llama-3.3-70b-versatile" if os.getenv("GROQ_API_KEY") else "gpt-4o"
    )

    EMB_MODEL: str = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

    # -----------------------------
    # 💰 Pricing Configuration
    # -----------------------------
    PRICE_IN: float = float(os.getenv("PRICE_IN", "0.000002"))
    PRICE_OUT: float = float(os.getenv("PRICE_OUT", "0.000006"))

    # -----------------------------
    # 🩹 Healing & Anomaly Thresholds (Medium Aggressive)
    # -----------------------------
    COVERAGE_MIN: float = float(os.getenv("COVERAGE_MIN", "0.97"))
    FAITHFULNESS_MIN: float = float(os.getenv("FAITHFULNESS_MIN", "0.85"))
    LATENCY_SPIKE_PCT: float = float(os.getenv("LATENCY_SPIKE_PCT", "0.10"))
    COST_MAX: float = float(os.getenv("COST_MAX", "0.02"))

    # NEW threshold for drift
    DRIFT_MIN: float = float(os.getenv("DRIFT_MIN", "0.08"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
