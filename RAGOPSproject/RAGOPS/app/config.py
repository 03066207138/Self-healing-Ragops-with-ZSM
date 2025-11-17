from fastapi import FastAPI, Query
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Dict, Any
import os
import pandas as pd

# -------------------- Internal Imports --------------------
from app.settings import settings
from app.self_healing_cp.metrics_logger import MetricsLogger
from app.rag.pipeline import RAGPipeline, embed_texts
from app.self_healing_cp import telemetry, anomaly, orchestrator, policy
from app.rag.pdf_router import router as pdf_router  # ← NEW: wire PDF endpoints

# -------------------- Initialize App --------------------
app = FastAPI(title="Self-Healing RAGOps (LLM-integrated)")

# ✅ CORS (Vite 5173 + CRA 3000)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000", "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Core Components --------------------
pipeline = RAGPipeline(corpus_path="app/data/corpus.json", k=5)
logger = MetricsLogger("metrics_log.csv")

thresholds = {
    "latency_spike_pct": settings.LATENCY_SPIKE_PCT,
    "coverage_min": settings.COVERAGE_MIN,
    "faithfulness_min": settings.FAITHFULNESS_MIN,
    "cost_max": settings.COST_MAX
}

stats = anomaly.RollingStats(window=100)
_last_row: Dict[str, Any] = {}

# -------------------- Models --------------------
class QueryIn(BaseModel):
    query: str
    k: int | None = None

# -------------------- Routers --------------------
# Mount the PDF router so /pdfs/* routes exist
app.include_router(pdf_router)

# -------------------- Routes --------------------
@app.get("/")
def root():
    """Simple root to avoid 404 when opening backend in browser."""
    return {"ok": True, "service": "Self-Healing RAGOps API"}

@app.get("/health")
def health():
    """Health check for backend and model connectivity"""
    ok = bool(settings.OPENAI_API_KEY or getattr(settings, "GROQ_API_KEY", ""))  # supports OpenAI/Groq
    return {
        "status": "ok" if ok else "missing_api_key",
        "gen_model": settings.GEN_MODEL,
        "emb_model": settings.EMB_MODEL,
        "retriever": getattr(pipeline, "retriever_type", "vector")
    }

@app.post("/query")
def query(payload: QueryIn):
    """Main query endpoint — runs RAG pipeline and detects anomalies"""
    q = payload.query.strip()
    if not q:
        return {"error": "Query text is empty."}

    # 1) Policy guard
    if not policy.prompt_is_safe(q):
        return {"error": "Unsafe prompt detected by policy guard."}

    # 2) Update retrieval depth if provided
    if payload.k:
        pipeline.k = payload.k

    # 3) Run pipeline
    answer, passages, tokens_in, tokens_out, latency_ms = pipeline.run(q)

    # Debug retrieved passages
    print("\n🔎 Retrieved passages:")
    for p in passages:
        print(f"  ID: {p.get('id', 'N/A')}, Score: {p.get('score', 'N/A')}")

    # 4) Metrics
    texts = [p["text"] for p in passages]
    cov = telemetry.coverage_at_k(answer, texts, embed_texts, sim_threshold=0.75)
    faith = telemetry.faithfulness(answer, texts, embed_texts, sim_threshold=0.80)
    cost = telemetry.estimate_cost_usd(tokens_in, tokens_out, settings.PRICE_IN, settings.PRICE_OUT)

    # 5) Row
    row = {
        "request_id": f"req_{int(datetime.now().timestamp())}",
        "ts_start": datetime.now(timezone.utc),
        "ts_end": datetime.now(timezone.utc),
        "latency_ms": float(latency_ms),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_usd": float(cost),
        "k": len(passages),
        "retrieved_doc_ids": ",".join(str(p.get("id", "")) for p in passages),
        "coverage_at_k": float(cov),
        "grounding_faithfulness": float(faith),
        "retriever_type": getattr(pipeline, "retriever_type", "vector")
    }

    # 6) Anomaly
    stats.update(row["latency_ms"], row["grounding_faithfulness"], row["coverage_at_k"], row["cost_usd"])
    info = anomaly.detect(row, stats, thresholds)
    row.update(info)

    # 7) Healing
    if row.get("anomaly_detected", False):
        row["ts_detection"] = datetime.now(timezone.utc)
        callbacks = {
            "reindex_cb": pipeline.reindex,
            "increase_k_cb": pipeline.increase_k,
            "switch_retriever_cb": pipeline.switch_retriever,
            "tighten_prompt_cb": pipeline.tighten_prompt,
            "fallback_llm_cb": pipeline.fallback_llm,
            "scale_out_cb": pipeline.scale_out,
            "shrink_context_cb": pipeline.shrink_context,
        }
        orchestrator.orchestrate(row["anomaly_type"], callbacks)
        row["ts_recovery_end"] = datetime.now(timezone.utc)
        row["mttr_s"] = (row["ts_recovery_end"] - row["ts_detection"]).total_seconds()
        row["healing_action"] = row["anomaly_type"]
        row["status_after_heal"] = "healthy"

    # 8) Persist + cache
    logger.log(row)
    global _last_row
    _last_row = row

    # 9) Response
    return {
        "answer": answer,
        "passages": passages[:pipeline.max_ctx],
        "metrics": {
            "latency_ms": row["latency_ms"],
            "coverage_at_k": row["coverage_at_k"],
            "grounding_faithfulness": row["grounding_faithfulness"],
            "cost_usd": row["cost_usd"],
            "retriever_type": row["retriever_type"],
            "anomaly": row.get("anomaly_type", "")
        }
    }

@app.get("/metrics/summary")
def metrics_summary():
    """
    Robust summary that tolerates malformed CSV rows and forces a fixed schema.
    """
    path = "metrics_log.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}

    expected = [
        "request_id", "ts_start", "ts_end", "latency_ms",
        "tokens_in", "tokens_out", "cost_usd",
        "k", "retrieved_doc_ids", "coverage_at_k",
        "grounding_faithfulness", "retriever_type",
        "anomaly_detected", "anomaly_type", "ts_detection",
        "ts_recovery_end", "mttr_s", "healing_action", "status_after_heal"
    ]

    # Read without trusting the first row as header and skip any bad lines
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        header=None,
        names=expected,            # force our schema
        quotechar='"',
        sep=",",
        keep_default_na=False
    )

    # Drop any stray header line that may have been appended as data
    df = df[df["request_id"] != "request_id"]

    # Coerce numeric columns
    num_cols = ["latency_ms", "tokens_in", "tokens_out", "cost_usd",
                "coverage_at_k", "grounding_faithfulness", "mttr_s"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.empty:
        return {}

    agg = {
        "latency_ms": "mean",
        "cost_usd": "mean",
        "coverage_at_k": "mean",
        "grounding_faithfulness": "mean",
        "mttr_s": "mean",
    }
    out = df.agg(agg).rename({
        "latency_ms": "Avg Latency (ms)",
        "cost_usd": "Avg Cost (USD)",
        "coverage_at_k": "Avg Coverage@K",
        "grounding_faithfulness": "Avg Faithfulness",
        "mttr_s": "Avg MTTR (s)",
    })

    # Return clean floats (no NaN)
    return {k: (0.0 if pd.isna(v) else float(v)) for k, v in out.to_dict().items()}

@app.get("/metrics/last")
def metrics_last():
    """Return last recorded metrics entry"""
    return _last_row or {}

@app.post("/heal")
def heal(action: str = Query(..., description="healing action")):
    """Manual healing endpoint — invoke one of the self-healing actions"""
    actions = {
        "increase_k": pipeline.increase_k,
        "reindex": pipeline.reindex,
        "switch_retriever": pipeline.switch_retriever,
        "tighten_prompt": pipeline.tighten_prompt,
        "fallback_llm": pipeline.fallback_llm,
        "shrink_context": pipeline.shrink_context,
        "scale_out": pipeline.scale_out
    }
    if action not in actions:
        return {"status": "error", "message": f"Unknown action: {action}"}
    actions[action]()
    return {"status": "ok", "applied": action}
