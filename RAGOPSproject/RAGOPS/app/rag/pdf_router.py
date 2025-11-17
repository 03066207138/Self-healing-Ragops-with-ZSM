# app/rag/pdf_router.py
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from .self_healing_cp import telemetry, anomaly, orchestrator, policy
from .self_healing_cp.metrics_logger import MetricsLogger
from .self_healing_cp.learner import LearningMemory  # ⭐ NEW
from .settings import settings

from .rag.generator import generate_answer
from .rag.pdf_store import PDF_DIR, list_docs, build_index_for_pdf, search_doc
from .rag.pipeline import embed_texts

# --- Optional extras ---
_HAS_HYBRID = False
_HAS_RERANK = False

try:
    from app.rag.pdf_store import hybrid_search_doc
    _HAS_HYBRID = True
except Exception:
    pass

try:
    import numpy as np
    from sentence_transformers import CrossEncoder
    _cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _HAS_RERANK = True
except Exception:
    _cross = None

router = APIRouter(prefix="/pdfs", tags=["pdfs"])

logger = MetricsLogger("metrics_log.csv")
stats = anomaly.RollingStats(window=100)

learning = LearningMemory()   # ⭐ NEW — Learning Layer instance


# ===========================
# Models
# ===========================
class QueryIn(BaseModel):
    query: str
    k: int | None = 8


# ===========================
# Endpoints
# ===========================
@router.get("/")
def list_pdfs():
    return list_docs()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe = "".join(c for c in file.filename if c.isalnum() or c in ("-", "_", ".")).strip(".")
    if not safe:
        safe = f"doc_{int(time.time())}.pdf"

    os.makedirs(PDF_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DIR, safe)
    data = await file.read()
    with open(save_path, "wb") as f:
        f.write(data)

    doc_id = os.path.splitext(safe)[0]
    info = build_index_for_pdf(doc_id, save_path)
    return {"status": "ok", **info}


# ===========================
# Helpers
# ===========================
def _dedupe_keep_order(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in passages:
        pid = p.get("id")
        if pid not in seen:
            out.append(p)
            seen.add(pid)
    return out


def _rerank_with_crossencoder(q: str, cands: List[Dict[str, Any]], keep: int = 12):
    if not (_HAS_RERANK and _cross and cands):
        return cands
    pairs = [(q, p["text"]) for p in cands]
    scores = _cross.predict(pairs)
    order = list(np.argsort(-np.array(scores)))[: min(keep, len(cands))]
    out = []
    for i in order:
        cp = dict(cands[i])
        cp["score"] = float(scores[i])
        out.append(cp)
    return out


# ===========================
# Main Query Endpoint
# ===========================
@router.post("/{doc_id}/query")
def ask_pdf(doc_id: str, payload: QueryIn):
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query text is empty.")

    if not policy.prompt_is_safe(q):
        raise HTTPException(status_code=400, detail="Unsafe prompt detected.")

    k = payload.k or 8

    t0 = time.time()
    try:
        passages = search_doc(doc_id, q, k=k)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{doc_id}' not found or not indexed yet."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # --------------------------
    # FIRST ANSWER
    # --------------------------
    if not passages:
        answer, tokens_in, tokens_out = generate_answer(q, [], allow_citations=False)
    else:
        answer, tokens_in, tokens_out = generate_answer(q, passages[:5], allow_citations=True)

    latency_ms = (time.time() - t0) * 1000.0

    texts = [p["text"] for p in passages] if passages else []
    cov = telemetry.coverage_at_k(answer, texts, embed_texts, 0.60) if texts else 0.0
    faith = telemetry.faithfulness(answer, texts, embed_texts, 0.65) if texts else 0.0
    cost = telemetry.estimate_cost_usd(tokens_in, tokens_out, settings.PRICE_IN, settings.PRICE_OUT)

    row: Dict[str, Any] = {
        "request_id": f"req_{int(datetime.now().timestamp())}",
        "ts_start": datetime.now(timezone.utc),
        "ts_end": datetime.now(timezone.utc),
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
        "k": len(passages),
        "retrieved_doc_ids": ",".join(str(p["id"]) for p in passages) if passages else "",
        "coverage_at_k": cov,
        "grounding_faithfulness": faith,
        "retriever_type": f"pdf:{doc_id}",
        "healing_action": "" if passages else "no_context_fallback",
    }

    healed_from = None

    # =============================
    # Healing helper functions
    # =============================
    def _regenerate_and_apply(ctx: List[Dict], reason: str):
        nonlocal answer, tokens_in, tokens_out, cov, faith, row, passages, healed_from

        a2, ti2, to2 = generate_answer(q, ctx[:5], allow_citations=True)
        texts2 = [p["text"] for p in ctx]

        cov2 = telemetry.coverage_at_k(a2, texts2, embed_texts, 0.60) if texts2 else 0.0
        faith2 = telemetry.faithfulness(a2, texts2, embed_texts, 0.65) if texts2 else 0.0

        if (cov2 > cov) or row["healing_action"] == "no_context_fallback":
            healed_from = {"coverage": cov, "faithfulness": faith}

            answer = a2
            tokens_in = int(ti2)
            tokens_out = int(to2)
            cov = cov2
            faith = faith2

            passages = ctx
            row["k"] = len(passages)
            row["retrieved_doc_ids"] = ",".join(str(x["id"]) for x in passages)
            row["coverage_at_k"] = cov
            row["grounding_faithfulness"] = faith

            row["healing_action"] = (
                row["healing_action"] + ">" if row["healing_action"] else ""
            ) + reason

            row["status_after_heal"] = "healthy"
            return True
        return False

    def _heal_increase_k():
        nonlocal passages
        k2 = min(32, max(k, 8) * 2)
        more = search_doc(doc_id, q, k=k2)
        if more:
            more = _dedupe_keep_order(more)
            _regenerate_and_apply(more, "increase_k")

    def _heal_hybrid():
        if not _HAS_HYBRID:
            return
        hybrid = hybrid_search_doc(doc_id, q, k=32, alpha=0.6)
        if hybrid:
            hybrid = _dedupe_keep_order(hybrid)
            _regenerate_and_apply(hybrid, "hybrid")

    def _heal_rerank():
        nonlocal passages
        if passages:
            ranked = _rerank_with_crossencoder(q, passages, keep=12)
            if ranked and ranked != passages:
                _regenerate_and_apply(ranked, "rerank")

    def _heal_tighten_prompt():
        if passages:
            tight = passages[:2]
            _regenerate_and_apply(tight, "tighten_prompt")

    # ===========================
    # Anomaly Detection
    # ===========================
    thresholds = {
        "latency_spike_pct": settings.LATENCY_SPIKE_PCT,
        "coverage_min": settings.COVERAGE_MIN,
        "faithfulness_min": settings.FAITHFULNESS_MIN,
        "cost_max": settings.COST_MAX,
    }

    stats.update(row["latency_ms"], row["grounding_faithfulness"], row["coverage_at_k"], row["cost_usd"])
    info = anomaly.detect(row, stats, thresholds)
    row.update(info)

    # ===========================
    # Healing Pipeline
    # ===========================
    if row.get("anomaly_detected"):
        row["ts_detection"] = datetime.now(timezone.utc)

        if row["anomaly_type"] == "coverage_drop":
            _heal_increase_k()
            _heal_hybrid()
            _heal_rerank()

        elif row["anomaly_type"] == "faithfulness_drop":
            _heal_tighten_prompt()

        orchestrator.orchestrate(
            row["anomaly_type"],
            callbacks={
                "increase_k_cb": _heal_increase_k,
                "tighten_prompt_cb": _heal_tighten_prompt,
                "fallback_llm_cb": lambda: None,
                "scale_out_cb": lambda: None,
                "shrink_context_cb": lambda: None,
                "reindex_cb": lambda: None,
                "switch_retriever_cb": lambda: None,
            },
        )

        row["ts_recovery_end"] = datetime.now(timezone.utc)
        row["mttr_s"] = (row["ts_recovery_end"] - row["ts_detection"]).total_seconds()

    # ===========================
    # Log
    # ===========================
    logger.log(row)

    # ===========================
    # ⭐ Learning Layer ⭐
    # ===========================
    query_text = q
    model_answer = answer

    if not passages:
        learning.add_failure(query_text, model_answer, reason="no_retrieval")
    else:
        if row["grounding_faithfulness"] < 0.50:
            learning.add_failure(query_text, model_answer, reason="low_faithfulness")

    if row.get("healing_action"):
        learning.add_verified_qa(query_text, model_answer)

    # ===========================
    # Response
    # ===========================
    metrics = {
        "latency_ms": row["latency_ms"],
        "coverage_at_k": row["coverage_at_k"],
        "grounding_faithfulness": row["grounding_faithfulness"],
        "cost_usd": row["cost_usd"],
        "retriever_type": row["retriever_type"],
        "anomaly": row.get("anomaly_type", ""),
        "healing_action": row.get("healing_action", "")
    }

    if healed_from is not None:
        metrics["healed_from"] = healed_from

    return {
        "answer": answer,
        "passages": passages[:5] if passages else [],
        "metrics": metrics,
    }
