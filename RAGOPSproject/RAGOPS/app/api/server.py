# app/api/server.py
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
import numpy as np

from app.rag.pipeline import RAGPipeline
from app.api import db

# -------------------- Initialize --------------------
db.init_db()
app = FastAPI(title="RAGOps Local API")
rag = RAGPipeline()
session = db.SessionLocal()

# -------------------- Routes --------------------
@app.post("/add_document")
def add_document(text: str):
    """Add a new document to the database and update FAISS"""
    try:
        # Store document
        embedding = rag.retriever._embed_texts([text])[0].tobytes()
        doc = db.Document(text=text, embedding=embedding)
        session.add(doc)
        session.commit()

        # Rebuild FAISS index
        rag.reindex()
        return {"status": "ok", "id": doc.id}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_endpoint(query: str, k: int = 5):
    """Run RAG pipeline and store metrics"""
    try:
        rag.k = k
        answer, passages, tokens_in, tokens_out, latency_ms = rag.run(query)

        # Save metrics
        cov, faith = 0.0, 0.0  # Optional: compute coverage/faithfulness
        metric = db.Metric(
            query=query,
            latency_ms=latency_ms,
            coverage=cov,
            faithfulness=faith,
            cost=0.0  # Optional: add cost calculation
        )
        session.add(metric)
        session.commit()

        return {
            "answer": answer,
            "passages": passages[:rag.max_ctx],
            "latency_ms": latency_ms
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents():
    """List all stored documents"""
    docs = session.query(db.Document).all()
    return [{"id": d.id, "text": d.text} for d in docs]

@app.get("/metrics")
def list_metrics():
    """List all metrics"""
    metrics = session.query(db.Metric).all()
    return [{
        "query": m.query,
        "latency_ms": m.latency_ms,
        "coverage": m.coverage,
        "faithfulness": m.faithfulness,
        "cost": m.cost,
        "created_at": m.created_at.isoformat()
    } for m in metrics]
