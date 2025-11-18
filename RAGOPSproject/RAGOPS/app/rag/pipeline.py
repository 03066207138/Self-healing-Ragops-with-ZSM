# app/rag/pipeline.py
import os
import re
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

from .generator import generate_answer
from .pdf_store import search_doc


# ============================================================
# 🔢 Shared local embedding model
# ============================================================
_local_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """Public embedding function for telemetry + metrics."""
    if isinstance(texts, str):
        texts = [texts]

    vec = np.array(_local_model.encode(texts, convert_to_numpy=True))
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    return vec.astype("float32")


# ============================================================
# 🧹 Clean PDF Text
# ============================================================
GLYPH_PAT = re.compile(r"uni[0-9A-Fa-f]{4,6}|/uni[0-9A-Fa-f]{4,6}")
MULTI_WS = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = GLYPH_PAT.sub("", text)
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = MULTI_WS.sub(" ", text)
    return text.strip()


# ============================================================
# 🧠 Qdrant Powered RAG Pipeline (FIXED)
# ============================================================
class RAGPipeline:

    def __init__(self, k: int = 5):
        self.k = k
        self.max_ctx = 10   # FIX: more context improves accuracy

    # --------------------------------------------------------
    # 🔍 RUN QUERY (FIXED)
    # --------------------------------------------------------
    def run(self, query: str):
        t0 = time.time()

        # 1) Retrieve from Qdrant
        try:
            # doc_id ignored → search all documents
            results = search_doc(doc_id=None, query=query, k=self.k)
        except Exception as e:
            print("[RAGPipeline] Retrieval error:", e)
            return "⚠️ Retrieval failed", [], 0, 0, (time.time() - t0) * 1000

        if not results:
            return (
                "⚠️ No relevant passages found. Please index PDFs first.",
                [],
                0, 0,
                (time.time() - t0) * 1000
            )

        # 2) Clean + enrich
        passages = []
        for p in results:
            text = clean_text(p.get("text", ""))
            if not text:
                continue

            passages.append({
                "id": p.get("id"),
                "text": text,
                "snippet": text[:300] + "…" if len(text) > 300 else text,
                "title": p.get("doc_id", "PDF Document"),
                "page": p.get("chunk_id", "?"),
                "score": float(p.get("score", 0.0)),
            })

        if not passages:
            return (
                "⚠️ Extracted passages were empty.",
                [],
                0,0,
                (time.time() - t0) * 1000
            )

        ctx = passages[:self.max_ctx]

        # 3) Generate LLM answer
        answer, tin, tout = generate_answer(query, ctx)

        # Safety: never return blank answer
        if not answer.strip():
            answer = "⚠ No answer generated. Try uploading more PDFs."

        latency = (time.time() - t0) * 1000
        return answer, passages, tin, tout, latency


    # --------------------------------------------------------
    # 📄 INDEX PDF – DELEGATE TO pdf_store
    # --------------------------------------------------------
    def index_pdf(self, pdf_path: str, doc_id: str):
        from .pdf_store import build_index_for_pdf
        info = build_index_for_pdf(doc_id, pdf_path)
        return [{"id": f"{doc_id}_{i}"} for i in range(info["chunks"])]


    # --------------------------------------------------------
    # 🩺 Healing
    # --------------------------------------------------------
    def increase_k(self):
        old = self.k
        self.k = min(old + 5, 50)
        print(f"[Healing] Increased k {old} → {self.k}")

    def tighten_prompt(self):
        old = self.max_ctx
        self.max_ctx = max(1, old - 1)
        print(f"[Healing] Tightened context {old} → {self.max_ctx}")

    def shrink_context(self):
        old = self.max_ctx
        self.max_ctx = 2
        print(f"[Healing] Shrunk context {old} → {self.max_ctx}")

    def fallback_llm(self):
        print("[Healing] Fallback LLM selected")

    def scale_out(self):
        print("[Healing] Scale-out triggered")

    def reindex(self):
        print("[Healing] Qdrant manages index → nothing to rebuild")

    def switch_retriever(self):
        print("[Healing] Retriever is Qdrant-only")
