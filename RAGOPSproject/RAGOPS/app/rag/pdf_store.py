# app/rag/pdf_store.py

import os
import uuid
from typing import List, Dict
from pathlib import Path

from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer

# Qdrant Embedded Mode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

# ============================================================
# 🧠 EMBEDDING MODEL
# ============================================================
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)
EMB_DIM = _model.get_sentence_embedding_dimension()


def embed(texts):
    """Return L2-normalized embeddings as float32."""
    if isinstance(texts, str):
        texts = [texts]

    vecs = _model.encode(texts, convert_to_numpy=True)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return vecs.astype("float32")


# ============================================================
# 📂 QDRANT EMBEDDED CLIENT (NO /var, ONLY LOCAL FOLDER)
# ============================================================
# IMPORTANT: this path must be writable on Render → use relative folder.
# It will live under /opt/render/project/src/qdrant_storage
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_storage")

# Make sure folder exists (relative, no /var)
Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)

qdrant = QdrantClient(path=QDRANT_PATH)

# Create collection if missing
try:
    qdrant.get_collection("rag_chunks")
except Exception:
    qdrant.create_collection(
        collection_name="rag_chunks",
        vectors_config=VectorParams(
            size=EMB_DIM,
            distance=Distance.COSINE,
        ),
    )


# ============================================================
# 📄 BUILD INDEX FOR PDF
# ============================================================
def build_index_for_pdf(doc_id: str, pdf_path: str) -> Dict[str, int]:
    """
    Read a PDF, chunk by page, embed with MiniLM, and upsert into Qdrant.
    doc_id: logical document identifier (e.g., filename without extension)
    pdf_path: filesystem path to the uploaded PDF
    """
    reader = PdfReader(pdf_path)
    chunks: List[str] = []

    for p in reader.pages:
        text = p.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(text)

    if not chunks:
        return {"chunks": 0}

    embeddings = embed(chunks)

    points: List[PointStruct] = []
    for idx, (txt, vec) in enumerate(zip(chunks, embeddings)):
        # Use a proper UUID so Qdrant never complains
        point_id = str(uuid.uuid4())

        points.append(
            PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "text": txt,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                },
            )
        )

    qdrant.upsert(
        collection_name="rag_chunks",
        points=points,
    )

    return {"chunks": len(points)}


# ============================================================
# 🔍 SEARCH INQDRANT
# ============================================================
def search_doc(doc_id: str, query: str, k: int = 5) -> List[Dict]:
    """
    Search relevant chunks for the query.
    Currently we ignore doc_id filter (global search).
    You can add a filter on doc_id later if you want per-document search.
    """
    qvec = embed(query)[0]

    hits = qdrant.search(
        collection_name="rag_chunks",
        query_vector=qvec,
        limit=k,
        # To filter by doc_id, uncomment this:
        # query_filter=models.Filter(
        #     must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        # )
    )

    out: List[Dict] = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "id": h.id,
                "score": float(h.score),
                "text": p.get("text", ""),
                "doc_id": p.get("doc_id"),
                "chunk_id": p.get("chunk_id"),
            }
        )

    return out
