# app/rag/pdf_store.py

import os
import uuid
from typing import List, Dict
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
# 🔤 EMBEDDING MODEL (MiniLM)
# ============================================================
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)
EMB_DIM = _model.get_sentence_embedding_dimension()


def embed(texts):
    """
    Convert texts to normalized embeddings.
    """
    if isinstance(texts, str):
        texts = [texts]

    vecs = _model.encode(texts, convert_to_numpy=True)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return vecs.astype("float32")


# ============================================================
# 🗄️ PERSISTENT QDRANT STORAGE (REQUIRED FOR RENDER)
# ============================================================
# Render persistent disk MUST be mounted to /var/data/qdrant
QDRANT_PATH = "/var/data/qdrant"
os.makedirs(QDRANT_PATH, exist_ok=True)

qdrant = QdrantClient(path=QDRANT_PATH)

# Create collection if not exists
if "rag_chunks" not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name="rag_chunks",
        vectors_config=VectorParams(
            size=EMB_DIM,
            distance=Distance.COSINE,
        )
    )


# ============================================================
# 📄 BUILD INDEX FOR PDF → Qdrant
# ============================================================
def build_index_for_pdf(doc_id: str, pdf_path: str) -> Dict[str, int]:
    """
    Extract text per page → embed → store in Qdrant.
    """
    reader = PdfReader(pdf_path)

    chunks = []
    for p in reader.pages:
        text = p.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(text)

    # If no extractable text
    if not chunks:
        return {"chunks": 0}

    embeddings = embed(chunks)

    points = []
    for idx, (txt, vec) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # Safe unique ID
                vector=vec,
                payload={
                    "text": txt,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                },
            )
        )

    # Save in Qdrant
    qdrant.upsert(
        collection_name="rag_chunks",
        points=points,
    )

    return {"chunks": len(points)}


# ============================================================
# 🔎 SEARCH RETRIEVAL
# ============================================================
def search_doc(doc_id: str, query: str, k=5) -> List[Dict]:
    qvec = embed(query)[0]

    hits = qdrant.search(
        collection_name="rag_chunks",
        query_vector=qvec,
        limit=k,
    )

    results = []
    for h in hits:
        payload = h.payload or {}

        results.append({
            "id": h.id,
            "score": float(h.score),
            "text": payload.get("text", ""),
            "doc_id": payload.get("doc_id", ""),
            "chunk_id": payload.get("chunk_id", 0),
        })

    return results
