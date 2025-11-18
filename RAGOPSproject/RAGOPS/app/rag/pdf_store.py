# app/rag/pdf_store.py

import os
import json
import uuid
from typing import List, Dict, Optional
from pypdf import PdfReader

import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

DATA_DIR = "app/data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
META_PATH = os.path.join(DATA_DIR, "qdrant_meta.json")
os.makedirs(PDF_DIR, exist_ok=True)

# ---------------------
# Embedding Model
# ---------------------
# -------------------------
# Load local model
# -------------------------
_model = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM = 384


def embed(texts: List[str]):
    arr = np.array(_model.encode(texts, convert_to_numpy=True))
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    return arr.astype("float32")


# ---------------------
# Qdrant Setup
# ---------------------
try:
    qdrant.get_collection("rag_chunks")
except:
    qdrant.create_collection(
        collection_name="rag_chunks",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


# ---------------------
# Metadata IO
# ---------------------
def _load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_meta(meta):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------
# Text chunking
# ---------------------
def chunk_text(text: str, chunk_size=800, overlap=120) -> List[str]:
    text = text.replace("\n", " ")
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks


# ---------------------
# Index PDF → Qdrant
# ---------------------
def build_index_for_pdf(doc_id: str, pdf_path: str) -> Dict:
    reader = PdfReader(pdf_path)

    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except:
            pages.append("")

    raw = " ".join(pages).strip()
    if not raw:
        raise ValueError("No extractable text in PDF.")

    chunks = chunk_text(raw)
    vecs = embed(chunks)

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "text": chunk
                }
            )
        )

    qdrant.upsert(collection_name="rag_chunks", points=points)

    meta = _load_meta()
    meta[doc_id] = {"chunks": len(chunks)}
    _save_meta(meta)

    print(f"[QDRANT] inserted {len(points)} chunks")

    return {"doc_id": doc_id, "chunks": len(chunks)}


# ---------------------
# Search (supports None doc_id = search all)
# ---------------------
def search_doc(doc_id: Optional[str], query: str, k: int = 5) -> List[Dict]:
    qvec = embed([query])[0]

    # If doc_id is None → search across all PDFs
    query_filter = None

    if doc_id not in (None, "", "all"):
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    hits = qdrant.search(
        collection_name="rag_chunks",
        query_vector=qvec,
        limit=k,
        query_filter=query_filter
    )

    return [
        {
            "doc_id": h.payload.get("doc_id"),
            "chunk_id": h.payload.get("chunk_id"),
            "text": h.payload.get("text"),
            "score": float(h.score)
        }
        for h in hits
    ]


# ============================================================
# Alias for compatibility with main.py
# ============================================================
def embed_texts(texts: List[str]):
    """
    Wrapper so main.py can import embed_texts instead of embed.
    """
    return embed(texts)
