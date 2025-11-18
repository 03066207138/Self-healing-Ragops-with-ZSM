# app/rag/pdf_store.py

import os
import json
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

# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)
EMB_DIM = _model.get_sentence_embedding_dimension()


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    vecs = _model.encode(texts, convert_to_numpy=True)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return vecs.astype("float32")


# -----------------------------------
# EMBEDDED QDRANT CLIENT
# -----------------------------------
qdrant = QdrantClient(path="qdrant_storage")

# Create collection if missing
try:
    qdrant.get_collection("rag_chunks")
except:
    qdrant.create_collection(
        collection_name="rag_chunks",
        vectors_config=VectorParams(
            size=EMB_DIM,
            distance=Distance.COSINE
        )
    )


# -----------------------------------
# BUILD INDEX FOR PDF
# -----------------------------------
def build_index_for_pdf(doc_id: str, pdf_path: str):
    reader = PdfReader(pdf_path)
    chunks = []

    for p in reader.pages:
        text = p.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(text)

    embeddings = embed(chunks)

    points = []
    for idx, (txt, vec) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=f"{doc_id}_{idx}",
                vector=vec,
                payload={
                    "text": txt,
                    "doc_id": doc_id,
                    "chunk_id": idx
                }
            )
        )

    qdrant.upsert(collection_name="rag_chunks", points=points)

    return {"chunks": len(points)}


# -----------------------------------
# SEARCH
# -----------------------------------
def search_doc(doc_id: str, query: str, k=5):
    qvec = embed(query)[0]

    hits = qdrant.search(
        collection_name="rag_chunks",
        query_vector=qvec,
        limit=k
    )

    out = []
    for h in hits:
        p = h.payload
        out.append({
            "id": h.id,
            "score": float(h.score),
            "text": p.get("text", ""),
            "doc_id": p.get("doc_id"),
            "chunk_id": p.get("chunk_id")
        })

    return out
