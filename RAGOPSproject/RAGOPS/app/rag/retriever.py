from __future__ import annotations
import os, json
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# 🚀 QDRANT
# -------------------------------
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

# ---------- Embedding Model ----------
MODEL_NAME = "all-MiniLM-L6-v2"


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _ensure(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class VectorRetriever:
    """
    UPDATED: QDRANT-POWERED RETRIEVER
    ---------------------------------
    - Stores all chunks in Qdrant
    - Metadata-rich memory
    - Cosine search
    - Fully drop-in compatible replacement for FAISS version
    """

    def __init__(
        self,
        corpus_path="app/data/corpus.json",
        collection_name="rag_vectors"
    ):
        self.corpus_path = corpus_path
        self.collection = collection_name
        _ensure(self.corpus_path)

        # Embedding model
        self.model = SentenceTransformer(MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Initialize Qdrant
        self.qdrant = QdrantClient(url="http://localhost:6333", timeout=1)


        # Create collection if missing
        self.qdrant.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE
            )
        )

        # load metadata from JSON
        self.texts = []
        self.meta = []
        self._load_corpus()

    # ----------------------------------------
    # EMBEDDING
    # ----------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embs = self.model.encode(texts, convert_to_numpy=True)
        embs = _normalize(embs.astype("float32"))
        return embs

    # ----------------------------------------
    # LOAD CORPUS.JSON (metadata only)
    # ----------------------------------------
    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            with open(self.corpus_path, "w") as f:
                json.dump([], f)

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        self.texts = [c["text"] for c in corpus]
        self.meta = [
            {
                "doc_id": c["doc_id"],
                "page": c.get("page", "?"),
                "title": c.get("title", c["doc_id"]),
            }
            for c in corpus
        ]

    # ----------------------------------------
    # ADD TO QDRANT INDEX
    # ----------------------------------------
    def add_to_index(self, chunks: List[str], doc_id: str):
        if not chunks:
            return

        embs = self._embed(chunks)

        points = []
        for i, (text, vec) in enumerate(zip(chunks, embs)):
            point_id = f"{doc_id}_{len(self.texts) + i}"

            # store metadata
            meta_item = {
                "text": text,
                "doc_id": doc_id,
                "page": "?",
                "title": doc_id,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=meta_item
                )
            )

            # append to local metadata list
            self.texts.append(text)
            self.meta.append(meta_item)

        # Upsert into Qdrant
        self.qdrant.upsert(collection_name=self.collection, points=points)

        # Save metadata to corpus.json
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"text": t, **m} for t, m in zip(self.texts, self.meta)],
                f,
                indent=2
            )

    # ----------------------------------------
    # SEARCH
    # ----------------------------------------
    def search(self, query: str, k=5, restrict_to=None) -> List[Dict]:
        if len(self.texts) == 0:
            return []

        q_vec = self._embed(query)[0]

        # Optional doc filter
        q_filter = None
        if restrict_to:
            q_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=restrict_to)
                    )
                ]
            )

        # Qdrant search
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=q_vec,
            limit=k,
            query_filter=q_filter
        )

        results = []
        for h in hits:
            payload = h.payload
            text = payload["text"]

            snippet = text[:300] + "…" if len(text) > 300 else text

            results.append({
                "id": h.id,
                "text": text,
                "snippet": snippet,
                "score": float(h.score),
                "doc_id": payload.get("doc_id"),
                "page": payload.get("page"),
                "title": payload.get("title"),
            })

        return results
