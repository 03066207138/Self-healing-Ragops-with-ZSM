# app/self_healing_cp/learner.py

import json
import os
import time
from typing import Dict, Any, List, Optional

# Qdrant Embedded
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from sentence_transformers import SentenceTransformer
import numpy as np


class LearningMemory:
    """
    Self-Healing Memory for RAGOPS
    Uses:
        • JSON store (verified + failures)
        • Qdrant Embedded Vector Memory
    """

    def __init__(self, file_path: Optional[str] = None, path: Optional[str] = None):
        # ---------------------------------------
        # JSON Memory File
        # ---------------------------------------
        self.path = file_path or path or "app/data/learning_memory.json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"verified": [], "failures": []}, f, indent=2)

        # ---------------------------------------
        # Embedding Model
        # ---------------------------------------
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # ---------------------------------------
        # 🚀 Embedded Qdrant Storage
        # ---------------------------------------
        # Stored locally inside container
        self.qdrant = QdrantClient(path="qdrant_learning")

        # Create collection safely
        try:
            self.qdrant.get_collection("healing_memory")
        except:
            self.qdrant.create_collection(
                collection_name="healing_memory",
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

    # ---------------------------------------
    # JSON Helpers
    # ---------------------------------------
    def _load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ---------------------------------------
    # Embedding Helper
    # ---------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], convert_to_numpy=True)[0]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.astype("float32")

    # ---------------------------------------
    # Store vector in Qdrant
    # ---------------------------------------
    def _store_qdrant(self, text: str, metadata: dict):
        vector = self._embed(text)

        point = PointStruct(
            id=str(time.time_ns()),
            vector=vector,
            payload=metadata
        )

        self.qdrant.upsert(
            collection_name="healing_memory",
            points=[point]
        )

    # ---------------------------------------
    # Add Verified QA
    # ---------------------------------------
    def add_verified_qa(self, query: str, answer: str):
        data = self._load()
        entry = {"query": query, "answer": answer}

        if entry not in data["verified"]:
            data["verified"].append(entry)
            self._save(data)

        # Store vector memory
        self._store_qdrant(
            text=f"{query} {answer}",
            metadata={
                "type": "verified",
                "query": query,
                "answer": answer,
                "timestamp": time.time(),
            }
        )

    # ---------------------------------------
    # Add Failure
    # ---------------------------------------
    def add_failure(self, query: str, wrong_answer: str, reason: str = ""):
        data = self._load()
        entry = {
            "query": query,
            "wrong_answer": wrong_answer,
            "reason": reason,
        }

        data["failures"].append(entry)
        self._save(data)

        # Store vector
        self._store_qdrant(
            text=f"{query} {wrong_answer} {reason}",
            metadata={
                "type": "failure",
                "query": query,
                "wrong_answer": wrong_answer,
                "reason": reason,
                "timestamp": time.time(),
            }
        )

    # ---------------------------------------
    # Search Healing Memory
    # ---------------------------------------
    def search_healing_memory(self, query: str, k: int = 3):
        vector = self._embed(query)

        hits = self.qdrant.search(
            collection_name="healing_memory",
            query_vector=vector,
            limit=k
        )

        results = []
        for h in hits:
            payload = h.payload
            payload["score"] = float(h.score)
            results.append(payload)

        return results
