# app/self_healing_cp/learner.py

import json
import os
import time
from typing import Dict, Any, List, Optional

# ---------------------------------------
# 🚀 NEW: Qdrant + Embedding dependencies
# ---------------------------------------
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
    Self-Healing Memory Layer
    -------------------------
    - Keeps original JSON storage (verified + failures)
    - Adds Qdrant Vector Memory for semantic recall
    - Stores embeddings of:
        • failures
        • corrections
        • verified Q/A
    """

    def __init__(self, file_path: Optional[str] = None, path: Optional[str] = None):
        # -------------------------------
        # JSON File Setup
        # -------------------------------
        self.path = file_path or path or "app/data/learning_memory.json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"verified": [], "failures": []}, f, indent=2)

        # -------------------------------
        # 🚀 Qdrant Setup
        # -------------------------------
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.qdrant = QdrantClient(host="localhost", port=6333)

        # Create collection if not exists
        self.qdrant.recreate_collection(
            collection_name="healing_memory",
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE
            )
        )

    # ------------------------------------------------------
    # Internal JSON utilities
    # ------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------
    # 🚀 NEW: Add to Qdrant Memory
    # ------------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], convert_to_numpy=True)[0]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.astype("float32")

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

    # ------------------------------------------------------
    # Add Verified Q/A (correct answers)
    # ------------------------------------------------------
    def add_verified_qa(self, query: str, answer: str) -> None:
        data = self._load()
        entry = {"query": query, "answer": answer}

        if entry not in data["verified"]:
            data["verified"].append(entry)

        self._save(data)

        # 🚀 Store in Qdrant
        self._store_qdrant(
            text=query + " " + answer,
            metadata={
                "type": "verified",
                "query": query,
                "answer": answer,
                "timestamp": time.time(),
            }
        )

    # ------------------------------------------------------
    # Add Failure (wrong answer)
    # ------------------------------------------------------
    def add_failure(self, query: str, wrong_answer: str, reason: str = "") -> None:
        data = self._load()
        entry = {
            "query": query,
            "wrong_answer": wrong_answer,
            "reason": reason,
        }
        data["failures"].append(entry)
        self._save(data)

        # 🚀 Store failure in Qdrant vector memory
        self._store_qdrant(
            text=query + " " + wrong_answer + " " + reason,
            metadata={
                "type": "failure",
                "query": query,
                "wrong_answer": wrong_answer,
                "reason": reason,
                "timestamp": time.time(),
            }
        )

    # ------------------------------------------------------
    # Helper: return all verified QA
    # ------------------------------------------------------
    def get_verified(self) -> List[Dict[str, str]]:
        data = self._load()
        return data.get("verified", [])

    # ------------------------------------------------------
    # Helper: return all failures
    # ------------------------------------------------------
    def get_failures(self) -> List[Dict[str, Any]]:
        data = self._load()
        return data.get("failures", [])

    # ------------------------------------------------------
    # 🚀 NEW: Search Healing Memory (semantic search)
    # ------------------------------------------------------
    def search_healing_memory(self, query: str, k: int = 3) -> List[Dict]:
        vector = self._embed(query)

        results = self.qdrant.search(
            collection_name="healing_memory",
            query_vector=vector,
            limit=k
        )

        output = []
        for hit in results:
            payload = hit.payload
            payload["score"] = float(hit.score)
            output.append(payload)

        return output
