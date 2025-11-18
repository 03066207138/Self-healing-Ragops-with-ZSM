from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import os

class QdrantStore:
    """
    Embedded Qdrant Store (No external server required)
    Works on Render, Railway, HuggingFace, Local PC.
    """

    def __init__(self, embed_dim=384, collection="rag_vectors"):
        self.collection = collection

        # ------------------------------
        # ⭐ Embedded Qdrant
        # ------------------------------
        # Stores vector DB in local folder "qdrant_storage"
        self.client = QdrantClient(path="qdrant_storage")

        # ------------------------------
        # ⭐ Create collection if missing
        # ------------------------------
        try:
            self.client.get_collection(self.collection)
        except:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=embed_dim,
                    distance=Distance.COSINE
                )
            )

    # ----------------------------------------
    # ADD VECTOR + PAYLOAD
    # ----------------------------------------
    def add(self, vector, text, metadata=None):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "metadata": metadata or {}
            }
        )

        self.client.upsert(
            collection_name=self.collection,
            points=[point]
        )

    # ----------------------------------------
    # SEARCH SIMILAR VECTORS
    # ----------------------------------------
    def search(self, vector, top_k=5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k
        )
        return results
