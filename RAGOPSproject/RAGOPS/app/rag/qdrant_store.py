from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

class QdrantStore:
    def __init__(self, host="localhost", port=6333, embed_dim=384):
        self.client = QdrantClient(host=host, port=port)

        self.client.recreate_collection(
            collection_name="rag_vectors",
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE)
        )

    def add(self, vector, text, metadata=None):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "metadata": metadata or {}
            }
        )
        self.client.upsert(collection_name="rag_vectors", points=[point])

    def search(self, vector, top_k=5):
        return self.client.search(
            collection_name="rag_vectors",
            query_vector=vector,
            limit=top_k
        )
