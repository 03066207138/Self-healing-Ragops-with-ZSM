import json
from app.rag.retriever import VectorRetriever

class CorpusManager:
    def __init__(self, corpus_path="app/data/corpus.json"):
        self.corpus_path = corpus_path

    def add_document(self, text: str):
        with open(self.corpus_path, "r+", encoding="utf-8") as f:
            corpus = json.load(f)
            new_id = len(corpus)
            corpus.append({"id": new_id, "text": text})
            f.seek(0)
            json.dump(corpus, f, indent=2)
        print(f"✅ Added document ID {new_id}")
        VectorRetriever(self.corpus_path).rebuild_index()
