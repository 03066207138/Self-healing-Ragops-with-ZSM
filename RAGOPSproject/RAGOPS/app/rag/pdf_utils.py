# app/rag/pdf_utils.py
import re
from typing import List
from pypdf import PdfReader

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    raw = "\n".join(texts)
    # normalize whitespace
    raw = raw.replace("\r", " ")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{2,}", "\n", raw)
    return raw.strip()

def chunk_text(text: str, max_tokens: int = 700, overlap: int = 120) -> List[str]:
    """
    Simple word-based chunker (approx tokens). You can later swap for tiktoken.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while i < len(words):
        chunk = " ".join(words[i:i+max_tokens])
        chunks.append(chunk)
        i += step
    return chunks
