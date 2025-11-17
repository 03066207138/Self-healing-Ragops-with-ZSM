# app/rag/pdf_router.py

import os
import time
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.rag.pipeline import pipeline
from app.rag.retriever import VectorRetriever

router = APIRouter()

PDF_DIR = "app/data/pdfs"
os.makedirs(PDF_DIR, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF → extract → clean → chunk → embed → index in Qdrant
    Uses RAGPipeline.index_pdf() NOT the broken pdf_store
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed.")

    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._-")
    if not safe_name:
        safe_name = f"doc_{int(time.time())}.pdf"

    path = os.path.join(PDF_DIR, safe_name)
    os.makedirs(PDF_DIR, exist_ok=True)

    # save the PDF
    with open(path, "wb") as f:
        f.write(await file.read())

    # IMPORTANT — USE PIPELINE FOR INDEXING
    from app.rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()

    doc_id = safe_name.replace(".pdf", "")
    chunks = pipeline.index_pdf(path, doc_id)

    return {
        "status": "ok",
        "doc_id": doc_id,
        "filename": safe_name,
        "chunks_indexed": len(chunks)
    }

