Self-Healing RAGOps with ZSM (Zero-Touch Management Inspired RAG Operating System)
Autonomous · Reliable · Adaptive · Real-Time Self-Monitoring RAG System

This project implements a next-generation Self-Healing Retrieval-Augmented Generation Operating System (RAGOps) inspired by Zero-Touch Network & Service Management (ZSM) principles.
It introduces a multi-agent Control Plane on top of the traditional RAG Data Plane, enabling the system to:

detect failures
diagnose root causes
self-heal automatically
validate improvements
learn from user feedback
ensure production-grade reliability

This repo contains a complete backend system (FastAPI, Groq LLM, Qdrant, MiniLM embeddings, anomaly detection, telemetry, and healing agents).

<br>
🌟 Core Features
🔹 1. Multi-Layered RAG Architecture

PDF ingestion

Text extraction

Intelligent chunking

MiniLM-L6-v2 embeddings

Qdrant vector store

Groq LLM answer generation

🔹 2. Autonomous Control Plane

Inspired by 6G ZSM principles:

Real-time telemetry

Semantic drift detection

Hallucination scoring

Coverage@K computation

Latency & cost monitoring

Rolling anomaly detection

🔹 3. Self-Healing Agent

Automatically performs actions such as:

Tighten prompt

Increase retrieval depth

Reduce context window

Switch to extractive fallback mode

Re-compute embeddings

Retry with recovery settings

🔹 4. Verified QA Memory

User feedback stored for learning

Reinforces correct answers

Tracks incorrect generations

Adaptive learning loop



Includes:

/query → RAG pipeline

/index_pdf → PDF ingestion

/feedback → RL loop

/metrics/* → Governance analytics

/upload → File management

