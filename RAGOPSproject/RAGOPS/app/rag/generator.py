from typing import List, Dict, Tuple
import os
import re

# Prefer GROQ if GROQ_API_KEY is available
USE_GROQ = bool(os.getenv("GROQ_API_KEY", "").strip())

if USE_GROQ:
    from groq import Groq
else:
    from openai import OpenAI

from ..settings import settings


def _make_client():
    """
    Create an LLM client with a short timeout.
    """
    if USE_GROQ:
        # Groq client doesn’t expose timeout on init; keep simple
        return Groq(api_key=os.getenv("GROQ_API_KEY"))
    # OpenAI supports a timeout kwarg on the client
    return OpenAI(api_key=settings.OPENAI_API_KEY, timeout=20.0)


def _choose_model():
    """
    Pick a sane default per provider if GEN_MODEL isn't set correctly.
    """
    gen = (settings.GEN_MODEL or "").strip()
    if gen:
        return gen
    if USE_GROQ:
        # A current Groq chat model (avoid deprecated ones)
        return "llama-3.1-70b-versatile"
    # OpenAI fallback (adjust to your account availability)
    return "gpt-4o-mini"


def _build_messages(query: str, passages: List[Dict], allow_citations: bool) -> list:
    """
    Strict RAG prompt when we have passages and citations allowed.
    Clean general-answer prompt when we don't.
    """
    if allow_citations and passages:
        context = "\n".join([f"[{p['id']}] {p['text']}" for p in passages])
        sys = (
            "You are a careful assistant. Answer using only the provided passages. "
            "Cite passage IDs inline like [d12]. If uncertain, say you don't know."
        )
        user = (
            f"Question: {query}\n\n"
            f"Passages:\n{context}\n\n"
            f"Guidelines:\n"
            f"- Only use information from the passages above.\n"
            f"- Cite inline like [dID] (e.g., [d5]).\n"
            f"- If nothing relevant is found, say you don't know.\n"
        )
    else:
        sys = (
            "You are a helpful assistant. Provide a concise answer based on general knowledge. "
            "Do NOT add any citations or bracketed IDs."
        )
        user = (
            f"Question: {query}\n\n"
            f"Guidelines:\n"
            f"- Be concise and clear.\n"
            f"- Do NOT include citations.\n"
        )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _strip_citations(text: str) -> str:
    # Remove [d10], [0], [12] patterns if they sneak into non-citation mode
    return re.sub(r"\s*\[(?:d?\d+)\]\s*", " ", text).strip()


def _extractive_fallback(passages: List[Dict], max_items: int = 3) -> str:
    """
    If the LLM call fails (network/API), return extractive snippets with citations.
    """
    if not passages:
        return "Sorry — I couldn’t reach the language model and there were no passages retrieved to quote."
    out_lines = ["Here are relevant excerpts:"]
    for p in passages[:max_items]:
        text = p.get("text", "").strip()
        # keep it short-ish
        snippet = (text[:400] + "…") if len(text) > 400 else text
        out_lines.append(f"- [d{p.get('id', 0)}] {snippet}")
    return "\n".join(out_lines)


def generate_answer(
    query: str,
    passages: List[Dict],
    allow_citations: bool = True,
    temperature: float = 0.2,
) -> Tuple[str, int, int]:
    """
    You are an expert AI research assistant and summarizer.
Your goal is to synthesize clear, well-structured, and academic-quality answers using only the retrieved excerpts.
Each response must sound like part of a research paper — analytical, coherent, and properly referenced.

Follow these rules:

1. **Read all provided excerpts carefully.**
   - Identify the main concepts, methods, results, and implications.
   - Ignore irrelevant or duplicate sentences.

2. **Synthesize** — don’t just list or copy.
   - Combine ideas logically to form complete explanations.
   - Use transitions (e.g., “Furthermore,” “In addition,” “This approach enables…”).

3. **Structure** your answer like this:
   - **Definition / Concept:** Briefly introduce what the paper or topic is about.
   - **Working / Mechanism:** Explain how it works or what methods are used.
   - **Applications / Outcomes:** Mention experiments, results, or use cases.
   - **Limitations / Future Work:** Optionally summarize open challenges or directions.

4. **Cite** evidence inline as `[d1]`, `[d2]`, etc.
   - Combine multiple sources (e.g., “[d1][d3]”) when they support the same point.

5. **Tone:** Formal, precise, and concise — like a scientific review.
   - Avoid generic phrases (“this is important”) and make every sentence informative.

If the question is conceptual (e.g., “What is RAGOps?”), provide a clear summary.
If it is analytical (e.g., “How does ZSM work?” or “What are TRiSM challenges?”), provide a technical explanation.
If it is comparative (e.g., “Compare SOA and Microservices”), use structured contrast points.

Your goal is to give the user an accurate, academic, and human-readable explanation of the topic, 
as if it were written in a research paper.
"""
    client = _make_client()
    model = _choose_model()
    messages = _build_messages(query, passages, allow_citations)

    try:
        if USE_GROQ:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            msg = resp.choices[0].message.content or ""
            tokens_in = getattr(resp, "usage", None).prompt_tokens if getattr(resp, "usage", None) else 0
            tokens_out = getattr(resp, "usage", None).completion_tokens if getattr(resp, "usage", None) else len(msg.split())
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            msg = resp.choices[0].message.content or ""
            tokens_in = resp.usage.prompt_tokens if resp.usage else 0
            tokens_out = resp.usage.completion_tokens if resp.usage else len(msg.split())
    except Exception:
        # Connection/APIs down → extractive fallback
        fallback = _extractive_fallback(passages)
        if not allow_citations:
            fallback = _strip_citations(fallback)
        return fallback, 0, 0

    if not allow_citations:
        msg = _strip_citations(msg)
    return msg, int(tokens_in or 0), int(tokens_out or 0)
