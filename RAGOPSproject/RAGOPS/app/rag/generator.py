# app/rag/generator.py

"""
GROQ-ONLY Generator
-------------------
Safe for Render (no proxies argument, no OpenAI SDK).
Handles:
• Chat Completion (LLM)
• Strict citation mode for RAG
• Non-citation mode for general answers
• Fallback when API fails
"""

from typing import List, Dict, Tuple
import os
import re
from groq import Groq


# ============================================================
# 🔧 Client Creation (GROQ ONLY)
# ============================================================
def _make_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment variables")

    return Groq(api_key=api_key)


# ============================================================
# 🔧 Model Selector
# ============================================================
def _choose_model():
    # Use environment model if available
    model = os.getenv("GEN_MODEL", "").strip()
    if model:
        return model

    # Default recommended Groq model
    return "llama-3.3-70b-versatile"


# ============================================================
# 🔧 RAG Message Builder
# ============================================================
def _build_messages(query: str, passages: List[Dict], allow_citations: bool):
    """
    Builds chat messages depending on whether citations are allowed.
    """

    if allow_citations and passages:
        context = "\n".join([f"[{p['id']}] {p['text']}" for p in passages])

        sys = (
            "You are a careful assistant. Answer using ONLY the provided passages. "
            "Cite passage IDs inline like [d12]. If uncertain, say you don't know."
        )

        user = (
            f"Question: {query}\n\n"
            f"Passages:\n{context}\n\n"
            "Guidelines:\n"
            "- Use ONLY information from passages.\n"
            "- Cite passage IDs like [dID].\n"
            "- If info is missing, say you don't know.\n"
        )

    else:
        # General non-citation mode
        sys = (
            "You are a helpful assistant. Provide a concise answer using general knowledge. "
            "Do NOT use citations."
        )
        user = (
            f"Question: {query}\n\n"
            "- Be concise.\n"
            "- No citations.\n"
        )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]


# ============================================================
# 🔧 Clean Output (non-citation mode)
# ============================================================
def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[(?:d?\d+)\]\s*", " ", text).strip()


# ============================================================
# 🔧 Extractive Fallback (when API fails)
# ============================================================
def _extractive_fallback(passages: List[Dict], max_items: int = 3) -> str:
    if not passages:
        return "LLM unreachable and no passages available."

    out = ["Here are relevant excerpts:"]
    for p in passages[:max_items]:
        txt = p.get("text", "").strip()
        snippet = txt[:350] + "…" if len(txt) > 350 else txt
        out.append(f"- [d{p.get('id', 0)}] {snippet}")
    return "\n".join(out)


# ============================================================
# 🔥 MAIN GENERATE FUNCTION (GROQ ONLY)
# ============================================================
def generate_answer(
    query: str,
    passages: List[Dict],
    allow_citations: bool = True,
    temperature: float = 0.2,
) -> Tuple[str, int, int]:

    client = _make_client()
    model = _choose_model()
    messages = _build_messages(query, passages, allow_citations)

    try:
        # GROQ LLM COMPLETION
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        msg = resp.choices[0].message.content or ""

        usage = getattr(resp, "usage", None)
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else len(msg.split())

    except Exception as e:
        print("GROQ ERROR:", e)
        fallback = _extractive_fallback(passages)
        if not allow_citations:
            fallback = _strip_citations(fallback)
        return fallback, 0, 0

    if not allow_citations:
        msg = _strip_citations(msg)

    return msg, int(tokens_in), int(tokens_out)
