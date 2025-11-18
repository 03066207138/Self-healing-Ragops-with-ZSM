# app/rag/generator.py

"""
GROQ-ONLY Generator (Stable Edition)
-----------------------------------
Fixes:
• Proxy injection crash on Render
• NoneType message.content
• Inconsistent fallback return
• Usage tokens missing
• API instability after multiple calls
"""

import os
import re
from typing import List, Dict, Tuple
from groq import Groq


# ============================================================
# 🔧 FIX: Disable all proxies (Render sometimes injects them)
# ============================================================
def _cleanup_proxies():
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        if key in os.environ:
            os.environ.pop(key, None)


# ============================================================
# 🔧 GROQ Client
# ============================================================
def _make_client():
    _cleanup_proxies()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")

    return Groq(api_key=api_key)


# ============================================================
# 🔧 Model Select
# ============================================================
def _choose_model():
    m = os.getenv("GEN_MODEL", "").strip()
    return m if m else "llama-3.3-70b-versatile"


# ============================================================
# 🔧 Build Messages
# ============================================================
def _build_messages(query: str, passages: List[Dict], allow_citations: bool):

    if allow_citations and passages:
        context = "\n".join([f"[{p['id']}] {p['text']}" for p in passages])

        return [
            {
                "role": "system",
                "content": (
                    "You are a careful assistant. Answer ONLY using passages. "
                    "Cite as [dID]. If missing info, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\nPassages:\n{context}\n\n"
                    "- Use ONLY the passages.\n"
                    "- Cite inline like [d4].\n"
                ),
            },
        ]

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. No citations.",
        },
        {
            "role": "user",
            "content": f"Question: {query}\n- Be concise.\n- No citations.",
        },
    ]


# ============================================================
# 🔧 Remove citations (for non-citation mode)
# ============================================================
def _strip_citations(text: str) -> str:
    return re.sub(r"\[d?\d+\]", "", text).strip()


# ============================================================
# 🔧 Extractive Fallback
# ============================================================
def _fallback(passages):
    if not passages:
        return "Model unavailable and no passages found.", 0, 0

    out = ["Relevant passages:"]
    for p in passages[:3]:
        t = p.get("text", "")[:350]
        out.append(f"- [d{p.get('id')}] {t}...")

    return "\n".join(out), 0, 0


# ============================================================
# 🔥 Main Generate Function
# ============================================================
def generate_answer(
    query: str,
    passages: List[Dict],
    allow_citations: bool = True,
    temperature: float = 0.2,
) -> Tuple[str, int, int]:

    try:
        client = _make_client()
        model = _choose_model()
        messages = _build_messages(query, passages, allow_citations)

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        msg = resp.choices[0].message.content or ""
        if not msg.strip():
            return _fallback(passages)

        usage = getattr(resp, "usage", None)
        tin = usage.prompt_tokens if usage else len(str(messages))
        tout = usage.completion_tokens if usage else len(msg.split())

        if not allow_citations:
            msg = _strip_citations(msg)

        return msg, int(tin), int(tout)

    except Exception as e:
        print("GROQ ERROR:", e)
        return _fallback(passages)
