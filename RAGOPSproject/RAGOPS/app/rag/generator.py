from typing import List, Dict, Tuple
import os
import re

# Prefer GROQ if GROQ_API_KEY is available
USE_GROQ = bool(os.getenv("GROQ_API_KEY", "").strip())

# Import correct clients
if USE_GROQ:
    from groq import Groq
else:
    from openai import OpenAI  # modern OpenAI SDK


from ..settings import settings


# ============================================================
# 🔧 FIXED CLIENT CREATION (NO PROXIES PROBLEM)
# ============================================================
def _make_client():
    """
    Create an LLM client safely.
    FIX:
    - No timeout in constructor for OpenAI (modern SDK)
    - No proxies parameter to avoid Render proxy injections
    """
    if USE_GROQ:
        return Groq(api_key=os.getenv("GROQ_API_KEY"))

    # OpenAI new SDK requires: OpenAI(api_key=...)
    # Timeout is passed inside the request, NOT constructor.
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _choose_model():
    gen = (settings.GEN_MODEL or "").strip()
    if gen:
        return gen

    if USE_GROQ:
        return "llama-3.1-70b-versatile"

    # Modern OpenAI model
    return "gpt-4o-mini"


def _build_messages(query: str, passages: List[Dict], allow_citations: bool):
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
            f"- Cite inline like [dID].\n"
            f"- If nothing relevant is found, say you don't know.\n"
        )
    else:
        sys = (
            "You are a helpful assistant. Provide a concise answer based on general knowledge. "
            "Do NOT add any citations or bracketed IDs."
        )
        user = (
            f"Question: {query}\n\n"
            f"- Be concise.\n"
            f"- No citations.\n"
        )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[(?:d?\d+)\]\s*", " ", text).strip()


def _extractive_fallback(passages: List[Dict], max_items=3):
    if not passages:
        return "LLM unreachable and no passages available."

    out = ["Here are relevant excerpts:"]
    for p in passages[:max_items]:
        t = p.get("text", "")
        snip = t[:400] + "…" if len(t) > 400 else t
        out.append(f"- [d{p.get('id', 0)}] {snip}")
    return "\n".join(out)


# ============================================================
# 💬 MAIN: LLM Answer Generator (Fixed)
# ============================================================
def generate_answer(
    query: str,
    passages: List[Dict],
    allow_citations=True,
    temperature=0.2,
) -> Tuple[str, int, int]:

    client = _make_client()
    model = _choose_model()
    messages = _build_messages(query, passages, allow_citations)

    try:
        # ---------------------------
        # GROQ LLM
        # ---------------------------
        if USE_GROQ:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            msg = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else len(msg.split())

        # ---------------------------
        # OPENAI LLM (Modern SDK)
        # ---------------------------
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=20.0,  # timeout placed HERE, not constructor
            )
            msg = resp.choices[0].message.content or ""
            usage = resp.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else len(msg.split())

    except Exception as e:
        print("LLM ERROR:", e)
        fallback = _extractive_fallback(passages)
        if not allow_citations:
            fallback = _strip_citations(fallback)
        return fallback, 0, 0

    if not allow_citations:
        msg = _strip_citations(msg)

    return msg, int(tokens_in or 0), int(tokens_out or 0)
