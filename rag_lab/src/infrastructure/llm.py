"""LLM adapter: OpenAI chat completions for Q&A and JSON extraction."""
from __future__ import annotations

import json

from openai import OpenAI

from application.json_utils import parse_llm_json
from config.settings import get_settings

_client: OpenAI | None = None

_SYSTEM_RAG = """You are a helpful AI assistant that answers questions based on the provided context (excerpted text from an uploaded or indexed source).

**CRITICAL RULES:**
1. Answer using ONLY the provided context; do not invent facts, features, or policies not supported by it
2. The source may be unstructured (notes, logs, PDF extract, etc.); do not assume it follows a standard outline
3. When the text includes headings, numbers, or IDs, cite them in your answer when relevant
4. If the answer is not in the context, say clearly that it is not stated in the provided text
5. If the context is ambiguous or contradictory, say so briefly and report what the text actually says
6. Be concise and professional; use bullet lists for multi-part answers when helpful
7. Do not assume a particular product, vendor, or domain beyond what the context states
8. For broad questions, summarize from the most relevant parts of the context first

Context:
{context}"""

_USER_RAG = """Question: {question}

Please provide a helpful answer based on the context above."""

_SYSTEM_RAG_EVAL = """You are an evaluation-time QA assistant.

Answer using ONLY the provided context.
Rules:
1. Give a complete factual answer in one to three short sentences. Every claim must be supported by the context.
2. When the context names a system, product, or document title, include that name when the question is about what something is, which version/release, or what the document describes (avoid bare fragments like only a version number unless the question asks for the number alone).
3. No preamble ("According to the context..."), no closing commentary, no bullet lists unless the question clearly asks for a list.
4. Prefer wording that keeps key terms from the source (names, version strings, role names) while staying grammatical.
5. If the answer is not explicitly present, output exactly: "Not stated in the provided context."

Context:
{context}"""

_USER_RAG_EVAL = """Question: {question}

Answer directly with the facts needed; keep every statement grounded in the context above."""


_DEFAULT_QA_MAX_DOCS = 5
_DEFAULT_QA_MAX_CHARS_PER_CHUNK = 2000


def _qa_context_from_docs(
    docs,
    *,
    max_context_docs: int | None,
    max_chars_per_chunk: int | None,
) -> str:
    seq = list(docs)
    if max_context_docs is not None:
        seq = seq[: max(0, int(max_context_docs))]
    parts: list[str] = []
    for d in seq:
        text = getattr(d, "page_content", "") or ""
        if max_chars_per_chunk is not None and len(text) > int(max_chars_per_chunk):
            text = text[: int(max_chars_per_chunk)]
        parts.append(text)
    return "\n\n".join(parts)


def _client_or_raise() -> OpenAI:
    global _client
    s = get_settings()
    if not s.openai_api_key:
        raise ValueError("Set OPENAI_API_KEY in rag_lab/.env")
    if _client is None:
        _client = OpenAI(api_key=s.openai_api_key)
    return _client


def answer_with_context(
    question: str,
    docs,
    *,
    temperature: float | None = None,
    evaluation_mode: bool = False,
    max_context_docs: int | None = _DEFAULT_QA_MAX_DOCS,
    max_chars_per_chunk: int | None = _DEFAULT_QA_MAX_CHARS_PER_CHUNK,
) -> str:
    context = _qa_context_from_docs(
        docs,
        max_context_docs=max_context_docs,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    s = get_settings()
    temp = s.generation_temperature if temperature is None else temperature
    if evaluation_mode:
        system_content = _SYSTEM_RAG_EVAL.format(context=context)
        user_content = _USER_RAG_EVAL.format(question=question)
    else:
        system_content = _SYSTEM_RAG.format(context=context)
        user_content = _USER_RAG.format(question=question)
    response = _client_or_raise().chat.completions.create(
        model=s.chat_model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=temp,
    )
    return (response.choices[0].message.content or "").strip()


def complete_json_object(
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
) -> dict:
    """Chat completion with JSON object mode (for feature / test-case extraction)."""
    s = get_settings()
    client = _client_or_raise()
    response = client.chat.completions.create(
        model=s.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        parsed = parse_llm_json(raw)
        return parsed if isinstance(parsed, dict) else {}
