"""LLM adapter: OpenAI chat completions for Q&A and JSON extraction."""
from __future__ import annotations

import json

from openai import OpenAI

from qbrain_rag.application.json_utils import parse_llm_json
from qbrain_rag.config.settings import get_settings

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
1. Output a short factual answer (one sentence, or two at most).
2. Do not add introductions, explanations, or assumptions.
3. Prefer wording that stays close to the source text.
4. If the answer is not explicitly present, output exactly: "Not stated in the provided context."

Context:
{context}"""

_USER_RAG_EVAL = """Question: {question}

Return only the direct answer."""


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
) -> str:
    context = "\n\n".join([d.page_content for d in docs])
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
    return response.choices[0].message.content or ""


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
