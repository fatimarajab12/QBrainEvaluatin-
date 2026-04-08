"""Parse JSON from LLM output (aligned with `backend/ai/reasoning/jsonUtils.js` ideas)."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_block(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if m and m.group(1):
        cleaned = m.group(1).strip()
    else:
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"```\s*$", "", cleaned)

    start = re.search(r"[\[\{]", cleaned)
    if not start:
        return cleaned
    i = start.start()
    start_char = cleaned[i]
    end_char = "]" if start_char == "[" else "}"
    depth = 0
    in_str = False
    escape = False
    for j in range(i, len(cleaned)):
        c = cleaned[j]
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if c == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == start_char:
            depth += 1
        elif c == end_char:
            depth -= 1
            if depth == 0:
                return cleaned[i : j + 1]
    return cleaned[i:]


def parse_llm_json(text: str) -> Any | None:
    raw = extract_json_block(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # tolerate trailing commas in simple cases
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None
