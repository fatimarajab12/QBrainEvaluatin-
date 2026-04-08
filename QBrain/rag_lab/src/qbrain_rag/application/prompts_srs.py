"""Feature extraction and test-case prompts (structure-agnostic; works with any ingested text)."""
from __future__ import annotations


def create_adaptive_prompt() -> str:
    """Consolidated extraction; no assumed document template or file format."""
    return """
**FORMAT & STRUCTURE (no assumptions):**
- The CONTEXT below is **plain text from every chunk** of the ingested file, concatenated in **reading order**. Each block starts with a compact marker: `#1`, `#2`, … If several source files were mixed into one context, markers look like `#3 @filename.pdf`. It is the **full indexed document** unless the statistics line says **truncated=yes** — then only an **initial sequence of complete chunks** (or a prefix of chunk 1 if the limit is very tight) is present; extract from that only and do not invent content from the missing tail.
- Text may still be noisy (OCR/scans) or mixed languages. The source may be an SRS, PRD, policy, runbook, meeting notes, slides export, chat log, email thread, ticket dump, spreadsheet pasted as text, code comments, or anything else — **do not require** numbered sections, requirement IDs, tables, or a standard template.
- Infer **testable obligations** (what must hold, what actors do, what the system or process must do) from whatever phrasing appears: bullets, paragraphs, “shall/must/should”, implicit expectations, examples, or checklists.
- **Whole-document coverage:** Scan **all chunks** from start to end. Include obligations from **early, middle, and late** parts of the context — not only the first paragraphs.

**CORE RULES (consolidated features):**
1. Extract testable **obligations and behaviors** relevant to software or product work: functional behavior, workflow, UI, data, quality, reporting, constraints, notifications, integrations — **only when** such content appears in the context. If the text is purely narrative/marketing with no verifiable obligation, extract little or nothing rather than inventing features.
2. **Grouping:**
   - If the document uses **explicit labels** (clause numbers, Req-IDs, ticket keys, heading text that repeats): prefer **one top-level feature per label**, with sub-points inside `description` and `acceptanceCriteria`. Do not split every sub-bullet under the **same** label unless the text clearly states **independent** obligations.
   - If there are **no stable IDs**: group by **distinct obligation or theme** (one feature per clearly separate capability or rule cluster). Do **not** emit one feature per sentence when those sentences describe the **same** mechanism (see rule 6).
3. Use **separate** features only for **genuinely different** obligations (different actors, rules, outcomes, or systems), not for rewordings of the same point.
4. Group many attributes of one **same entity** into one DATA_MODEL feature when the text clearly describes one data object.
5. Skip content that is **only** pure hosting/deployment/infrastructure **unless** it changes something observable to users or operators of the product described.
6. **Anti over-splitting:** Do **not** create many top-level features that differ only by wording or a minor trigger when they implement the **same underlying capability**. Merge into one feature; list variants in `description` / `acceptanceCriteria`. Split only when the text gives **distinct anchors** (labels or clearly separate rules) or **materially different** outcomes.

**GROUNDING:** Every feature must be justified by the DOCUMENT CONTEXT. Do not invent requirements, products, or integrations not supported by the text.

**EXTRACTION BALANCE:** Prefer clearly stated or strongly implied obligations; merge same-mechanism variants per rule 6.

**FEATURE TYPES:** FUNCTIONAL, DATA, DATA_MODEL, INTERFACE, QUALITY, REPORT, CONSTRAINT, NOTIFICATION, WORKFLOW

**NAMING & QUALITY:**
- `name`: short, unique, action- or outcome-oriented; avoid duplicate titles.
- `description`: 1–4 sentences; paraphrase the source; include actors and constraints when the text gives them.
- `acceptanceCriteria`: short checks derived from the text; use `[]` if none are explicit.
- `reasoning`: where this came from (quote fragment, heading phrase, chunk tag, or “inferred from paragraph about …”).
- `matchedSections`: **optional location hints** — phrases copied from the text, heading lines, ticket IDs, or `[]` if the context has no usable anchor (never invent fake section numbers).
- `confidence`: 0.85–1.0 when directly stated; 0.5–0.84 when strongly implied; below 0.5 only for weak inference (prefer omitting weak guesses).

**OUTPUT (JSON only):** Root object must have key `"features"` (array). Each element:
- featureId: string (e.g. "feature_001")
- name, description: strings (no markdown fences inside values)
- featureType: one of the types above
- priority: "High" | "Medium" | "Low"
- status: "pending"
- acceptanceCriteria: string[]
- reasoning: string
- matchedSections: string[]
- confidence: number 0.0–1.0
"""


FEATURE_EXTRACTION_USER_TEMPLATE = """You are an expert at turning unstructured or semi-structured document text into a structured feature list.

IMPORTANT:
- **Do not assume** a particular file type, standard, or outline. The CONTEXT is the **entire** ingested file as chunks in order (or a truncated prefix — see statistics line).
- Use paragraphs, lists, tables rendered as text, examples, and implicit “must / will / should” style meaning **throughout** the whole context.

{adaptive_prompt}

{context_stats}

**DOCUMENT CONTEXT:**
{context}

**OUTPUT RULES:** Return a single JSON object with exactly top-level key "features". No markdown, no prose before or after, no trailing commas. Use UTF-8 text in string values as needed.
"""


TEST_CASE_FEW_SHOT = """
Example test case object (structure only — adapt to the feature and SOURCE CONTEXT):
{{
  "testCaseId": "TC_001",
  "title": "Verify required behavior under valid conditions (cite anchor from context if any)",
  "description": "Primary success path supported by the source text",
  "steps": [
    "Establish preconditions stated in the context",
    "Perform the main action with valid inputs",
    "Observe the outcome",
    "Confirm it matches the described obligation"
  ],
  "expectedResult": "Specific, observable outcome aligned with the source",
  "priority": "high",
  "status": "pending",
  "preconditions": ["Only what the context requires before the scenario"],
  "testData": {{}}
}}
"""


def build_test_case_user_prompt(
    *,
    feature_description: str,
    context: str,
    feature_type: str,
    matched_sections: list[str],
) -> str:
    sec = ""
    if matched_sections:
        sec = f"\n**Optional anchors (from feature, if any):** {', '.join(matched_sections)}\n"
    return f"""You are an expert QA engineer. Generate test cases for the feature below using ONLY the SOURCE CONTEXT below.

**FEATURE TYPE:** {feature_type}
{sec}
**FEW-SHOT STRUCTURE:**
{TEST_CASE_FEW_SHOT}

**RULES:**
- The context may lack formal sections; do not assume a template. Ground every scenario in the text you see.
- Do not invent rules, fields, or integrations the context does not support. If unclear, state the assumption in `description` or omit that edge case.
- Reference short phrases or labels from the context in titles when helpful.
- Aim for **3–8** test cases when the context allows: at least one happy path; add negative, boundary, authorization, and data-validation cases only if implied by the text.
- `steps`: 3–10 short, imperative steps; each step executable by a tester.
- `expectedResult`: observable outcome (not “passes” or “works”).
- `testData`: concrete values only when the context supplies them; otherwise `{{}}` or clearly labeled placeholders.
- Each test case includes: testCaseId, title, description, steps[], expectedResult, priority (high/medium/low), status "pending", preconditions[], testData {{}}

**FEATURE DESCRIPTION:**
{feature_description}

**SOURCE CONTEXT:**
{context}

**OUTPUT:** Single JSON object with top-level key "testCases" (array). No markdown, no prose outside JSON, no trailing commas.
"""
