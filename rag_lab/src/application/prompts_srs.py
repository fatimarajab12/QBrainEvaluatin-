"""Feature extraction and test-case prompts (structure-agnostic; works with any ingested text)."""
from __future__ import annotations


def create_adaptive_prompt() -> str:
    """Shorter rules: essential vs preferred vs schema — reduces instruction competition."""
    return """
**Context:** Chunks appear in reading order (`#1`, `#2`, …). If the statistics line says truncated, only that prefix exists — do not invent the rest.

**Essential**
- Extract only **testable obligations** grounded in the text (behaviour, data, workflows, constraints). Skip pure marketing with no checkable claim.
- **Deduplicate aggressively**: keep one feature for the same obligation even when phrased differently.
- Merge candidates when they share the same actor + action + object + condition/outcome.
- Split only for clearly separate obligations (different actors, outcomes, or stable labels with distinct behavior).
- **Do not invent** requirements, products, or integrations absent from the context.
- Use realistic confidence (0.5–1.0), not a constant score.

**Preferred**
- One feature per explicit label (req ID, ticket key, repeated heading) when the text supports it; otherwise group by theme.
- One DATA_MODEL feature when many fields describe one entity.
- Ignore hosting-only detail unless it changes something observable to users/operators.
- Normalize names/descriptions into one canonical wording per obligation.

**Feature types:** FUNCTIONAL, DATA, DATA_MODEL, INTERFACE, QUALITY, REPORT, CONSTRAINT, NOTIFICATION, WORKFLOW

**Schema:** `features` array. Each element: featureId, name, description, featureType, priority ("High"|"Medium"|"Low"), status "pending", acceptanceCriteria[] (short checks; [] if none), reasoning (source anchor), matchedSections[] (text phrases; [] if none), confidence 0.0–1.0 (omit weak guesses).
"""


FEATURE_EXTRACTION_USER_TEMPLATE = """Turn the document context into a JSON feature list. Do not assume a fixed SRS template; use whatever structure appears in the text.

{adaptive_prompt}

{context_stats}

**DOCUMENT CONTEXT:**
{context}

**Output:** One JSON object, top-level key `"features"` only. No markdown, no trailing commas.
"""


FEATURE_PARTIAL_USER_TEMPLATE = """**Segment only:** The context is one slice of the file. Extract features for obligations visible **here** only.

{adaptive_prompt}

**Segment statistics:** {context_stats}

**DOCUMENT CONTEXT (this segment):**
{context}

**Output:** JSON with top-level `"features"` only; same schema as full extraction.
"""


FEATURE_CONSOLIDATION_USER_TEMPLATE = """Merge segment feature lists into one deduplicated list.

**Input:** JSON array of objects with segment_index, chunk_id_range, features.

**Essential:** Merge same obligation across segments; renumber featureId (feature_001, …). Do not add requirements absent from the candidates.
- Remove semantic duplicates and near-duplicates (same meaning with different wording).
- Keep one canonical feature per obligation and union their evidence in `matchedSections`.

**Preferred:** Combine descriptions/criteria when merging; keep distinct capabilities separate.

**PARTIAL EXTRACTIONS (JSON):**
{candidates_json}

**Output:** JSON object, key `"features"` only.
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
        sec = f"\n**Anchors (from feature):** {', '.join(matched_sections)}\n"
    return f"""**Essential:** Use ONLY the SOURCE CONTEXT. Ground every case in the text; do not invent integrations or rules.
- If a test case cannot be **directly supported** by the SOURCE CONTEXT, **do not** generate it (no filler or generic scenarios).

**Preferred:** 3–8 cases when the text supports it; add negative/boundary/auth cases only if implied. Short context phrases in titles help traceability.

**Shape example:**
{TEST_CASE_FEW_SHOT}

**Fields:** testCaseId, title, description, steps (≥3 imperative steps), expectedResult (observable, not "passes"), priority high|medium|low, status pending, preconditions[], testData {{}}

**FEATURE TYPE:** {feature_type}
{sec}
**FEATURE DESCRIPTION:**
{feature_description}

**SOURCE CONTEXT:**
{context}

**Output:** JSON with top-level `"testCases"` only. No markdown.
"""

