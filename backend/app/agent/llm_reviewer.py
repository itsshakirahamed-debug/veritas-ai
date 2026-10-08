"""LLM-backed case review: fact extraction with source quotes + gap report.

Hard rules enforced downstream: every fact must carry a chunk_id and verbatim
quote; unsupported content is reported as a gap, never invented.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

import anthropic

from ..schemas import Fact, Contradiction, ContradictionValue, GapReport

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    """Lazily build the Anthropic client so import works without an API key."""
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def extract_json(content: str) -> Dict[str, Any]:
    """Parse a JSON object from LLM output, tolerating markdown code fences."""
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _build_contradictions(raw_items: Any) -> List[Contradiction]:
    """Map LLM contradiction output into the Contradiction schema.

    The prompt allows two shapes per contradiction:
      - {"field": ..., "values": [{value, chunk_id, quote}, ...]}
      - {"field": ..., "source1": {value, chunk_id, quote}, "source2": {...}}
    The schema requires ``values`` with at least 2 entries, so source1/source2
    are folded into that shape here (previously this raised a ValidationError).
    """
    contradictions: List[Contradiction] = []
    if not isinstance(raw_items, list):
        return contradictions

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        field = item.get("field", "")
        raw_values: List[Any] = []

        if isinstance(item.get("values"), list):
            raw_values = item["values"]
        else:
            for key in ("source1", "source2", "value1", "value2"):
                if isinstance(item.get(key), dict):
                    raw_values.append(item[key])

        values: List[ContradictionValue] = []
        for v in raw_values:
            if not isinstance(v, dict):
                continue
            values.append(ContradictionValue(
                value=str(v.get("value", "")),
                chunk_id=v.get("chunk_id") or None,
                quote=v.get("quote") or None,
            ))

        if field and len(values) >= 2:
            contradictions.append(Contradiction(field=field, values=values))

    return contradictions


def generate_review(doc_ids: List[str], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is missing")

    context = ""
    for chk in chunks:
        context += f"Chunk ID: {chk['chunk_id']} | Source Document: {chk['doc_id']}\n"
        context += f"Text: {chk['text']}\n\n"

    system_prompt = (
        "You are an Agentic Legal Assistant performing a case review.\n"
        "Your task is to identify key facts and flag contradictions or missing information across the documents.\n"
        "Hard rules:\n"
        "1. No fact may appear unless traced to a chunk, with a verbatim supporting quote.\n"
        "2. Output must be valid JSON matching this schema exactly:\n"
        "{\n"
        '  "extracted_facts": [\n'
        '    {\n'
        '      "field": "fact description",\n'
        '      "value": "extracted value",\n'
        '      "chunk_id": "the chunk_id",\n'
        '      "quote": "verbatim quote"\n'
        '    }\n'
        '  ],\n'
        '  "gap_report": {\n'
        '    "missing_fields": ["list of important missing facts"],\n'
        '    "contradictions": [\n'
        '      {\n'
        '        "field": "field name with contradiction",\n'
        '        "source1": {"chunk_id": "...", "value": "...", "quote": "..."},\n'
        '        "source2": {"chunk_id": "...", "value": "...", "quote": "..."}\n'
        '      }\n'
        '    ],\n'
        '    "confidence_per_field": {"fact description": 0.99}\n'
        '  }\n'
        "}"
    )

    prompt = (
        f"Review the provided context and extract key facts.\n"
        f"Context:\n{context}\n\n"
        "Respond ONLY with valid JSON."
    )

    response = _get_client().messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )

    data = extract_json(response.content[0].text)

    facts = []
    for f in data.get("extracted_facts", []):
        facts.append(Fact(
            field=f.get("field", ""),
            value=f.get("value", ""),
            chunk_id=f.get("chunk_id") or None,
            quote=f.get("quote") or None
        ))

    gap_data = data.get("gap_report", {})
    gap_report = GapReport(
        missing_fields=gap_data.get("missing_fields", []),
        contradictions=_build_contradictions(gap_data.get("contradictions", [])),
        confidence_per_field=gap_data.get("confidence_per_field", {})
    )

    return {
        "doc_ids": doc_ids,
        "extracted_facts": facts,
        "gap_report": gap_report
    }
