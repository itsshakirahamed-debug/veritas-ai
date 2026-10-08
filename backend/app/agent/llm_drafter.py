"""LLM-backed legal drafting: grounded claims with source quotes + gap report.

Hard rules: every claim must cite a chunk with a verbatim quote; anything
unsupported becomes the literal text [INFORMATION NOT IN RECORD].
"""

import os
from typing import List, Dict, Any, Optional

import anthropic

from ..schemas import Claim, ClaimStatus, GapReport, SectionDraft, DraftResult
from .llm_reviewer import DEFAULT_MODEL, _get_client, extract_json, _build_contradictions


def generate_draft(doc_type: str, required_fields: List[str], chunks: List[Dict[str, Any]]) -> DraftResult:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is missing")

    # Combine chunk text as context
    context = ""
    for chk in chunks:
        context += f"Chunk ID: {chk['chunk_id']} | Source Document: {chk['doc_id']}\n"
        context += f"Text: {chk['text']}\n\n"

    system_prompt = (
        "You are an Agentic Legal Assistant. Your task is to draft a legal document strictly based on the provided context.\n"
        "Hard rules:\n"
        "1. No claim or fact may appear unless traced to a chunk, with a verbatim supporting quote.\n"
        "2. Unsupported content becomes the literal text [INFORMATION NOT IN RECORD].\n"
        "3. Output must be perfectly formatted JSON adhering to the DraftResult schema structure.\n\n"
        "DraftResult schema:\n"
        "{\n"
        '  "sections": [\n'
        '    {\n'
        '      "heading": "Section Heading",\n'
        '      "claims": [\n'
        '        {\n'
        '          "text": "The claim text",\n'
        '          "chunk_id": "the chunk_id, or null if unsupported",\n'
        '          "quote": "verbatim quote from chunk, or null if unsupported",\n'
        '          "status": "verified" or "not_in_record"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ],\n'
        '  "gap_report": {\n'
        '    "missing_fields": ["list of fields that were required but missing"],\n'
        '    "contradictions": [],\n'
        '    "confidence_per_field": {"field_name": 0.99}\n'
        '  }\n'
        "}"
    )

    prompt = (
        f"Draft a {doc_type}.\n"
        f"Required Fields to look for: {', '.join(required_fields or [])}\n\n"
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

    sections = []
    for sec in data.get("sections", []):
        claims = []
        for c in sec.get("claims", []):
            claims.append(Claim(
                text=c.get("text", ""),
                chunk_id=c.get("chunk_id"),
                quote=c.get("quote"),
                status=ClaimStatus.VERIFIED if c.get("chunk_id") else ClaimStatus.NOT_IN_RECORD
            ))
        sections.append(SectionDraft(heading=sec.get("heading", ""), claims=claims))

    gap_data = data.get("gap_report", {})
    gap_report = GapReport(
        missing_fields=gap_data.get("missing_fields", []),
        contradictions=_build_contradictions(gap_data.get("contradictions", [])),
        confidence_per_field=gap_data.get("confidence_per_field", {})
    )

    return DraftResult(sections=sections, gap_report=gap_report)
