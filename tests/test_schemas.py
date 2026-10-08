"""
Unit tests for Veritas Legal schemas (Pydantic v2).
"""

import pytest
import sys
from pathlib import Path

# Ensure backend package is in import path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.schemas import (
    DocType,
    ClaimStatus,
    Chunk,
    Fact,
    Claim,
    ContradictionValue,
    Contradiction,
    GapReport,
    SectionDraft,
    DraftResult,
)


def test_chunk_creation():
    chunk = Chunk(
        chunk_id="chk_101",
        doc_id="doc_statute_01",
        doc_type=DocType.STATUTE,
        page=1,
        para_id=2,
        char_start=120,
        char_end=450,
        text="Section 437 in The Code Of Criminal Procedure, 1973..."
    )
    assert chunk.chunk_id == "chk_101"
    assert chunk.doc_type == DocType.STATUTE
    assert chunk.page == 1


def test_fact_creation():
    fact = Fact(
        field="accused_name",
        value="John Doe",
        chunk_id="chk_101",
        quote="John Doe was apprehended at the site"
    )
    assert fact.field == "accused_name"
    assert fact.value == "John Doe"
    assert fact.chunk_id == "chk_101"


def test_claim_defaults_and_status():
    claim_unverified = Claim(text="The accused has no prior criminal records.")
    assert claim_unverified.status == ClaimStatus.NOT_IN_RECORD
    assert claim_unverified.chunk_id is None
    assert claim_unverified.quote is None

    claim_verified = Claim(
        text="The incident took place on 12th Jan 2024.",
        chunk_id="chk_202",
        quote="incident took place on 12th Jan 2024",
        status=ClaimStatus.VERIFIED
    )
    assert claim_verified.status == ClaimStatus.VERIFIED


def test_gap_report_and_contradiction():
    contradiction = Contradiction(
        field="date_of_arrest",
        values=[
            ContradictionValue(value="2024-01-12", chunk_id="chk_1", quote="arrested on 12 Jan 2024"),
            ContradictionValue(value="2024-01-14", chunk_id="chk_2", quote="taken into custody on 14 Jan 2024")
        ]
    )
    gap_report = GapReport(
        missing_fields=["FIR_number", "bailable_offense_status"],
        contradictions=[contradiction],
        confidence_per_field={"date_of_arrest": 0.5, "FIR_number": 0.0}
    )
    assert len(gap_report.missing_fields) == 2
    assert len(gap_report.contradictions) == 1
    assert gap_report.contradictions[0].field == "date_of_arrest"


def test_draft_result_json_serialization():
    claim = Claim(
        text="The applicant is entitled to bail under Section 437 CrPC.",
        chunk_id="chk_301",
        quote="Section 437 CrPC provides discretion for bail",
        status=ClaimStatus.VERIFIED
    )
    section = SectionDraft(heading="Grounds for Bail", claims=[claim])
    draft = DraftResult(sections=[section])
    
    json_data = draft.model_dump_json()
    reconstructed = DraftResult.model_validate_json(json_data)
    
    assert len(reconstructed.sections) == 1
    assert reconstructed.sections[0].heading == "Grounds for Bail"
    assert reconstructed.sections[0].claims[0].status == ClaimStatus.VERIFIED
