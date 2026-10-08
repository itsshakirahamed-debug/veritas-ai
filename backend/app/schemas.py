"""
Pydantic v2 schemas for Veritas Legal engine.
Enforces strict typing and validation for Chunk, Fact, Claim, GapReport, DraftResult, etc.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class DocType(str, Enum):
    STATUTE = "statute"
    JUDGMENT = "judgment"
    CASE_FILE = "case_file"


class ClaimStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    NOT_IN_RECORD = "not_in_record"


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the text chunk")
    doc_id: str = Field(..., description="ID of the parent document")
    doc_type: DocType = Field(..., description="Document type: statute, judgment, or case_file")
    page: int = Field(..., ge=1, description="1-based page number")
    para_id: Any = Field(..., description="Paragraph index or identifier")
    char_start: int = Field(..., ge=0, description="Start character offset in page/doc")
    char_end: int = Field(..., ge=0, description="End character offset in page/doc")
    text: str = Field(..., description="Verbatim text content of the chunk")

    model_config = ConfigDict(extra="ignore")


class Fact(BaseModel):
    field: str = Field(..., description="Field name (e.g., accused_name, FIR_number, date_of_incident)")
    value: str = Field(..., description="Extracted fact value")
    chunk_id: Optional[str] = Field(None, description="ID of chunk supporting this fact")
    quote: Optional[str] = Field(None, description="Verbatim supporting quote from the chunk")

    model_config = ConfigDict(extra="ignore")


class Claim(BaseModel):
    text: str = Field(..., description="Claim statement")
    chunk_id: Optional[str] = Field(None, description="Cited chunk ID")
    quote: Optional[str] = Field(None, description="Verbatim supporting quote")
    status: ClaimStatus = Field(default=ClaimStatus.NOT_IN_RECORD, description="Verification status")

    model_config = ConfigDict(extra="ignore")


class ContradictionValue(BaseModel):
    value: str = Field(..., description="Conflicting fact value")
    chunk_id: Optional[str] = Field(None, description="Source chunk ID")
    quote: Optional[str] = Field(None, description="Verbatim supporting quote")

    model_config = ConfigDict(extra="ignore")


class Contradiction(BaseModel):
    field: str = Field(..., description="Field with conflicting facts")
    values: List[ContradictionValue] = Field(..., min_length=2, description="List of conflicting values with sources")

    model_config = ConfigDict(extra="ignore")


class GapReport(BaseModel):
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields")
    contradictions: List[Contradiction] = Field(default_factory=list, description="List of contradictions found")
    confidence_per_field: Dict[str, float] = Field(default_factory=dict, description="Confidence scores per field")

    model_config = ConfigDict(extra="ignore")


class SectionDraft(BaseModel):
    heading: str = Field(..., description="Section title / heading")
    claims: List[Claim] = Field(default_factory=list, description="Claims under this section")

    model_config = ConfigDict(extra="ignore")


class DraftResult(BaseModel):
    sections: List[SectionDraft] = Field(default_factory=list, description="Drafted sections with claims")
    gap_report: Optional[GapReport] = Field(None, description="Gap and contradiction report")

    model_config = ConfigDict(extra="ignore")


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    doc_type: DocType
    num_chunks: int = 0
    num_pages: int = 0
    uploaded_at: str

    model_config = ConfigDict(extra="ignore")


class ReviewRequest(BaseModel):
    doc_ids: List[str] = Field(default_factory=list, description="Documents to review; empty = all indexed documents")
    doc_type: Optional[DocType] = None
    required_fields: Optional[List[str]] = Field(
        default=None,
        description="Fields that must be present; missing ones are reported in the gap report"
    )


class DraftRequest(BaseModel):
    case_id: str
    doc_type: str  # e.g., bail_application, notice, affidavit
    required_fields: Optional[List[str]] = None


class VerifyRequest(BaseModel):
    """Request to check that a quote is grounded in an indexed chunk (Hard Rule #1)."""
    chunk_id: str
    quote: str
