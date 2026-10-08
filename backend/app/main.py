"""
FastAPI Main Application for Veritas Legal.
Provides CORS enabled REST API endpoints for Document Upload, Review, Drafting,
Hybrid Search, Quote Verification, Source Retrieval, and Health checks.

When no ANTHROPIC_API_KEY is configured (or ?use_mock=true is passed), review and
draft endpoints fall back to a *grounded* deterministic engine that only emits
facts/claims traceable to indexed chunks — never fabricated content.
"""

import os
import re
import uuid
import tempfile
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    DocType,
    ClaimStatus,
    Fact,
    Claim,
    ContradictionValue,
    Contradiction,
    GapReport,
    SectionDraft,
    DraftResult,
    DocumentMeta,
    ReviewRequest,
    DraftRequest,
    VerifyRequest,
)
from .ingest.parser import PDFParser
from .index.chunker import ClauseChunker
from .index.vector_store import IndexManager
from .retrieve import hybrid_search
from .verify import verify_quote
from .agent.llm_drafter import generate_draft
from .agent.llm_reviewer import generate_review

app = FastAPI(
    title="VERITAS LEGAL API",
    description="Agentic Legal Assistant — Don't just cite. Prove.",
    version="1.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Index manager: in-memory by default (zero-setup), Qdrant when configured.
# Set QDRANT_IN_MEMORY=false to use the docker-compose Qdrant at QDRANT_HOST:QDRANT_PORT.
_QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_USE_IN_MEMORY = os.environ.get("QDRANT_IN_MEMORY", "true").lower() in {"1", "true", "yes"}
_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

index_manager = IndexManager(
    qdrant_host=_QDRANT_HOST,
    qdrant_port=_QDRANT_PORT,
    use_in_memory=_USE_IN_MEMORY,
    embedding_model_name=_EMBEDDING_MODEL,
)
doc_registry: Dict[str, DocumentMeta] = {}


# ---------------------------------------------------------------------------
# Helpers: grounded (non-LLM) review / draft engines
# ---------------------------------------------------------------------------

# "Field Name: value" spans, e.g. "Accused Name: Rajesh Kumar, aged 34 years".
# Anchored at line start or right after a sentence-ending period so mid-sentence
# colons (times like "14:30 hrs") are never mistaken for field labels.
_FIELD_SPAN_RE = re.compile(
    r"(?:^|\.\s)([A-Z][A-Za-z0-9][A-Za-z0-9 _()/&'\-.]{0,58}):\s+([^\n]{1,300})",
    re.MULTILINE,
)


def _normalize_field(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")


def _norm_phrase(text: str) -> str:
    """Normalize a phrase/line for substring comparison (case + whitespace + underscores)."""
    return " ".join(str(text).replace("_", " ").split()).lower()


def _chunks_for_docs(doc_ids: List[str]) -> List[Any]:
    """Chunks for the requested doc_ids; empty list means ALL indexed documents."""
    all_chunks = list(index_manager.chunks_registry.values())
    if not doc_ids:
        return all_chunks
    wanted = set(doc_ids)
    return [c for c in all_chunks if c.doc_id in wanted]


def _extract_field_facts(chunks: List[Any]) -> Tuple[List[Fact], List[Contradiction]]:
    """Deterministic fact extraction from verbatim `Field: value` lines.

    Every returned Fact quotes an exact line of its source chunk, so the quote
    is verbatim by construction (Hard Rule #1). Conflicting values for the same
    field across chunks are surfaced as Contradictions (Hard Rule #4).
    """
    facts: List[Fact] = []
    seen: Dict[str, set] = {}   # field -> set of normalized values
    contradiction_values: Dict[str, List[ContradictionValue]] = {}

    for chunk in chunks:
        if len(facts) >= 60:
            break
        for m in _FIELD_SPAN_RE.finditer(chunk.text):
            if len(facts) >= 60:
                break
            raw_field, value = m.group(1).strip(), m.group(2).strip()
            # Reject sentence-like labels (too many words for a field name)
            if len(raw_field.split()) > 8:
                continue
            # Reject time fragments like field "... at 14" + value "30 hrs"
            if re.search(r"\d{1,2}$", raw_field) and re.match(r"\d{2}\b", value):
                continue
            field = _normalize_field(raw_field)
            if not field:
                continue

            # Quote is the matched span itself — a verbatim substring by construction
            quote = chunk.text[m.start(1):m.end()].strip()
            if not quote:
                continue

            facts.append(Fact(
                field=field,
                value=value,
                chunk_id=chunk.chunk_id,
                quote=quote,
            ))

            norm_value = " ".join(value.lower().split())
            seen.setdefault(field, set())
            if norm_value not in seen[field]:
                seen[field].add(norm_value)
                contradiction_values.setdefault(field, []).append(ContradictionValue(
                    value=value,
                    chunk_id=chunk.chunk_id,
                    quote=quote,
                ))

    contradictions = [
        Contradiction(field=field, values=values[:4])
        for field, values in contradiction_values.items()
        if len(values) >= 2
    ]
    return facts, contradictions


def _find_required_fields(required_fields: List[str], chunks: List[Any]) -> Tuple[Dict[str, Fact], List[str]]:
    """Locate required fields in the corpus; returns (found_facts, missing_fields).

    A field counts as found only if its phrase appears verbatim in some chunk;
    the containing line is returned as the supporting quote.
    """
    found: Dict[str, Fact] = {}
    missing: List[str] = []

    for req in required_fields or []:
        target = _norm_phrase(req)
        if not target:
            continue
        hit: Optional[Fact] = None
        for chunk in chunks:
            if target not in _norm_phrase(chunk.text):
                continue
            for line in chunk.text.splitlines():
                if target in _norm_phrase(line):
                    hit = Fact(
                        field=_normalize_field(req),
                        value=line.strip(),
                        chunk_id=chunk.chunk_id,
                        quote=line.strip(),
                    )
                    break
            if hit:
                break
        if hit:
            found[_normalize_field(req)] = hit
        else:
            missing.append(req)
    return found, missing


def _grounded_review(doc_ids: List[str], required_fields: List[str]) -> Dict[str, Any]:
    chunks = _chunks_for_docs(doc_ids)
    facts, contradictions = _extract_field_facts(chunks)
    found, missing = _find_required_fields(required_fields, chunks)

    # Required fields found by phrase scan but not by `Field: value` extraction
    for field, fact in found.items():
        if field not in {f.field for f in facts}:
            facts.append(fact)

    # Quotes are verbatim by construction → deterministic confidence of 1.0
    confidence = {f.field: 1.0 for f in facts if f.quote and f.chunk_id}

    return {
        "doc_ids": doc_ids or sorted(doc_registry.keys()),
        "extracted_facts": facts,
        "gap_report": GapReport(
            missing_fields=missing,
            contradictions=contradictions,
            confidence_per_field=confidence,
        ),
    }


def _first_sentence(text: str) -> str:
    """First sentence of text, verbatim (a substring of the original).

    Skips abbreviation-style periods (followed by a digit or lowercase word,
    or a very short last word like "v.").
    """
    sentence = ""
    for m in re.finditer(r"[.!?]", text):
        after = text[m.end():m.end() + 24].lstrip()
        word = re.match(r"[A-Za-z]+", after)
        if after[:1].isdigit() or (word and word.group(0)[0].islower()):
            continue
        candidate = text[:m.end()].strip()
        if len(candidate) < 15:
            continue
        if re.search(r"\b[A-Za-z]{1,3}\.$", candidate):
            continue  # e.g. "v.", "Sec.", "No."
        sentence = candidate
        break

    if not sentence:
        prefix = text[:400]
        cut = prefix.rfind(" ")
        if 50 < cut < len(prefix):
            prefix = prefix[:cut]
        sentence = prefix.strip()

    if len(sentence) > 400:
        cut = sentence[:400].rfind(" ")
        if cut > 100:
            sentence = sentence[:cut]
    return sentence


def _grounded_draft(doc_type: str, required_fields: List[str]) -> DraftResult:
    """Deterministic draft built strictly from indexed chunks (no LLM)."""
    chunks = _chunks_for_docs([])
    sections: List[SectionDraft] = []

    # One section per source document, with verbatim grounded claims
    by_doc: Dict[str, List[Any]] = {}
    for chk in chunks:
        by_doc.setdefault(chk.doc_id, []).append(chk)

    for doc_id, doc_chunks in by_doc.items():
        meta = doc_registry.get(doc_id)
        heading = f"RECORD: {meta.filename}" if meta else f"RECORD: {doc_id}"
        claims: List[Claim] = []
        for chk in doc_chunks[:4]:
            quote = _first_sentence(chk.text)
            if not quote:
                continue
            text = quote if len(quote) < 390 else quote[:390] + "…"
            claims.append(Claim(
                text=text,
                chunk_id=chk.chunk_id,
                quote=quote,
                status=ClaimStatus.VERIFIED,
            ))
        if claims:
            sections.append(SectionDraft(heading=heading, claims=claims))

    # Required fields: grounded value or the literal gap marker (Hard Rule #3)
    found, missing = _find_required_fields(required_fields, chunks)
    field_claims: List[Claim] = []
    confidence: Dict[str, float] = {}
    for req in required_fields or []:
        fact = found.get(_normalize_field(req))
        if fact:
            field_claims.append(Claim(
                text=f"{req}: {fact.value}",
                chunk_id=fact.chunk_id,
                quote=fact.quote,
                status=ClaimStatus.VERIFIED,
            ))
            confidence[_normalize_field(req)] = 1.0
        else:
            field_claims.append(Claim(
                text=f"{req}: [INFORMATION NOT IN RECORD]",
                chunk_id=None,
                quote=None,
                status=ClaimStatus.NOT_IN_RECORD,
            ))
    if field_claims:
        sections.insert(0, SectionDraft(heading="REQUIRED FIELD STATUS", claims=field_claims))

    return DraftResult(
        sections=sections,
        gap_report=GapReport(
            missing_fields=missing,
            contradictions=[],
            confidence_per_field=confidence,
        ),
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine": "VERITAS LEGAL",
        "hard_rules_active": True,
        "indexed_documents": len(doc_registry),
        "indexed_chunks": len(index_manager.chunks_registry),
        "llm_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "index_backend": "in_memory" if _USE_IN_MEMORY else f"{_QDRANT_HOST}:{_QDRANT_PORT}",
    }


@app.post("/documents/upload", response_model=DocumentMeta)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: Optional[DocType] = Form(DocType.CASE_FILE)
):
    """
    Ingests and indexes a PDF document into the vector store & BM25 index.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parser = PDFParser(doc_id=doc_id, doc_type=doc_type or DocType.CASE_FILE)
        paragraphs = parser.parse_pdf(tmp_path)

        if not paragraphs:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in PDF (it may be a scanned image without OCR text)."
            )

        # Count unique pages in parsed paragraphs
        num_pages = len(set(p["page"] for p in paragraphs))

        chunker = ClauseChunker()
        chunks = chunker.create_chunks(paragraphs)

        num_indexed = index_manager.index_chunks(chunks)

        meta = DocumentMeta(
            doc_id=doc_id,
            filename=file.filename,
            doc_type=doc_type or DocType.CASE_FILE,
            num_chunks=num_indexed,
            num_pages=num_pages,
            uploaded_at=datetime.now(timezone.utc).isoformat()
        )
        doc_registry[doc_id] = meta
        return meta
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Parsing/Indexing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/documents", response_model=List[DocumentMeta])
def list_documents():
    """Lists all indexed documents, most recently uploaded first."""
    return sorted(doc_registry.values(), key=lambda d: d.uploaded_at, reverse=True)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Removes a document and all of its chunks from the index."""
    if doc_id not in doc_registry:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in registry.")
    meta = doc_registry.pop(doc_id)
    removed = index_manager.remove_document(doc_id)
    return {"doc_id": doc_id, "filename": meta.filename, "removed_chunks": removed}


@app.get("/documents/{doc_id}/chunks")
def list_doc_chunks(doc_id: str):
    """Lists all chunks belonging to a specific document."""
    if doc_id not in doc_registry:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in registry.")
    chunks = [
        {
            "chunk_id": chk.chunk_id,
            "page": chk.page,
            "para_id": chk.para_id,
            "char_start": chk.char_start,
            "char_end": chk.char_end,
            "text_preview": chk.text[:200] + ("..." if len(chk.text) > 200 else "")
        }
        for chk in index_manager.chunks_registry.values()
        if chk.doc_id == doc_id
    ]
    return {"doc_id": doc_id, "total_chunks": len(chunks), "chunks": chunks}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    doc_id: Optional[str] = Query(None, description="Restrict search to one document"),
):
    """Hybrid search: dense vector + BM25 keyword, fused with Reciprocal Rank Fusion."""
    if doc_id and doc_id not in doc_registry:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in registry.")

    results = hybrid_search(index_manager, q, top_k=top_k, doc_id_filter=doc_id)
    return {"query": q, "total": len(results), "results": results}


@app.post("/verify")
def verify_claim(req: VerifyRequest):
    """Verifies that a quote is grounded verbatim in the cited chunk (Hard Rule #1)."""
    chunk = index_manager.chunks_registry.get(req.chunk_id)
    result = verify_quote(req.quote, chunk)
    return {
        "chunk_id": req.chunk_id,
        "verified": result["verified"],
        "char_start": result["char_start"],
        "char_end": result["char_end"],
        "reason": result["reason"],
        "doc_id": chunk.doc_id if chunk else None,
        "page": chunk.page if chunk else None,
    }


@app.get("/source/{chunk_id}")
def get_source_chunk(chunk_id: str):
    """
    Returns chunk text, page, and char span offsets for source highlighting.
    Unknown chunks return 404 — never fabricated sample text (Hard Rule #1).
    """
    chunk = index_manager.chunks_registry.get(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found in index.")
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type.value,
        "page": chunk.page,
        "para_id": chunk.para_id,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "text": chunk.text
    }


@app.post("/review", response_model=Dict[str, Any])
def review_case(req: ReviewRequest, use_mock: bool = Query(False)):
    """
    Extracts facts with source quotes and generates GapReport (missing fields & contradictions).
    With no ANTHROPIC_API_KEY (or ?use_mock=true), a grounded deterministic engine is used.
    """
    matching_chunks = _chunks_for_docs(req.doc_ids)

    if not matching_chunks:
        raise HTTPException(
            status_code=400,
            detail="No indexed chunks found for the requested documents. Upload a PDF first."
        )

    required_fields = req.required_fields or []

    # Grounded fallback path (no key or explicit mock)
    if use_mock or not os.environ.get("ANTHROPIC_API_KEY"):
        return _grounded_review(req.doc_ids, required_fields)

    # Live LLM Generation Path
    case_chunks = [
        {"chunk_id": chk.chunk_id, "doc_id": chk.doc_id, "text": chk.text}
        for chk in matching_chunks
    ]
    try:
        return generate_review(req.doc_ids or sorted(doc_registry.keys()), case_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")


@app.post("/draft", response_model=DraftResult)
def draft_document(req: DraftRequest, use_mock: bool = Query(False)):
    """
    Drafts legal document (Bail application / notice / affidavit) producing grounded Claims.
    With no ANTHROPIC_API_KEY (or ?use_mock=true), a grounded deterministic engine is used.
    """
    if not index_manager.chunks_registry:
        raise HTTPException(
            status_code=400,
            detail="No indexed documents found to draft from. Please upload a case file first."
        )

    required_fields = req.required_fields or []

    # Grounded fallback path (no key or explicit mock)
    if use_mock or not os.environ.get("ANTHROPIC_API_KEY"):
        return _grounded_draft(req.doc_type, required_fields)

    # Live LLM Generation Path: all indexed chunks form the case record
    case_chunks = [
        {
            "chunk_id": chk.chunk_id,
            "doc_id": chk.doc_id,
            "text": chk.text
        }
        for chk in index_manager.chunks_registry.values()
    ]
    try:
        return generate_draft(req.doc_type, required_fields, case_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Static frontend (production single-service deploy: API + UI on one origin)
# Registered last so every API route above takes precedence.
# ---------------------------------------------------------------------------
_FRONTEND_DIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
)

if os.path.isdir(_FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
