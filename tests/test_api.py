"""
API-level tests for the FastAPI app (upload, search, verify, review, draft, delete).
Forces the grounded fallback path by removing ANTHROPIC_API_KEY for test determinism.
"""

import os
import sys
from pathlib import Path

import pytest
import pymupdf as fitz

# Ensure backend package is in import path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Keep review/draft on the deterministic grounded path during tests."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "case.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50),
        "CASE RECORD: FIR 89/2024\n\n"
        "FIRST INFORMATION REPORT (FIR No. 89/2024)\n"
        "Date of Incident: 12th January 2024\n"
        "Accused Name: Rajesh Kumar, resident of Green Park, New Delhi.\n"
        "Alleged Offences: Section 420 and Section 406 of the Indian Penal Code."
    )
    doc.save(str(path))
    doc.close()
    return str(path)


def _upload(sample_pdf):
    with open(sample_pdf, "rb") as fh:
        resp = client.post(
            "/documents/upload",
            files={"file": ("case.pdf", fh, "application/pdf")},
            data={"doc_type": "case_file"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["hard_rules_active"] is True
    assert "llm_available" in body


def test_upload_list_chunks_delete(sample_pdf):
    meta = _upload(sample_pdf)
    assert meta["num_chunks"] >= 1
    doc_id = meta["doc_id"]

    listed = client.get("/documents").json()
    assert any(d["doc_id"] == doc_id for d in listed)

    chunks = client.get(f"/documents/{doc_id}/chunks").json()
    assert chunks["total_chunks"] >= 1
    chunk_id = chunks["chunks"][0]["chunk_id"]

    # Source: known chunk 200, unknown chunk 404 (never fabricated)
    assert client.get(f"/source/{chunk_id}").status_code == 200
    assert client.get("/source/chk_does_not_exist").status_code == 404

    # Delete removes the document and its chunks
    deleted = client.request("DELETE", f"/documents/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["removed_chunks"] >= 1
    assert client.get(f"/documents/{doc_id}/chunks").status_code == 404
    assert client.get(f"/source/{chunk_id}").status_code == 404


def test_upload_rejects_non_pdf(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not a pdf")
    with open(bad, "rb") as fh:
        resp = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", fh, "text/plain")},
            data={"doc_type": "case_file"},
        )
    assert resp.status_code == 400


def test_search_hybrid(sample_pdf):
    meta = _upload(sample_pdf)
    doc_id = meta["doc_id"]
    try:
        resp = client.get("/search", params={"q": "Rajesh Kumar bail", "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert "chunk_id" in body["results"][0]
        assert "score" in body["results"][0]
        # unknown doc filter -> 404
        assert client.get("/search", params={"q": "x", "doc_id": "nope"}).status_code == 404
    finally:
        client.request("DELETE", f"/documents/{doc_id}")


def test_verify_quote_grounded(sample_pdf):
    meta = _upload(sample_pdf)
    doc_id = meta["doc_id"]
    chunk_id = client.get(f"/documents/{doc_id}/chunks").json()["chunks"][0]["chunk_id"]
    try:
        ok = client.post("/verify", json={"chunk_id": chunk_id, "quote": "Rajesh Kumar"}).json()
        assert ok["verified"] is True
        assert ok["char_start"] is not None

        bad = client.post("/verify", json={"chunk_id": chunk_id, "quote": "never said here"}).json()
        assert bad["verified"] is False

        missing = client.post("/verify", json={"chunk_id": "ghost", "quote": "x"}).json()
        assert missing["verified"] is False
    finally:
        client.request("DELETE", f"/documents/{doc_id}")


def test_review_grounded_fallback(sample_pdf):
    meta = _upload(sample_pdf)
    doc_id = meta["doc_id"]
    try:
        resp = client.post("/review", json={
            "doc_ids": [doc_id],
            "required_fields": ["accused_name", "medical_condition"],
        })
        assert resp.status_code == 200
        body = resp.json()
        facts = body["extracted_facts"]
        assert facts, "expected extracted facts"
        # Every fact must carry a chunk_id and verbatim quote (Hard Rule #1)
        for f in facts:
            assert f["chunk_id"] and f["quote"]
        # Required-field gap detection
        assert "medical_condition" in body["gap_report"]["missing_fields"]

        # Unknown doc ids -> 400
        assert client.post("/review", json={"doc_ids": ["nope"]}).status_code == 400
    finally:
        client.request("DELETE", f"/documents/{doc_id}")


def test_review_flags_contradictions_across_documents():
    """Two docs with conflicting values for the same field -> contradiction (Hard Rule #4)."""
    import pymupdf as fitz
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    doc_ids = []
    for i, date in enumerate(["12th January 2024", "14th January 2024"]):
        path = tmp / f"doc{i}.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), f"CASE RECORD\nDate of Incident: {date}\nAccused Name: Rajesh Kumar")
        doc.save(str(path))
        doc.close()
        with open(path, "rb") as fh:
            resp = client.post(
                "/documents/upload",
                files={"file": (path.name, fh, "application/pdf")},
                data={"doc_type": "case_file"},
            )
        assert resp.status_code == 200, resp.text
        doc_ids.append(resp.json()["doc_id"])

    try:
        body = client.post("/review", json={"doc_ids": doc_ids}).json()
        contradictions = body["gap_report"]["contradictions"]
        fields = [c["field"] for c in contradictions]
        assert "date_of_incident" in fields
        pair = next(c for c in contradictions if c["field"] == "date_of_incident")
        assert len(pair["values"]) >= 2
        for v in pair["values"]:
            assert v["chunk_id"] and v["quote"], "contradiction values must be sourced"
    finally:
        for doc_id in doc_ids:
            client.request("DELETE", f"/documents/{doc_id}")


def test_draft_grounded_fallback(sample_pdf):
    meta = _upload(sample_pdf)
    doc_id = meta["doc_id"]
    try:
        resp = client.post("/draft", json={
            "case_id": "case-1",
            "doc_type": "bail_application",
            "required_fields": ["accused_name", "medical_condition"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["sections"], "expected drafted sections"
        claims = [c for s in body["sections"] for c in s["claims"]]
        assert claims
        for c in claims:
            if c["status"] == "verified":
                assert c["chunk_id"] and c["quote"], "verified claims must cite a quote"
            else:
                assert "[INFORMATION NOT IN RECORD]" in c["text"]
        assert "medical_condition" in body["gap_report"]["missing_fields"]
    finally:
        client.request("DELETE", f"/documents/{doc_id}")
