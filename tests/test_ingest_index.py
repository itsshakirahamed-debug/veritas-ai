"""
Unit tests for Step 2: Ingestion, Chunking, and Qdrant + BM25 Indexing on sample PDFs.
"""

import os
import sys
import tempfile
import pymupdf as fitz  # PyMuPDF
import pytest
from pathlib import Path

# Add backend directory to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.schemas import DocType, Chunk
from app.ingest.parser import PDFParser
from app.index.chunker import ClauseChunker
from app.index.vector_store import IndexManager


@pytest.fixture
def temp_sample_pdfs():
    """Creates 3 sample legal PDFs for testing ingestion and indexing."""
    temp_dir = tempfile.mkdtemp()
    
    # 1. Statute PDF
    statute_path = os.path.join(temp_dir, "statute_crpc.pdf")
    doc1 = fitz.open()
    page1 = doc1.new_page()
    page1.insert_text((50, 50), "SECTION 437: BAIL IN NON-BAILABLE OFFENCES\n\nWhen any person accused of, or suspected of, the commission of any non-bailable offence is arrested or detained without warrant by an officer in charge of a police station or appears before a Court, he may be released on bail by discretion of the Magistrate.\n\nProvided that such person shall not be so released if there appear reasonable grounds for believing that he has been guilty of an offence punishable with death or imprisonment for life.")
    doc1.save(statute_path)
    doc1.close()

    # 2. Judgment PDF
    judgment_path = os.path.join(temp_dir, "judgment_sc.pdf")
    doc2 = fitz.open()
    page2 = doc2.new_page()
    page2.insert_text((50, 50), "IN THE SUPREME COURT OF INDIA\nCRIMINAL APPELLATE JURISDICTION\n\nSatender Kumar Antil v. Central Bureau of Investigation (2022)\n\nParagraph 19: Personal liberty is a priceless value. The policy of the criminal law is that bail is the rule and jail is the exception. Where custodial interrogation of the accused is no longer required, continued detention serves no lawful purpose.")
    doc2.save(judgment_path)
    doc2.close()

    # 3. Case File FIR PDF
    casefile_path = os.path.join(temp_dir, "fir_case.pdf")
    doc3 = fitz.open()
    page3 = doc3.new_page()
    page3.insert_text((50, 50), "FIRST INFORMATION REPORT (FIR No. 89/2024)\nStation: Sadar Police Station, New Delhi\nDate of Incident: 12th January 2024\n\nAccused Name: Rajesh Kumar, Son of Ramesh Kumar, resident of House 45, Green Park, New Delhi.\nAlleged Offences: Section 420 (Cheating) and Section 406 (Criminal Breach of Trust) of Indian Penal Code.")
    doc3.save(casefile_path)
    doc3.close()

    yield {
        "statute": statute_path,
        "judgment": judgment_path,
        "case_file": casefile_path
    }


def test_pdf_parser(temp_sample_pdfs):
    parser = PDFParser(doc_id="doc_statute_01", doc_type=DocType.STATUTE)
    paragraphs = parser.parse_pdf(temp_sample_pdfs["statute"])

    assert len(paragraphs) >= 1
    p = paragraphs[0]
    assert p["doc_id"] == "doc_statute_01"
    assert p["doc_type"] == DocType.STATUTE
    assert p["page"] == 1
    assert "SECTION 437" in p["text"]
    assert p["char_end"] > p["char_start"]


def test_clause_chunker(temp_sample_pdfs):
    parser = PDFParser(doc_id="doc_fir_89", doc_type=DocType.CASE_FILE)
    paragraphs = parser.parse_pdf(temp_sample_pdfs["case_file"])

    chunker = ClauseChunker(target_min_chars=100, target_max_chars=1000)
    chunks = chunker.create_chunks(paragraphs)

    assert len(chunks) >= 1
    chk = chunks[0]
    assert isinstance(chk, Chunk)
    assert chk.doc_id == "doc_fir_89"
    assert chk.doc_type == DocType.CASE_FILE
    assert chk.page == 1
    assert "FIR No. 89/2024" in chk.text or "Rajesh Kumar" in chk.text


def test_indexing_and_search(temp_sample_pdfs):
    # Ingest 3 sample PDFs
    parser_statute = PDFParser(doc_id="stat_01", doc_type=DocType.STATUTE)
    parser_judgment = PDFParser(doc_id="judg_01", doc_type=DocType.JUDGMENT)
    parser_case = PDFParser(doc_id="case_01", doc_type=DocType.CASE_FILE)

    paras_statute = parser_statute.parse_pdf(temp_sample_pdfs["statute"])
    paras_judgment = parser_judgment.parse_pdf(temp_sample_pdfs["judgment"])
    paras_case = parser_case.parse_pdf(temp_sample_pdfs["case_file"])

    chunker = ClauseChunker(target_min_chars=50, target_max_chars=800)
    all_chunks = []
    all_chunks.extend(chunker.create_chunks(paras_statute))
    all_chunks.extend(chunker.create_chunks(paras_judgment))
    all_chunks.extend(chunker.create_chunks(paras_case))

    assert len(all_chunks) >= 3

    # Index in vector store (in-memory mode for test speed)
    index_mgr = IndexManager(use_in_memory=True)
    num_indexed = index_mgr.index_chunks(all_chunks)
    assert num_indexed == len(all_chunks)

    # Test Vector Search
    vec_results = index_mgr.search_vector("bail is the rule and jail is the exception", top_k=2)
    assert len(vec_results) > 0
    assert "Satender Kumar Antil" in vec_results[0]["text"] or "bail" in vec_results[0]["text"].lower()

    # Test BM25 Keyword Search
    bm25_results = index_mgr.search_bm25("Rajesh Kumar", top_k=2)
    assert len(bm25_results) > 0
    assert "Rajesh Kumar" in bm25_results[0]["text"]
