"""
Unit tests for retrieval (RRF fusion) and quote verification modules.
"""

import sys
from pathlib import Path

# Ensure backend package is in import path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.schemas import Chunk, DocType
from app.retrieve import reciprocal_rank_fusion
from app.verify import find_quote_span, verify_quote


def _chunk(text):
    return Chunk(
        chunk_id="chk_1",
        doc_id="doc_1",
        doc_type=DocType.CASE_FILE,
        page=1,
        para_id=1,
        char_start=0,
        char_end=len(text),
        text=text,
    )


def test_rrf_fusion_ranks_common_hits_first():
    list_a = [
        {"chunk_id": "c1", "score": 0.9},
        {"chunk_id": "c2", "score": 0.8},
    ]
    list_b = [
        {"chunk_id": "c3", "score": 0.95},
        {"chunk_id": "c1", "score": 0.7},
    ]
    fused = reciprocal_rank_fusion([list_a, list_b])
    ids = [f["chunk_id"] for f in fused]
    # c1 appears in both lists -> highest fused score
    assert ids[0] == "c1"
    assert set(ids) == {"c1", "c2", "c3"}
    assert fused[0]["score"] > fused[-1]["score"]


def test_find_quote_span_exact_and_case_insensitive():
    text = "The accused was arrested on 12th January 2024 at dawn."
    assert find_quote_span("arrested on 12th January 2024", text) == (16, 45)
    # case-insensitive fallback
    span = find_quote_span("ARRESTED ON 12TH", text)
    assert span is not None and text[span[0]:span[1]].lower() == "arrested on 12th"
    # whitespace-insensitive fallback
    text_spaced = "arrested   on\n12th   January 2024"
    span = find_quote_span("arrested on 12th January 2024", text_spaced)
    assert span is not None
    # not present at all
    assert find_quote_span("never mentioned", text) is None


def test_verify_quote_results():
    chunk = _chunk("Accused Name: Rajesh Kumar, aged 34 years.")

    ok = verify_quote("Rajesh Kumar", chunk)
    assert ok["verified"] is True
    assert ok["char_start"] is not None and ok["char_end"] > ok["char_start"]

    bad = verify_quote("Sunita Sharma", chunk)
    assert bad["verified"] is False

    no_quote = verify_quote("", chunk)
    assert no_quote["verified"] is False

    no_chunk = verify_quote("anything", None)
    assert no_chunk["verified"] is False
