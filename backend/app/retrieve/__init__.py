"""Retrieval module for hybrid search, RRF fusion, and reranking."""

from typing import List, Dict, Any, Optional

from ..index.vector_store import IndexManager


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked result lists with Reciprocal Rank Fusion (RRF).

    Each list must contain dicts with a ``chunk_id`` key. The fused score for a
    chunk is ``sum(1 / (k + rank))`` across every list it appears in.
    """
    scores: Dict[str, float] = {}
    merged: Dict[str, Dict[str, Any]] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the (richer) first occurrence of the chunk payload
            merged.setdefault(chunk_id, item)

    fused: List[Dict[str, Any]] = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        item = dict(merged[chunk_id])
        item["score"] = score
        fused.append(item)
    return fused


def hybrid_search(
    index_manager: IndexManager,
    query: str,
    top_k: int = 10,
    doc_id_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Dense vector + BM25 keyword search fused with RRF, then truncated to top_k."""
    if not query or not query.strip():
        return []

    vector_hits = index_manager.search_vector(
        query, top_k=max(top_k * 2, 10), doc_id_filter=doc_id_filter
    )

    bm25_hits = index_manager.search_bm25(query, top_k=max(top_k * 2, 10))
    if doc_id_filter:
        bm25_hits = [h for h in bm25_hits if h.get("doc_id") == doc_id_filter]

    if not vector_hits and not bm25_hits:
        return []
    if not vector_hits:
        return bm25_hits[:top_k]
    if not bm25_hits:
        return vector_hits[:top_k]

    return reciprocal_rank_fusion([vector_hits, bm25_hits])[:top_k]
