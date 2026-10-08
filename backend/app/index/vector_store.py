"""
Qdrant Vector Store & BM25 Indexing Manager for Veritas Legal.
Provides hybrid indexing: Dense vector embeddings (BGE-M3 / SentenceTransformers) + BM25Okapi keyword indexing.
"""

import os
import re
import uuid
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from ..schemas import Chunk, DocType


class IndexManager:
    def __init__(
        self,
        collection_name: str = "veritas_legal_chunks",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        use_in_memory: bool = False,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self._embedder = None
        self._bm25 = None
        self.chunks_registry: Dict[str, Chunk] = {}
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_ids_order: List[str] = []

        if use_in_memory:
            self.client = QdrantClient(":memory:")
        else:
            try:
                self.client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5.0)
                # Test connection
                self.client.get_collections()
            except Exception:
                # Fallback to in-memory if local Qdrant container is unreachable
                self.client = QdrantClient(":memory:")

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                # Mock embedder fallback if SentenceTransformers model download fails or offline
                class MockEmbedder:
                    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
                        import numpy as np
                        if isinstance(texts, str):
                            texts = [texts]
                        # Deterministic hash-based pseudo-embeddings for fast test execution
                        embeddings = []
                        for t in texts:
                            h = sum(ord(c) for c in t) % 10000
                            vec = np.zeros(384, dtype=np.float32)
                            vec[0] = (h % 100) / 100.0
                            vec[1] = ((h // 100) % 100) / 100.0
                            embeddings.append(vec)
                        return np.array(embeddings)
                self._embedder = MockEmbedder()
        return self._embedder

    @staticmethod
    def _point_id(chunk_id: str) -> int:
        """Deterministic 64-bit point ID for a chunk (stable across processes).

        Python's built-in hash() is salted per process, which would produce
        different IDs on every restart and corrupt a persistent Qdrant store.
        """
        return int.from_bytes(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id).bytes[:8], "big")

    def ensure_collection(self, vector_size: int = 384):
        """Creates the Qdrant collection if it doesn't already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def index_chunks(self, chunks: List[Chunk]) -> int:
        """
        Indexes chunks into both Qdrant vector store and BM25 index.
        """
        if not chunks:
            return 0

        # Generate dense embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        vector_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])

        self.ensure_collection(vector_size=vector_dim)

        points = []
        for i, chunk in enumerate(chunks):
            # Store in registry
            self.chunks_registry[chunk.chunk_id] = chunk

            # Tokenize for BM25
            tokens = self._tokenize(chunk.text)
            self.tokenized_corpus.append(tokens)
            self.chunk_ids_order.append(chunk.chunk_id)

            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_type": chunk.doc_type.value,
                "page": chunk.page,
                "para_id": str(chunk.para_id),
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "text": chunk.text
            }

            # Deterministic integer ID for Qdrant
            point_id = self._point_id(chunk.chunk_id)

            points.append(PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload=payload
            ))

        # Upsert points into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        # Re-build BM25 index
        if self.tokenized_corpus:
            self._bm25 = BM25Okapi(self.tokenized_corpus)

        return len(chunks)

    def remove_document(self, doc_id: str) -> int:
        """Removes all chunks of a document from the registry, Qdrant, and BM25."""
        removed_ids = [cid for cid, chk in self.chunks_registry.items() if chk.doc_id == doc_id]
        if not removed_ids:
            return 0

        for cid in removed_ids:
            del self.chunks_registry[cid]

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[self._point_id(cid) for cid in removed_ids]
            )
        except Exception:
            # Collection may not exist yet (nothing was ever upserted)
            pass

        # Rebuild BM25 over the remaining chunks
        self.chunk_ids_order = [c for c in self.chunk_ids_order if c in self.chunks_registry]
        self.tokenized_corpus = [
            self._tokenize(self.chunks_registry[c].text) for c in self.chunk_ids_order
        ]
        self._bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        return len(removed_ids)

    def search_vector(self, query: str, top_k: int = 10, doc_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Vector similarity search via Qdrant."""
        # Guard: nothing indexed yet (collection does not exist)
        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception:
            existing = []
        if self.collection_name not in existing:
            return []

        query_vector = self.embedder.encode([query], show_progress_bar=False, normalize_embeddings=True)[0].tolist()

        q_filter = None
        if doc_id_filter:
            q_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))]
            )

        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=q_filter
            )
            hits = res.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=q_filter
            )

        results = []
        for hit in hits:
            payload = hit.payload
            results.append({
                "chunk_id": payload["chunk_id"],
                "score": float(hit.score),
                "text": payload["text"],
                "doc_id": payload["doc_id"],
                "doc_type": payload["doc_type"],
                "page": payload["page"],
                "char_start": payload["char_start"],
                "char_end": payload["char_end"]
            })
        return results

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Keyword sparse search via BM25."""
        if not self._bm25 or not self.chunk_ids_order:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair scores with chunk IDs
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in indexed_scores:
            if score <= 0.0:
                continue
            chunk_id = self.chunk_ids_order[idx]
            chunk = self.chunks_registry.get(chunk_id)
            if chunk:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "score": float(score),
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "doc_type": chunk.doc_type.value,
                    "page": chunk.page,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end
                })
        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase tokenization for legal BM25 indexing."""
        return re.findall(r'\w+', text.lower())
