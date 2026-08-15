"""
Topic 5: Hybrid Retrieval — BM25 + Dense Fusion
===================================================
Why hybrid: BM25 wins on exact-term queries (IDs, names, acronyms, jargon a dense
model was never trained to embed distinctly). Dense wins on paraphrase/semantic
queries. Neither dominates the other across all query types — production RAG
systems almost always run both and fuse the results rather than picking one.

Fusion strategy — Reciprocal Rank Fusion (RRF):
Instead of combining raw scores (which live on different, incomparable scales —
BM25 scores are unbounded, cosine similarity is 0-1), RRF combines RANKS:

    RRF_score(doc) = sum over each retriever of  1 / (k + rank_in_that_retriever)

A chunk that ranks highly in BOTH retrievers gets a much higher fused score than
one that ranks highly in only one. k (typically 60) is a smoothing constant that
dampens the impact of rank differences far down the list.

This is deliberately simpler than learned fusion (e.g. training a weighting
model) — RRF is a strong, parameter-light baseline used in real hybrid search
systems (e.g. Elasticsearch's RRF support, Weaviate's hybrid search) precisely
because it doesn't require score calibration between retrievers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Dict
from chunking import Chunk


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, rrf_k: int = 60):
        """
        Takes already-constructed BM25Retriever and DenseRetriever instances
        (both built over the SAME chunk list) rather than rebuilding them here —
        keeps this class focused purely on fusion logic.
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> List[Tuple[Chunk, float]]:
        """
        candidate_pool: how many results to pull from EACH retriever before fusing.
        Pull more than top_k so a chunk that's e.g. rank 15 in BM25 but rank 2 in
        dense still has a chance to surface after fusion.
        """
        bm25_results = self.bm25.retrieve(query, top_k=candidate_pool)
        dense_results = self.dense.retrieve(query, top_k=candidate_pool)

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Chunk] = {}

        for rank, (chunk, _score) in enumerate(bm25_results):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_lookup[chunk.id] = chunk

        for rank, (chunk, _score) in enumerate(dense_results):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_lookup[chunk.id] = chunk

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(chunk_lookup[chunk_id], score) for chunk_id, score in ranked]


if __name__ == "__main__":
    # Uses the mock hash embedder so fusion logic can be smoke-tested offline.
    # On your machine, drop embed_fn to use real sentence-transformers embeddings.
    from chunking import build_chunks
    from retrievers.bm25_retriever import BM25Retriever
    from retrievers.dense_retriever import DenseRetriever, mock_hash_embedder

    sample_docs = {
        "doc1": (
            "Retrieval-Augmented Generation combines a retriever with a generator "
            "to reduce hallucination in large language models.\n\n"
            "BM25 is a probabilistic ranking function used for sparse retrieval, "
            "improving on raw TF-IDF by accounting for document length and term "
            "saturation.\n\n"
            "Dense retrieval uses neural embeddings to capture semantic similarity "
            "between a query and a document, even when they don't share exact words.\n\n"
            "An automobile requires regular maintenance including oil changes and "
            "tire rotations to run reliably."
        )
    }
    chunks = build_chunks(sample_docs, strategy="recursive", max_chars=200)

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embed_fn=mock_hash_embedder())
    hybrid = HybridRetriever(bm25, dense)

    query = "how does BM25 improve on TF-IDF?"
    print("=== Hybrid (RRF) ===")
    for chunk, score in hybrid.retrieve(query, top_k=3):
        print(f"[{score:.4f}] {chunk.text}\n")
