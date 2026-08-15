"""
Topic 6: Cross-Encoder Reranking
====================================
Why rerank after retrieval instead of just trusting the retriever's ranking:

Every retriever so far (TF-IDF, BM25, dense, hybrid) is a BI-ENCODER pattern —
query and chunk are each embedded/scored INDEPENDENTLY, then compared. This is
fast (you can precompute chunk embeddings once) but limits accuracy: the model
never actually looks at the query and chunk TOGETHER.

A cross-encoder does the opposite: it takes (query, chunk) as a SINGLE joint
input and outputs one relevance score directly. It can model interactions
between specific query words and specific chunk words that a bi-encoder's
fixed-size vector compression loses. This makes cross-encoders meaningfully
more accurate — but too slow to run over your whole corpus (you'd need one
forward pass PER chunk PER query).

The standard pattern (used here): retrieve a larger candidate set cheaply with
your bi-encoder/hybrid retriever (e.g. top 20), then rerank just those 20 with
the expensive-but-accurate cross-encoder, and keep the new top-K. Cheap recall,
expensive precision — this is the two-stage pattern almost every serious
retrieval system uses (search engines, recommender systems, RAG alike).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Callable, Optional

from chunking import Chunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        score_fn: Optional[Callable[[List[Tuple[str, str]]], List[float]]] = None,
    ):
        """
        score_fn is injectable so the rerank/sort logic below is testable
        without a model download — see mock_overlap_scorer() at the bottom.
        Default path loads a real CrossEncoder (a DIFFERENT model class from
        SentenceTransformer's bi-encoder — it takes paired inputs directly and
        is never used to build an index, only to score query-chunk pairs).
        """
        self._score = score_fn or self._default_score_fn(model_name)

    @staticmethod
    def _default_score_fn(model_name: str) -> Callable[[List[Tuple[str, str]]], List[float]]:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name)
        return lambda pairs: model.predict(pairs)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        candidates: output of any retriever (chunk, original_score) pairs.
        The original_score is discarded — cross-encoder scores are on a totally
        different scale and are strictly more trustworthy for final ranking.
        """
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        ce_scores = self._score(pairs)

        reranked = sorted(
            zip([c for c, _ in candidates], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in reranked[:top_k]]


def mock_overlap_scorer() -> Callable[[List[Tuple[str, str]]], List[float]]:
    """
    NOT a real cross-encoder — just counts shared words between query and chunk.
    Only exists to smoke-test the rerank/sort mechanics offline. Swap back to
    the real CrossEncoder default once you have model download access.
    """
    def score(pairs: List[Tuple[str, str]]) -> List[float]:
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(float(len(q_words & t_words)))
        return scores
    return score


if __name__ == "__main__":
    # Uses mock_overlap_scorer() so the rerank/sort mechanics run offline.
    # On your machine, drop score_fn to use the real cross-encoder model.
    from chunking import build_chunks
    from retrievers.bm25_retriever import BM25Retriever

    sample_docs = {
        "doc1": (
            "Retrieval-Augmented Generation combines a retriever with a generator "
            "to reduce hallucination in large language models.\n\n"
            "BM25 is a probabilistic ranking function used for sparse retrieval, "
            "improving on raw TF-IDF by accounting for document length and term "
            "saturation.\n\n"
            "Cross-encoders score a query and document jointly, which is more "
            "accurate than bi-encoder similarity but too slow to run over an "
            "entire corpus, so they are used only to rerank a small candidate set."
        )
    }
    chunks = build_chunks(sample_docs, strategy="recursive", max_chars=200)

    query = "why do we rerank a small candidate set instead of the whole corpus?"

    bm25 = BM25Retriever(chunks)
    candidates = bm25.retrieve(query, top_k=5)

    print("=== BM25 candidates (pre-rerank) ===")
    for chunk, score in candidates:
        print(f"[{score:.4f}] {chunk.text}\n")

    reranker = CrossEncoderReranker(score_fn=mock_overlap_scorer())
    reranked = reranker.rerank(query, candidates, top_k=3)

    print("=== After cross-encoder rerank ===")
    for chunk, score in reranked:
        print(f"[{score:.4f}] {chunk.text}\n")
