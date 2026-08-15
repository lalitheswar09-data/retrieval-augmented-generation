"""
Topic 3: Sparse Retrieval — BM25
====================================
Why BM25 after TF-IDF: BM25 is TF-IDF's more refined sibling and is still the
default sparse retriever in most production RAG systems (e.g. Elasticsearch's
default scoring). Two fixes over raw TF-IDF:

1. Term frequency saturation:
   In TF-IDF, if a word appears 10x more, its score can scale ~10x more.
   BM25 uses a saturation curve (the k1 parameter) so that going from 1 -> 2
   occurrences of a term matters a lot, but going from 20 -> 21 occurrences
   barely moves the score. This matches intuition: a chunk repeating "RAG" 20
   times isn't 20x more "about RAG" than one that says it twice.

2. Document length normalization:
   Long chunks naturally contain more words, so raw term overlap unfairly favors
   them. BM25's `b` parameter penalizes matches in longer documents relative to
   the corpus's average document length, so a short precise chunk isn't
   drowned out by a long chunk that happens to mention the query term once.

k1 (typically 1.2-2.0) controls saturation strength; b (0-1) controls how much
length normalization applies. You'll tune these later during the RAGAS eval step.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from rank_bm25 import BM25Okapi

from chunking import Chunk


def _tokenize(text: str) -> List[str]:
    """Simple lowercase whitespace tokenizer. BM25 operates on token overlap,
    so tokenization quality matters — in a real pipeline you'd swap this for
    a proper tokenizer (e.g. spaCy) to handle punctuation/stemming better."""
    return text.lower().split()


class BM25Retriever:
    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


if __name__ == "__main__":
    from chunking import build_chunks
    from retrievers.tfidf_retriever import TFIDFRetriever

    sample_docs = {
        "doc1": (
            "Retrieval-Augmented Generation combines a retriever with a generator "
            "to reduce hallucination in large language models.\n\n"
            "BM25 is a probabilistic ranking function used for sparse retrieval, "
            "improving on raw TF-IDF by accounting for document length and term "
            "saturation.\n\n"
            "Dense retrieval uses neural embeddings to capture semantic similarity "
            "between a query and a document, even when they don't share exact words."
        )
    }

    chunks = build_chunks(sample_docs, strategy="recursive", max_chars=200)

    query = "how does sparse retrieval improve on term frequency scoring?"

    print("=== BM25 ===")
    bm25 = BM25Retriever(chunks)
    for chunk, score in bm25.retrieve(query, top_k=3):
        print(f"[{score:.4f}] {chunk.text}\n")

    print("=== TF-IDF (for comparison) ===")
    tfidf = TFIDFRetriever(chunks)
    for chunk, score in tfidf.retrieve(query, top_k=3):
        print(f"[{score:.4f}] {chunk.text}\n")
