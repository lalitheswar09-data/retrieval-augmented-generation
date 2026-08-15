"""
Topic 2: Sparse Retrieval — TF-IDF
=====================================
Why TF-IDF as the first retriever in the benchmark:
It's the simplest possible baseline. If your fancy dense/hybrid pipeline can't beat
TF-IDF on your eval set, that's a real signal something's wrong upstream (bad
embeddings, bad chunking) — so you always want this as your floor.

How it works:
- Every chunk becomes a sparse vector where each dimension = a vocabulary word.
- The value in each dimension = (term frequency in this chunk) x (inverse document
  frequency across the whole corpus) — i.e. words that are common in THIS chunk but
  RARE across the corpus score highest (they're the most distinguishing words).
- A query becomes a vector the same way, and we rank chunks by cosine similarity
  to the query vector.

Limitation you should be able to state out loud: TF-IDF has zero notion of meaning.
"car" and "automobile" are unrelated dimensions to it. That's exactly the gap
dense retrieval (Topic 4) closes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunking import Chunk


class TFIDFRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        # ngram_range=(1,2) captures some phrase-level signal ("machine learning"
        # as a unit, not just "machine" + "learning" separately) — cheap upgrade
        # over pure unigrams, still fundamentally sparse/lexical matching.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # argsort descending, take top_k
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


if __name__ == "__main__":
    from chunking import build_chunks

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
    retriever = TFIDFRetriever(chunks)

    query = "how does sparse retrieval improve on term frequency scoring?"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}\n")
    for chunk, score in results:
        print(f"[{score:.4f}] {chunk.text}\n")
