"""
Topic 4: Dense Retrieval — Embeddings + FAISS
=================================================
Why dense retrieval after BM25: BM25/TF-IDF only match on shared words. If a query
says "automobile" and the chunk says "car", sparse retrieval scores it near zero
even though they mean the same thing. Dense retrieval fixes this by mapping both
query and chunk into a shared vector space where MEANING determines distance, not
vocabulary overlap.

How it works:
1. An embedding model (a small transformer, NOT the LLM you'll use for generation)
   converts each chunk into a fixed-length vector (e.g. 384 or 768 dims) such that
   semantically similar text lands near each other in vector space.
2. We store all chunk vectors in a FAISS index — a data structure built for fast
   nearest-neighbor search over potentially millions of vectors (this is the same
   kind of index a production vector DB like Qdrant/Weaviate/Pinecone uses under
   the hood).
3. At query time, we embed the query with the SAME model, then ask FAISS for the
   chunks whose vectors are closest to the query vector.

Model choice: 'all-MiniLM-L6-v2' — 384-dim, fast, good enough for a benchmark.
In production you'd consider larger models (e.g. bge-large, e5-large) that trade
speed for retrieval quality — worth A/B-ing later in your RAGAS eval.

Index choice: IndexFlatIP (inner product) on L2-normalized vectors == cosine
similarity search. It's exact (no approximation), which is fine at your corpus
scale — HNSW/IVF indexes only matter once you're at millions of vectors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Callable, Optional
import numpy as np
import faiss

from chunking import Chunk


class DenseRetriever:
    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str = "all-MiniLM-L6-v2",
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        """
        embed_fn is injectable so this class isn't hard-wired to
        sentence-transformers: swap in OpenAI/Cohere embeddings, or (as below)
        a mock embedder for offline testing of the indexing/search logic —
        without touching anything past this constructor.
        """
        self.chunks = chunks
        self._embed = embed_fn or self._default_embed_fn(model_name)

        embeddings = self._embed([c.text for c in chunks]).astype("float32")
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
        self.embeddings = embeddings

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    @staticmethod
    def _default_embed_fn(model_name: str) -> Callable[[List[str]], np.ndarray]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return lambda texts: model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query_vec = self._embed([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)
        # indices/scores come back as (1, top_k) arrays — flatten the single row
        return [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]


def mock_hash_embedder(dim: int = 64) -> Callable[[List[str]], np.ndarray]:
    """
    NOT semantically meaningful — just hashes words into fixed vector slots.
    Only exists to let you smoke-test the FAISS indexing/search plumbing without
    an internet connection. Swap back to the real sentence-transformers default
    once you have model download access; that's when you'll see actual semantic
    matches (e.g. "car" retrieving an "automobile" chunk).
    """
    def embed(texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            for word in t.lower().split():
                vecs[i, hash(word) % dim] += 1.0
        return vecs
    return embed


if __name__ == "__main__":
    from chunking import build_chunks
    from retrievers.bm25_retriever import BM25Retriever

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

    query = "how does sparse retrieval improve on term frequency scoring?"

    print("=== Dense — smoke test with mock hash embedder (no internet needed) ===")
    dense = DenseRetriever(chunks, embed_fn=mock_hash_embedder())
    for chunk, score in dense.retrieve(query, top_k=3):
        print(f"[{score:.4f}] {chunk.text}\n")

    print("=== BM25 (for comparison) ===")
    bm25 = BM25Retriever(chunks)
    for chunk, score in bm25.retrieve(query, top_k=3):
        print(f"[{score:.4f}] {chunk.text}\n")

    print("On your machine: DenseRetriever(chunks)  <- no embed_fn needed,")
    print("downloads the real model and gives genuine semantic matches, e.g.")
    print("a 'car' query retrieving an 'automobile' chunk with zero word overlap.")
