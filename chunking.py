"""
Topic 1: Data Ingestion & Chunking Strategies
================================================
Why chunking matters: retrievers work over "documents" (chunks), not whole files.
Chunk size/strategy directly controls retrieval quality:
  - Too large  -> chunk contains multiple topics, embedding/TF-IDF vector gets diluted,
                  retrieval precision drops.
  - Too small  -> chunk loses context, the LLM generation step gets fragments that
                  don't make sense on their own.

This module gives you 3 chunking strategies so you can A/B them later in the benchmark.
"""

from dataclasses import dataclass
from typing import List
import re

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


@dataclass
class Chunk:
    """A single chunk with metadata — keeping metadata is what lets you trace
    a retrieved chunk back to its source document later (important for citations
    in your RAG chat system)."""
    id: str
    text: str
    source_doc: str
    chunk_index: int


def load_corpus_from_texts(docs: dict) -> dict:
    """
    docs: {doc_id: raw_text}
    Just a passthrough right now — but this is the seam where you'd later plug in
    PDF/HTML loaders (e.g. via unstructured, PyMuPDF) without touching chunking logic.
    """
    return docs


def fixed_size_chunk(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """
    Splits by raw word count with overlap.
    - Simplest strategy, used as your control/baseline in the benchmark.
    - Overlap prevents a sentence from being cleanly cut in half at a chunk boundary,
      losing meaning on both sides.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def sentence_aware_chunk(text: str, max_words: int = 200) -> List[str]:
    """
    Groups whole sentences together until the word budget is hit, instead of
    cutting mid-sentence. Better semantic coherence than fixed_size_chunk,
    at the cost of variable chunk sizes (worse for models needing roughly
    equal chunk lengths).
    """
    if _HAS_NLTK:
        sentences = sent_tokenize(text)
    else:
        # naive fallback: split on '.', '!', '?' followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def recursive_chunk(text: str, max_chars: int = 1000, separators=None) -> List[str]:
    """
    LangChain-style recursive splitter: try to split on the "biggest" separator
    first (paragraph breaks), and only fall back to smaller separators
    (sentences, then words) if a piece is still too big.
    This gives the best balance of respecting document structure vs. hitting
    a target chunk size — it's the strategy most production RAG systems default to.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, seps: List[str]) -> List[str]:
        if len(text) <= max_chars or not seps:
            return [text] if text.strip() else []

        sep = seps[0]
        pieces = text.split(sep)
        results = []
        buffer = ""

        for piece in pieces:
            candidate = (buffer + sep + piece) if buffer else piece
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    results.append(buffer)
                # piece itself might still be too big -> recurse with next separator
                if len(piece) > max_chars:
                    results.extend(_split(piece, seps[1:]))
                    buffer = ""
                else:
                    buffer = piece

        if buffer:
            results.append(buffer)
        return results

    return [c for c in _split(text, separators) if c.strip()]


def build_chunks(docs: dict, strategy: str = "recursive", **kwargs) -> List[Chunk]:
    """
    Entry point for the benchmark pipeline. Given {doc_id: text}, returns a flat
    list of Chunk objects tagged with which document + position they came from.
    """
    strategy_fn = {
        "fixed": fixed_size_chunk,
        "sentence": sentence_aware_chunk,
        "recursive": recursive_chunk,
    }.get(strategy)

    if strategy_fn is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    all_chunks = []
    for doc_id, text in docs.items():
        pieces = strategy_fn(text, **kwargs)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(id=f"{doc_id}_{i}", text=piece, source_doc=doc_id, chunk_index=i)
            )
    return all_chunks


if __name__ == "__main__":
    # quick smoke test
    sample_docs = {
        "doc1": (
            "Retrieval-Augmented Generation combines a retriever with a generator. "
            "The retriever finds relevant chunks from a knowledge base. "
            "The generator then uses those chunks as context to produce an answer. "
            "This reduces hallucination compared to using an LLM alone.\n\n"
            "Chunking strategy has a large effect on retrieval quality. "
            "Poor chunking leads to context that is either too broad or too narrow."
        )
    }

    for strat, kwargs in [
        ("fixed", {"chunk_size": 15, "overlap": 3}),
        ("sentence", {"max_words": 20}),
        ("recursive", {"max_chars": 150}),
    ]:
        print(f"\n--- {strat} ---")
        chunks = build_chunks(sample_docs, strategy=strat, **kwargs)
        for c in chunks:
            print(f"[{c.id}] {c.text}")
