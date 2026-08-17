"""
Semantic Chunking (extends Topic 1)
=======================================
Where this sits relative to your existing 3 strategies:
- fixed_size_chunk    : splits by word count, blind to meaning
- sentence_aware_chunk: splits by word count but respects sentence boundaries
- recursive_chunk     : splits by document structure (paragraphs -> sentences)
None of these look at MEANING. They can (and often do) cut a chunk right in
the middle of a coherent idea just because a word-count budget ran out, or
merge two unrelated ideas into one chunk just because they're both short.

Semantic chunking fixes this by using embeddings to find where the TOPIC
actually shifts, and splitting there instead of at an arbitrary size:

1. Split the text into sentences.
2. Embed every sentence.
3. Walk through consecutive sentence pairs, computing cosine similarity
   between each pair's embeddings.
4. A big DROP in similarity between sentence i and i+1 means the topic
   likely changed there -- that's a "breakpoint." Cut the chunk there.
5. Sentences between breakpoints get grouped into one chunk.

The key design decision is how you decide "big drop": a fixed similarity
threshold (e.g. "always split below 0.5") is fragile because raw similarity
scores drift depending on the embedding model and domain. The more standard
approach (used by LlamaIndex's semantic splitter, which this mirrors) is
PERCENTILE-based: compute all the similarity drops in the document, and split
at the ones in, say, the bottom 5% (i.e. the drops that are unusually large
FOR THIS DOCUMENT). This adapts automatically instead of needing per-corpus
threshold tuning.

Trade-off worth stating out loud in an interview: semantic chunking is
strictly more expensive than the other 3 strategies -- it requires embedding
every sentence up front (an embedding call per sentence, not per chunk), so
for a large corpus this is a real latency/cost cost at ingestion time, in
exchange for better-formed chunks at retrieval time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Callable, Optional
import numpy as np

from chunking import _HAS_NLTK

if _HAS_NLTK:
    from nltk.tokenize import sent_tokenize
else:
    import re


def _sentences(text: str) -> List[str]:
    if _HAS_NLTK:
        return sent_tokenize(text)
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def semantic_chunk(
    text: str,
    embed_fn: Callable[[List[str]], np.ndarray],
    breakpoint_percentile: float = 20.0,
    buffer_size: int = 1,
) -> List[str]:
    """
    embed_fn: same shape contract as in dense_retriever.py -- a function that
              takes a list of strings and returns an (N, dim) array. Inject
              the real sentence-transformers model on your machine, or
              mock_sentence_embedder() (below) for offline testing.
    breakpoint_percentile: split at similarity drops in the bottom X percentile
              for THIS document. Lower = fewer, bigger chunks (only splits at
              the most dramatic topic shifts). Higher = more, smaller chunks.
    buffer_size: number of neighboring sentences to average together before
              embedding, a.k.a. "sentence-window" smoothing. A single sentence
              is often too short to embed meaningfully on its own (e.g. "It
              also handles this case." has almost no content by itself) --
              grouping it with its neighbors gives the embedding more signal
              before we use it to detect a breakpoint. buffer_size=1 means
              "embed this sentence plus 1 neighbor on each side."
    """
    sentences = _sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    # Build the "windowed" text used ONLY for embedding (buffer_size smoothing) --
    # the ORIGINAL sentences are still what gets grouped into final chunks.
    windowed = []
    for i in range(len(sentences)):
        lo = max(0, i - buffer_size)
        hi = min(len(sentences), i + buffer_size + 1)
        windowed.append(" ".join(sentences[lo:hi]))

    embeddings = embed_fn(windowed)

    # similarity between consecutive windowed embeddings
    sims = [
        _cosine_sim(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    if not sims:
        return [text]

    # percentile-based threshold: split at the most unusual (biggest) DROPS
    # in similarity for this specific document, not a fixed global number
    distances = [1 - s for s in sims]  # convert similarity -> distance
    threshold = np.percentile(distances, 100 - breakpoint_percentile)
    breakpoints = {i for i, d in enumerate(distances) if d >= threshold}

    chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        if (i - 1) in breakpoints:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))

    return chunks


def mock_sentence_embedder(dim: int = 64) -> Callable[[List[str]], np.ndarray]:
    """
    Same offline stand-in pattern as dense_retriever.py's mock_hash_embedder --
    hashes shared words into fixed vector slots. NOT semantically meaningful,
    but sentences sharing more vocabulary DO end up more similar under this
    scheme, which is enough to demonstrate the breakpoint-detection logic
    without a model download. Swap for a real sentence-transformers model on
    your machine to get genuine topic-shift detection.
    """
    def embed(texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            for word in t.lower().split():
                vecs[i, hash(word) % dim] += 1.0
        return vecs
    return embed


if __name__ == "__main__":
    # Deliberately 2 distinct topics back to back, no paragraph break, so
    # fixed/sentence/recursive chunking (Topic 1) would have no signal to
    # split on -- but semantic chunking should still find the seam.
    text = (
        "BM25 is a probabilistic ranking function used for sparse retrieval. "
        "It improves on TF-IDF by accounting for term frequency saturation. "
        "It also normalizes for document length so long documents aren't "
        "unfairly favored. "
        "Dense retrieval uses neural embeddings to capture semantic similarity. "
        "It can match a query to a document even with no shared words. "
        "This makes it complementary to keyword-based sparse retrieval."
    )

    print("=== Semantic chunking (mock embedder, no internet needed) ===\n")
    chunks = semantic_chunk(text, embed_fn=mock_sentence_embedder(), breakpoint_percentile=20)
    for i, c in enumerate(chunks):
        print(f"[Chunk {i}] {c}\n")

    print("On your machine, pass a real embed_fn (e.g. the same")
    print("sentence-transformers wrapper used in dense_retriever.py) and you")
    print("should see the BM25 sentences and dense-retrieval sentences split")
    print("into two separate chunks even with zero paragraph breaks in the text.")
