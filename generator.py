"""
Topic 7: Generation Layer
=============================
This is where retrieval output finally becomes an answer. Three separate
concerns live here, kept as separate functions on purpose (each is a place
your benchmark might want to swap implementations later):

1. Context budgeting — you can't just dump all retrieved chunks into the
   prompt. LLMs have a context window, and even within that window, stuffing
   in irrelevant/low-ranked chunks measurably hurts answer quality (the
   "lost in the middle" effect — models attend less to info buried in a long
   context). So we truncate to a token/word budget, keeping only the
   highest-ranked chunks that fit.

2. Prompt assembly — the retrieved chunks get formatted with explicit source
   tags so the model CAN cite them, and so a human reading the prompt (i.e.
   you, debugging) can immediately see what evidence the model had access to.

3. Generation — the actual LLM call. Written provider-agnostic via a
   `generate_fn` you inject, with an Anthropic implementation as the default
   and a mock implementation for testing without an API key.

Why this matters for your RAGAS eval (Topic 8): RAGAS's "faithfulness" metric
checks whether the generated answer is actually supported by the retrieved
context. If your prompt doesn't clearly separate context from instructions,
or if you leak irrelevant chunks into the context, faithfulness scores drop —
so getting this layer right isn't just plumbing, it directly affects your
benchmark numbers.
"""

from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chunking import Chunk


# ---------------------------------------------------------------------------
# 1. Context budgeting
# ---------------------------------------------------------------------------
def select_chunks_within_budget(
    ranked_chunks: List[Tuple[Chunk, float]],
    max_words: int = 800,
) -> List[Chunk]:
    """
    Greedily keeps chunks in rank order until the word budget is used up.
    Word count is a rough stand-in for token count (roughly 0.75 words/token
    for English) — good enough for budgeting; swap in a real tokenizer
    (tiktoken / the Anthropic tokenizer) if you need exact counts.
    """
    selected = []
    used = 0
    for chunk, _score in ranked_chunks:
        n_words = len(chunk.text.split())
        if used + n_words > max_words:
            continue  # skip (not break) — a later, shorter chunk might still fit
        selected.append(chunk)
        used += n_words
    return selected


# ---------------------------------------------------------------------------
# 2. Prompt assembly
# ---------------------------------------------------------------------------
def build_rag_prompt(query: str, chunks: List[Chunk]) -> str:
    """
    Explicit [Source N: doc_id] tags do two things:
    - Let the model attribute claims to a specific source when asked to cite.
    - Let YOU, reading raw prompts while debugging, instantly see what
      evidence was actually available for a given answer.
    Instruction is deliberately strict about not using outside knowledge —
    this is what RAGAS's faithfulness metric is checking for.
    """
    context_block = "\n\n".join(
        f"[Source {i+1}: {c.source_doc}]\n{c.text}"
        for i, c in enumerate(chunks)
    )

    return f"""Answer the question using ONLY the sources below. If the sources don't contain enough information to answer, say so explicitly instead of guessing.

{context_block}

Question: {query}

Answer (cite sources like [Source N] where relevant):"""


# ---------------------------------------------------------------------------
# 3. Generation
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    answer: str
    prompt: str          # kept for debugging / RAGAS input
    used_chunks: List[Chunk]


def anthropic_generate_fn(model: str = "claude-sonnet-4-6") -> Callable[[str], str]:
    """
    Real generation path. Requires ANTHROPIC_API_KEY in your environment.
    Kept as a factory (not called at import time) so importing this module
    never requires an API key — only calling generate() does.
    """
    import anthropic
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


def mock_generate_fn() -> Callable[[str], str]:
    """
    No API call — just echoes which sources were in context. Lets you test
    the budgeting + prompt-assembly pipeline end-to-end without a key,
    the same offline-smoke-test pattern used in Topics 4 and 6.
    """
    def call(prompt: str) -> str:
        # count only "[Source N:" (the actual context tags), not the
        # "[Source N]" citation-format instruction in the prompt template
        n_sources = prompt.count("[Source ") - prompt.count("[Source N]")
        return f"[MOCK ANSWER — {n_sources} source(s) were in context; wire up anthropic_generate_fn() for a real answer]"

    return call


def generate_answer(
    query: str,
    ranked_chunks: List[Tuple[Chunk, float]],
    generate_fn: Optional[Callable[[str], str]] = None,
    max_context_words: int = 800,
) -> GenerationResult:
    """
    Full pipeline: rank -> budget -> assemble prompt -> generate.
    This is the function your benchmark harness (Topic 9) and chat wrapper
    (Topic 10) will actually call — everything above it is building blocks.
    """
    generate_fn = generate_fn or mock_generate_fn()

    selected = select_chunks_within_budget(ranked_chunks, max_words=max_context_words)
    prompt = build_rag_prompt(query, selected)
    answer = generate_fn(prompt)

    return GenerationResult(answer=answer, prompt=prompt, used_chunks=selected)


if __name__ == "__main__":
    from chunking import build_chunks
    from retrievers.bm25_retriever import BM25Retriever

    sample_docs = {
        "doc1": (
            "Retrieval-Augmented Generation combines a retriever with a generator "
            "to reduce hallucination in large language models.\n\n"
            "BM25 is a probabilistic ranking function used for sparse retrieval, "
            "improving on raw TF-IDF by accounting for document length and term "
            "saturation."
        )
    }
    chunks = build_chunks(sample_docs, strategy="recursive", max_chars=200)

    bm25 = BM25Retriever(chunks)
    query = "how does BM25 improve on TF-IDF?"
    ranked = bm25.retrieve(query, top_k=3)

    print("=== Mock generation (no API key needed) ===")
    result = generate_answer(query, ranked, max_context_words=100)
    print("Prompt sent to model:\n")
    print(result.prompt)
    print("\n--- Answer ---")
    print(result.answer)

    print("\n\nOn your machine with ANTHROPIC_API_KEY set:")
    print("generate_answer(query, ranked, generate_fn=anthropic_generate_fn())")
    print("-> real generated answer grounded in the retrieved chunks.")
