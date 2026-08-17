# Retrieval Benchmark + RAG Chat System

A from-scratch benchmark comparing sparse, dense, and hybrid retrieval strategies for Retrieval-Augmented Generation, evaluated with [RAGAS](https://github.com/explodinggradients/ragas) and wrapped into a working RAG chat interface.

Built to understand *why* each retrieval design choice matters, not just to call a framework — every component (chunking, retrieval, reranking, generation, evaluation) is implemented from first principles with no black-box wrappers.

---

## Why this exists

Most RAG tutorials wire together a vector DB and an LLM and call it done. This project instead asks: **which retrieval strategy actually retrieves the right context, and by how much?**

Five retrieval pipelines are implemented and benchmarked head-to-head on the same corpus and the same evaluation questions:

| Pipeline | Retrieval strategy |
|---|---|
| `tfidf` | Sparse — TF-IDF + cosine similarity |
| `bm25` | Sparse — BM25 (term saturation + length normalization) |
| `dense` | Dense — sentence embeddings + FAISS nearest-neighbor search |
| `hybrid` | Sparse + dense fused via Reciprocal Rank Fusion (RRF) |
| `hybrid+rerank` | Hybrid retrieval, top candidates re-scored by a cross-encoder |

Each is scored on **faithfulness**, **answer relevancy**, **context precision**, and **context recall** using RAGAS's LLM-judged metrics.

---

## Architecture

```
                        ┌─────────────────┐
   raw documents  ───▶  │    Chunking      │  fixed / sentence / recursive / semantic
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │    Retrieval      │  TF-IDF · BM25 · Dense (FAISS) · Hybrid (RRF)
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │   Reranking       │  Cross-encoder re-scoring of top candidates
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │   Generation      │  Context budgeting → prompt assembly → LLM
                        └────────┬─────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 ▼                                 ▼
        ┌─────────────────┐              ┌──────────────────┐
        │  RAGAS Evaluation │              │   Chat Interface  │
        │  (per pipeline)    │              │   (best pipeline)  │
        └─────────────────┘              └──────────────────┘
```

Every stage is built as an injectable function/interface (`embed_fn`, `score_fn`, `generate_fn`), so swapping an embedding provider, LLM, or reranker never requires touching the orchestration logic.

---

## Project structure

```
retrieval_benchmark/
├── chunking.py                   # Fixed-size, sentence-aware, and recursive chunking
├── semantic_chunker.py           # Embedding-based chunking (splits on topic shifts)
├── generator.py                  # Context budgeting, prompt assembly, LLM generation
├── evaluation.py                 # RAGAS evaluation harness (faithfulness, relevancy, precision, recall)
├── benchmark.py                  # Runs all pipelines across an eval set, produces comparison table
├── chat.py                       # Interactive CLI chat over the best-performing pipeline
└── retrievers/
    ├── tfidf_retriever.py        # Sparse retrieval — TF-IDF + cosine similarity
    ├── bm25_retriever.py         # Sparse retrieval — BM25
    ├── dense_retriever.py        # Dense retrieval — sentence embeddings + FAISS
    ├── hybrid_retriever.py       # Sparse + dense fusion via Reciprocal Rank Fusion
    └── reranker.py                # Cross-encoder reranking of retrieved candidates
```

---

## Setup

```bash
git clone <your-repo-url>
cd retrieval_benchmark
pip install sentence-transformers faiss-cpu rank_bm25 scikit-learn nltk ragas langchain-anthropic anthropic pandas
export ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

**Run a single retriever:**
```bash
python retrievers/bm25_retriever.py
```

**Compare all 5 pipelines on an eval set:**
```python
from benchmark import make_pipeline, run_benchmark, summarize, EvalQuestion

pipelines = {
    "tfidf": make_pipeline(tfidf.retrieve),
    "bm25": make_pipeline(bm25.retrieve),
    "dense": make_pipeline(dense.retrieve),
    "hybrid": make_pipeline(hybrid.retrieve),
    "hybrid+rerank": make_pipeline(hybrid.retrieve, rerank_fn=reranker.rerank),
}

questions = [
    EvalQuestion(question="...", ground_truth="..."),
]

results = run_benchmark(pipelines, questions, use_real_ragas=True)
print(summarize(results))
```

**Chat with your corpus:**
```python
from chat import chat_loop
chat_loop(your_docs, use_real_models=True)
```

---

## Results

*Populate this section after running `benchmark.py` on your own corpus and eval set — this is where the actual comparison numbers go.*

| Pipeline | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| TF-IDF | — | — | — | — |
| BM25 | — | — | — | — |
| Dense | — | — | — | — |
| Hybrid | — | — | — | — |
| Hybrid + Rerank | — | — | — | — |

---

## Design notes

- **Every embedding-dependent component is injectable** (`embed_fn`, `score_fn`, `generate_fn`) — production models and offline mock stand-ins share the same interface, so the pipeline logic never needs to change to swap providers.
- **Hybrid retrieval uses Reciprocal Rank Fusion**, not raw score averaging — BM25 and cosine-similarity scores live on incomparable scales, so rank-based fusion avoids one retriever silently dominating the other.
- **Context is budgeted before prompting**, not just top-K truncated — chunks are greedily selected up to a word/token budget in rank order, which directly affects RAGAS faithfulness scores.
- **Semantic chunking uses percentile-based breakpoint detection**, not a fixed similarity threshold, so it adapts to each document instead of needing per-corpus tuning.

---

## Roadmap / Extensions

- [ ] Query rewriting for multi-turn follow-up questions
- [ ] Approximate nearest-neighbor indexing (HNSW/IVF) for corpus-scale search
- [ ] Streamlit/web UI over `chat.py`
- [ ] Additional rerankers (e.g. ColBERT-style late interaction)

---

## Author

Lalith — B.Tech CSE, IIIT Naya Raipur
