# PaperMind — RAG Chatbot over ArXiv AI/ML Papers

A production-style **Retrieval-Augmented Generation** chatbot over recent ArXiv
papers (cs.AI, cs.LG, cs.CL, cs.CV). Built as a portfolio project to demonstrate
end-to-end LLM engineering: ingestion, chunking, hybrid retrieval, reranking,
grounded generation with citations, observability-friendly eval harness, and a
deployable Streamlit UI.

> **Live demo:** runs with **no API key** (local fallback). Add a free Groq key
> for high-quality answers.

## ✨ Features

- **Hybrid retrieval** — BM25 (sparse) + dense (FAISS cosine) fused with
  Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` re-scores top
  candidates for precision.
- **Grounded generation** — strict system prompt forces answers from context
  with inline `[arxiv_id]` citations; refuses when evidence is weak.
- **Provider-agnostic LLM** — Groq (free tier) → HuggingFace Inference →
  local `flan-t5-base` fallback → retrieval-only mode. Ships runnable with
  zero keys.
- **Eval harness** — 10 hand-written Q&A pairs with Hit@k, MRR, Precision@k.
  Current results: **Hit@3 = 1.0, Hit@5 = 1.0, MRR = 1.0, P@5 = 0.34**.
- **Streamlit UI** — chat, source cards, settings sidebar, example questions.
- **Reproducible** — one command (`python ingest.py`) rebuilds the whole index.

## 🏗️ Architecture

```
ArXiv API ──► papers.json ──► chunker ──► chunks.json
                                          │
                                          ▼
                          SentenceTransformer (MiniLM-L6-v2)
                                          │
                                          ▼
                          FAISS (IndexFlatIP)  +  BM25 (rank_bm25)
                                          │
User query ──► embed ──► dense top-16 ─┐
           └──► BM25 top-16 ──────────┤──► RRF fusion ──► rerank (cross-enc) ──► top-5
                                                                   │
                                                                   ▼
                                              LLM (Groq/HF/local) + citations
                                                                   │
                                                                   ▼
                                                          Streamlit chat UI
```

## 🚀 Quick Start

```bash
git clone https://github.com/<you>/papermind-rag.git
cd papermind-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add GROQ_API_KEY

# Build the index (fetches ~60 recent papers, ~1-2 min)
python ingest.py

# Run the app
streamlit run app.py
```

Optional: add a free Groq key to `.env` for high-quality answers:
```
GROQ_API_KEY=gsk_...
```

## 📊 Evaluation

```bash
python eval.py
```

```
{
  "hit@3": 1.0,
  "hit@5": 1.0,
  "mrr": 1.0,
  "p@5": 0.34
}
```

The eval set (`data/eval.json`) contains 10 hand-written questions each tied to
a specific paper, measuring whether the retriever surfaces the gold document.

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector store | FAISS (IndexFlatIP, cosine) |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq / HF Inference / local flan-t5-base |
| Data source | ArXiv API (arxiv lib) |
| Chunking | word-based, 180 words, 40 overlap |

## 📂 Project Structure

```
papermind-rag/
├── app.py          # Streamlit UI
├── ingest.py       # fetch + chunk + embed + index
├── retriever.py    # hybrid retrieval + rerank
├── llm.py          # provider-agnostic LLM
├── eval.py         # retrieval eval harness
├── config.py       # all settings
├── data/           # papers.json, chunks.json, eval.json
└── index/          # faiss.index, meta.json
```

## 🔮 Roadmap / What I'd Improve Next

- Full-text ingestion (parse PDFs) instead of abstracts only.
- Query rewriting for follow-up questions.
- LangFuse tracing for production observability.
- Docker compose + one-click deploy to HuggingFace Spaces.
- Expand corpus to 500+ papers across more categories.

## 📄 License

MIT
