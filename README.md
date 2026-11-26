# PaperMind

A RAG chatbot over recent ArXiv AI/ML papers (cs.AI, cs.LG, cs.CL, cs.CV). Built to show end-to-end LLM engineering: ingestion, chunking, hybrid retrieval, reranking, grounded generation with citations, a corrective retrieval loop, and an eval harness.

Live demo: https://papermind-rag-v8u2klen3mruvd9kbarwvh.streamlit.app/

## What it does

Ask a question and PaperMind retrieves relevant chunks from ~60 recent papers, reranks them, and answers using only that context, citing the arXiv IDs it used. If nothing relevant turns up, it says so instead of making something up. Out-of-domain questions get refused rather than hallucinated.

## Features

- Hybrid retrieval: BM25 (sparse) plus dense FAISS (cosine) fused with Reciprocal Rank Fusion.
- Cross-encoder reranking with ms-marco-MiniLM-L-6-v2 to re-score the top candidates.
- Grounded generation: a strict system prompt forces answers from the retrieved context with inline [arxiv_id] citations.
- Corrective RAG (CRAG): grades retrieved chunks by relevance; if the top score is too low, rewrites the query and re-retrieves once; if still low, refuses. Follow-up questions get rewritten into standalone search queries using conversation history.
- Provider-agnostic LLM: tries Groq, then HuggingFace Inference, then a local flan-t5-base model, then retrieval-only mode. Ships runnable with zero API keys.
- Eval harness: 10 hand-written Q&A pairs with Hit@k, MRR, and Precision@k.
- Streamlit UI with chat, source cards, a settings sidebar, and example questions.
- Reproducible: one command (python ingest.py) rebuilds the whole index.

## Architecture

```
ArXiv API -> papers.json -> chunker -> chunks.json
                                    |
                                    v
                  SentenceTransformer (MiniLM-L6-v2)
                                    |
                                    v
                  FAISS (IndexFlatIP)  +  BM25 (rank_bm25)
                                    |
User query -> embed -> dense top-16 --+
          \-> BM25 top-16 ----------+-> RRF fusion -> rerank (cross-enc) -> top-5
                                                              |
                                                              v
                                           LLM (Groq/HF/local) + citations
                                                              |
                                                              v
                                                     Streamlit chat UI
```

## Quick start

```
git clone https://github.com/hrudai-1006/papermind-rag.git
cd papermind-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add GROQ_API_KEY

# Build the index (fetches ~60 recent papers, takes 1-2 minutes)
python ingest.py

# Run the app
streamlit run app.py
```

Optional: add a free Groq key to .env for higher-quality generated answers:

```
GROQ_API_KEY=gsk_...
```

## Evaluation

```
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

The eval set (data/eval.json) has 10 hand-written questions, each tied to a specific paper, measuring whether the retriever surfaces the gold document. CRAG refused 0 of 10 in-domain questions and 2 of 2 tested out-of-domain questions, with no hallucinations.

## Tech stack

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

## Project structure

```
papermind-rag/
├── app.py          # Streamlit UI
├── ingest.py       # fetch + chunk + embed + index
├── retriever.py    # hybrid retrieval + rerank
├── llm.py          # provider-agnostic LLM
├── rag.py          # CRAG pipeline
├── eval.py         # retrieval eval harness
├── config.py       # all settings
├── data/           # papers.json, chunks.json, eval.json
├── index/          # faiss.index, meta.json
└── assets/         # screenshot
```

## Deploy

Already deployed at https://papermind-rag-v8u2klen3mruvd9kbarwvh.streamlit.app/

To deploy your own fork:

- Streamlit Cloud (free): fork this repo, go to share.streamlit.io, create a new app pointing at app.py. No secrets needed; it runs in retrieval-only mode. Add GROQ_API_KEY in the app's secrets for generated answers.
- HuggingFace Spaces (free, Docker): create a Space with the Streamlit SDK, push this repo, and add GROQ_API_KEY as a Space secret.

## What I would improve next

- Full-text ingestion by parsing PDFs instead of using abstracts only.
- Query rewriting for follow-up questions (partially done; could be stronger).
- LangFuse tracing for production observability.
- Docker compose and one-click deploy to HuggingFace Spaces.
- Expand the corpus to 500+ papers across more categories.

## License

MIT
