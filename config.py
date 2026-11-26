"""Shared configuration for PaperMind (ArXiv RAG chatbot)."""
from __future__ import annotations

import torch
torch.set_num_threads(1)

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
ARXIV_MAX_RESULTS = 60
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_LLM = "google/flan-t5-base"

DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
PAPERS_JSON = DATA_DIR / "papers.json"
CHUNKS_JSON = DATA_DIR / "chunks.json"
FAISS_INDEX = INDEX_DIR / "faiss.index"
META_JSON = INDEX_DIR / "meta.json"
EVAL_JSON = DATA_DIR / "eval.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

TOP_K = 8
RELEVANCE_THRESHOLD = 0.0   # cross-encoder logit; above = relevant (fast path)
AMBIGUOUS_LOW = -2.0        # below this -> refuse without LLM grading
AMBIGUOUS_HIGH = 2.0       # above this -> accept without LLM grading
RERANK_TOP_N = 5

for d in (DATA_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)
