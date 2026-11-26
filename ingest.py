"""Fetch ArXiv papers, chunk, embed, and build a FAISS index."""
from __future__ import annotations

import json
import re
from pathlib import Path

import arxiv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config as cfg


def fetch_papers() -> list[dict]:
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV",
        max_results=cfg.ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers = []
    for r in tqdm(client.results(search), desc="fetching arxiv"):
        papers.append({
            "id": r.entry_id,
            "arxiv_id": r.get_short_id(),
            "title": r.title.strip(),
            "abstract": r.summary.strip().replace("\n", " "),
            "authors": [a.name for a in r.authors[:6]],
            "published": r.published.strftime("%Y-%m-%d"),
            "categories": r.categories,
            "pdf_url": r.pdf_url,
        })
    return papers


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    out = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            out.append(chunk)
        if i + size >= len(words):
            break
    return out


def build_chunks(papers: list[dict]) -> list[dict]:
    # One chunk per paper. Abstracts are ~150-250 words, so splitting them
    # into overlapping windows only creates near-duplicate vectors that hurt
    # precision and inflate the index. Keeping one chunk per paper keeps each
    # paper a single retrieval unit and makes precision@k meaningful.
    chunks = []
    for p in papers:
        text = f"Title: {p['title']}\nAbstract: {p['abstract']}"
        chunks.append({
            "chunk_id": f"{p['arxiv_id']}__0",
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "text": text,
            "authors": p["authors"],
            "published": p["published"],
            "pdf_url": p["pdf_url"],
            "categories": p["categories"],
        })
    return chunks


def main() -> None:
    print("Fetching papers...")
    papers = fetch_papers()
    cfg.PAPERS_JSON.write_text(json.dumps(papers, indent=2, ensure_ascii=False))
    print(f"Saved {len(papers)} papers -> {cfg.PAPERS_JSON}")

    chunks = build_chunks(papers)
    cfg.CHUNKS_JSON.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"Built {len(chunks)} chunks -> {cfg.CHUNKS_JSON}")

    print("Embedding chunks...")
    model = SentenceTransformer(cfg.EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(cfg.FAISS_INDEX))
    cfg.META_JSON.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"FAISS index ({index.ntotal} vectors) -> {cfg.FAISS_INDEX}")


if __name__ == "__main__":
    main()
