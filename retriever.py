"""Hybrid retrieval: BM25 + dense (FAISS) fused with RRF, then cross-encoder rerank."""
from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import config as cfg


class Retriever:
    def __init__(self) -> None:
        self.chunks: list[dict] = json.loads(cfg.META_JSON.read_text())
        self.index = faiss.read_index(str(cfg.FAISS_INDEX))
        self.embedder = SentenceTransformer(cfg.EMBED_MODEL)
        self.reranker = None  # lazy
        # BM25 corpus
        tokenized = [self._tok(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idx = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def _sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(self._tok(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]

    @staticmethod
    def _rrf(results_list: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
        agg: dict[int, float] = {}
        for res in results_list:
            for rank, (idx, _) in enumerate(res):
                agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(agg.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = cfg.TOP_K, rerank: bool = True) -> list[dict]:
        dense = self._dense(query, top_k * 2)
        sparse = self._sparse(query, top_k * 2)
        fused = self._rrf([dense, sparse])[: top_k * 2]

        cands = [self.chunks[i] for i, _ in fused]
        if rerank and cands:
            if self.reranker is None:
                self.reranker = CrossEncoder(cfg.RERANK_MODEL)
            pairs = [(query, c["text"]) for c in cands]
            scores = self.reranker.predict(pairs)
            order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
            cands = [cands[i] for i in order[:cfg.RERANK_TOP_N]]
        else:
            cands = cands[:cfg.RERANK_TOP_N]
        return cands


if __name__ == "__main__":
    r = Retriever()
    for c in r.retrieve("What is retrieval augmented generation?"):
        print(c["arxiv_id"], "|", c["title"][:60])
