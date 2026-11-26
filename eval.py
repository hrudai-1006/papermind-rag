"""Retrieval evaluation harness: Hit@k, MRR, Precision@k + CRAG refusal stats."""
from __future__ import annotations

import json
import sys

from llm import LLM
from rag import RagPipeline
from retriever import Retriever
import config as cfg


def load_eval() -> list[dict]:
    if not cfg.EVAL_JSON.exists():
        return []
    return json.loads(cfg.EVAL_JSON.read_text())


def hit_at_k(retrieved_ids, gold, k):
    return 1.0 if (set(retrieved_ids[:k]) & set(gold)) else 0.0


def mrr(retrieved_ids, gold):
    for i, rid in enumerate(retrieved_ids):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_ids, gold, k):
    top = retrieved_ids[:k]
    return sum(1 for r in top if r in gold) / k


def run_retrieval() -> dict:
    data = load_eval()
    if not data:
        print("No eval set found.")
        return {}
    r = Retriever()
    metrics = {"hit@3": [], "hit@5": [], "mrr": [], "p@5": []}
    for ex in data:
        chunks, _ = r.retrieve_with_scores(ex["question"], top_k=cfg.TOP_K, rerank=True)
        ids = [c["arxiv_id"] for c in chunks]
        gold = ex["gold_arxiv_ids"]
        metrics["hit@3"].append(hit_at_k(ids, gold, 3))
        metrics["hit@5"].append(hit_at_k(ids, gold, 5))
        metrics["mrr"].append(mrr(ids, gold))
        metrics["p@5"].append(precision_at_k(ids, gold, 5))
    return {k: round(sum(v) / len(v), 4) for k, v in metrics.items()}


def run_crag() -> dict:
    data = load_eval()
    if not data:
        return {}
    pipe = RagPipeline(Retriever(), LLM())
    refused = 0
    rewritten = 0
    for ex in data:
        res = pipe.run(ex["question"])
        if res.refused:
            refused += 1
        if res.rewritten:
            rewritten += 1
    return {"n": len(data), "refused": refused, "rewritten": rewritten}


if __name__ == "__main__":
    print("== Retrieval metrics ==")
    print(json.dumps(run_retrieval(), indent=2))
    print("\n== CRAG stats (on eval set) ==")
    print(json.dumps(run_crag(), indent=2))
