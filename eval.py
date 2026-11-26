"""Retrieval evaluation harness: Hit@k, MRR, Precision@k."""
from __future__ import annotations

import json
from pathlib import Path

from retriever import Retriever
import config as cfg


def load_eval() -> list[dict]:
    if not cfg.EVAL_JSON.exists():
        return []
    return json.loads(cfg.EVAL_JSON.read_text())


def hit_at_k(retrieved_ids: list[str], gold: list[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    return 1.0 if top & set(gold) else 0.0


def mrr(retrieved_ids: list[str], gold: list[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_ids: list[str], gold: list[str], k: int) -> float:
    top = retrieved_ids[:k]
    return sum(1 for r in top if r in gold) / k


def run() -> dict:
    data = load_eval()
    if not data:
        print("No eval set found.")
        return {}
    r = Retriever()
    metrics = {"hit@3": [], "hit@5": [], "mrr": [], "p@5": []}
    for ex in data:
        chunks = r.retrieve(ex["question"], top_k=cfg.TOP_K, rerank=True)
        ids = [c["arxiv_id"] for c in chunks]
        gold = ex["gold_arxiv_ids"]
        metrics["hit@3"].append(hit_at_k(ids, gold, 3))
        metrics["hit@5"].append(hit_at_k(ids, gold, 5))
        metrics["mrr"].append(mrr(ids, gold))
        metrics["p@5"].append(precision_at_k(ids, gold, 5))
    result = {k: round(sum(v) / len(v), 4) for k, v in metrics.items()}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
