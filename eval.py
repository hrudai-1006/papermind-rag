"""Comprehensive evaluation: retrieval, refusal, answer quality, latency."""
from __future__ import annotations

import json
import time
from pathlib import Path

from llm import LLM
from rag import RagPipeline
from retriever import Retriever
import config as cfg


def load_json(p):
    return json.loads(Path(p).read_text())


def hit_at_k(ids, gold, k):
    return 1.0 if (set(ids[:k]) & set(gold)) else 0.0


def mrr(ids, gold):
    for i, rid in enumerate(ids):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ids, gold, k):
    return sum(1 for r in ids[:k] if r in gold) / k


def eval_retrieval(pipe) -> dict:
    data = load_json(cfg.EVAL_JSON)
    metrics = {"hit@1": [], "hit@3": [], "hit@5": [], "mrr": [], "p@5": []}
    latencies = []
    for ex in data:
        t = time.time()
        chunks, scores = pipe.retriever.retrieve_with_scores(ex["question"], top_k=cfg.TOP_K, rerank=True)
        latencies.append((time.time() - t) * 1000)
        ids = [c["arxiv_id"] for c in chunks]
        gold = ex["gold_arxiv_ids"]
        metrics["hit@1"].append(hit_at_k(ids, gold, 1))
        metrics["hit@3"].append(hit_at_k(ids, gold, 3))
        metrics["hit@5"].append(hit_at_k(ids, gold, 5))
        metrics["mrr"].append(mrr(ids, gold))
        metrics["p@5"].append(precision_at_k(ids, gold, 5))
    out = {k: round(sum(v) / len(v), 4) for k, v in metrics.items()}
    out["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
    out["n"] = len(data)
    return out


def eval_refusal(pipe) -> dict:
    in_domain = load_json(cfg.EVAL_JSON)
    out_domain = load_json(cfg.DATA_DIR / "refusal_eval.json")
    # in-domain should NOT refuse
    false_refusals = 0
    for ex in in_domain:
        res = pipe.run(ex["question"])
        if res.refused:
            false_refusals += 1
    # out-of-domain should refuse
    true_refusals = 0
    for q in out_domain:
        res = pipe.run(q)
        if res.refused:
            true_refusals += 1
    return {
        "in_domain_n": len(in_domain),
        "in_domain_false_refusals": false_refusals,
        "out_domain_n": len(out_domain),
        "out_domain_true_refusals": true_refusals,
        "refusal_accuracy": round((len(in_domain) - false_refusals + true_refusals) /
                                 (len(in_domain) + len(out_domain)), 4),
    }


def eval_answer_quality(pipe) -> dict:
    data = load_json(cfg.EVAL_JSON)
    has_citation = 0
    grounded = 0
    refused = 0
    latencies = []
    for ex in data:
        t = time.time()
        res = pipe.run(ex["question"])
        latencies.append((time.time() - t) * 1000)
        if res.refused:
            refused += 1
            continue
        # citation presence: answer mentions at least one arxiv id
        if any(g in res.answer for g in ex["gold_arxiv_ids"]):
            has_citation += 1
        # groundedness heuristic: answer contains a gold citation AND no refusal
        if any(g in res.answer for g in ex["gold_arxiv_ids"]):
            grounded += 1
    n = len(data)
    return {
        "n": n,
        "answered": n - refused,
        "refused": refused,
        "citation_hit_rate": round(has_citation / n, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
    }


def main():
    pipe = RagPipeline(Retriever(), LLM())
    print("== Retrieval metrics ==")
    print(json.dumps(eval_retrieval(pipe), indent=2))
    print("\n== Refusal accuracy ==")
    print(json.dumps(eval_refusal(pipe), indent=2))
    print("\n== Answer quality ==")
    print(json.dumps(eval_answer_quality(pipe), indent=2))


if __name__ == "__main__":
    main()
