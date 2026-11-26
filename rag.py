"""CRAG (Corrective RAG) pipeline: rewrite -> retrieve -> grade -> (rewrite+retry) -> generate/refuse."""
from __future__ import annotations

from dataclasses import dataclass, field

import config as cfg
from llm import LLM
from retriever import Retriever


REFUSAL = ("I don't have enough evidence in the indexed papers to answer that "
           "confidently. Try rephrasing, or ask about a topic covered by the "
           "corpus (recent ArXiv AI/ML papers).")


@dataclass
class RagResult:
    answer: str
    chunks: list[dict]
    scores: list[float]
    query_used: str
    original_query: str
    rewritten: bool = False
    refused: bool = False
    meta: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLM) -> None:
        self.retriever = retriever
        self.llm = llm

    def _rewrite(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        # Build a compact conversation snippet
        turns = []
        for m in history[-4:]:
            turns.append(f"{m['role']}: {m['content']}")
        convo = "\n".join(turns)
        prompt = (f"Given this conversation:\n{convo}\n\n"
                  f"Rewrite the follow-up question into a standalone search query "
                  f"(no preamble, just the query):\n{question}")
        # Use LLM generate with no chunks (retrieval-only fallback returns text)
        try:
            out = self.llm.generate(prompt, [])
            # clean
            out = out.strip().split("\n")[0].strip().strip('"')
            return out if out else question
        except Exception:
            return question

    def run(self, question: str, history: list[dict] | None = None,
            top_k: int = cfg.TOP_K, rerank: bool = True) -> RagResult:
        history = history or []
        original = question
        rewritten = False

        # 1. query rewriting for follow-ups
        query = question
        if history:
            query = self._rewrite(question, history)
            rewritten = (query != question)

        # 2. retrieve + grade
        chunks, scores = self.retriever.retrieve_with_scores(query, top_k=top_k, rerank=rerank)
        relevant = self.retriever.is_relevant(scores, cfg.RELEVANCE_THRESHOLD)

        # 3. CRAG: if not relevant, rewrite query and retry once
        if not relevant and not rewritten:
            # try a keyword-focused rewrite via LLM
            alt = self._rewrite(question + " (rephrase for keyword search)", [])
            if alt and alt != query:
                chunks2, scores2 = self.retriever.retrieve_with_scores(alt, top_k=top_k, rerank=rerank)
                if self.retriever.is_relevant(scores2, cfg.RELEVANCE_THRESHOLD):
                    chunks, scores = chunks2, scores2
                    query = alt
                    rewritten = True
                    relevant = True

        # 4. refuse if still not relevant
        if not relevant:
            return RagResult(answer=REFUSAL, chunks=chunks, scores=scores,
                            query_used=query, original_query=original,
                            rewritten=rewritten, refused=True)

        # 5. generate
        answer = self.llm.generate(question, chunks)
        return RagResult(answer=answer, chunks=chunks, scores=scores,
                        query_used=query, original_query=original,
                        rewritten=rewritten, refused=False)
