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

    def retrieve_and_decide(self, question: str, history: list[dict] | None = None,
                           top_k: int = cfg.TOP_K, rerank: bool = True):
        """Run retrieval + CRAG grading without generating. Returns (chunks, scores, query_used, rewritten, refused)."""
        history = history or []
        original = question
        rewritten = False
        query = question
        if history:
            query = self._rewrite(question, history)
            rewritten = (query != question)

        chunks, scores = self.retriever.retrieve_with_scores(query, top_k=top_k, rerank=rerank)
        max_score = max(scores) if scores else float("-inf")

        def llm_grade(chs):
            if not chs:
                return False
            ctx = chs[0]["text"][:600]
            prompt = ("Is the following passage relevant to answering the question? "
                      "Answer only 'yes' or 'no'.\n\nQuestion: " + question +
                      "\n\nPassage: " + ctx)
            try:
                out = self.llm.generate(prompt, []).lower()
                return out.strip().startswith("yes")
            except Exception:
                return max_score >= cfg.RELEVANCE_THRESHOLD

        if max_score >= cfg.AMBIGUOUS_HIGH:
            relevant = True
        elif max_score < cfg.AMBIGUOUS_LOW:
            relevant = False
        else:
            relevant = llm_grade(chunks)

        if not relevant and not rewritten:
            alt = self._rewrite(question + " (rephrase for keyword search)", [])
            if alt and alt != query:
                chunks2, scores2 = self.retriever.retrieve_with_scores(alt, top_k=top_k, rerank=rerank)
                ms2 = max(scores2) if scores2 else float("-inf")
                if ms2 >= cfg.AMBIGUOUS_HIGH or (cfg.AMBIGUOUS_LOW <= ms2 < cfg.AMBIGUOUS_HIGH and llm_grade(chunks2)):
                    chunks, scores, query = chunks2, scores2, alt
                    rewritten, relevant = True, True
                elif ms2 >= cfg.RELEVANCE_THRESHOLD:
                    chunks, scores, query = chunks2, scores2, alt
                    rewritten, relevant = True, True

        refused = not relevant
        return chunks, scores, query, rewritten, refused

    def run(self, question: str, history: list[dict] | None = None,
            top_k: int = cfg.TOP_K, rerank: bool = True) -> RagResult:
        history = history or []
        chunks, scores, query, rewritten, refused = self.retrieve_and_decide(
            question, history=history, top_k=top_k, rerank=rerank)

        if refused:
            return RagResult(answer=REFUSAL, chunks=chunks, scores=scores,
                            query_used=query, original_query=question,
                            rewritten=rewritten, refused=True)

        answer = self.llm.generate(question, chunks)
        return RagResult(answer=answer, chunks=chunks, scores=scores,
                        query_used=query, original_query=question,
                        rewritten=rewritten, refused=False)
