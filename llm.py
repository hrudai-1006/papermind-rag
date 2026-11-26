"""LLM provider with graceful fallback: Groq -> HuggingFace Inference -> local flan-t5."""
from __future__ import annotations

import os
from typing import Iterator

import config as cfg


SYSTEM_PROMPT = """You are PaperMind, a research assistant over recent ArXiv AI/ML papers.
Answer the user's question using ONLY the provided context passages. Cite sources inline as [arxiv_id].
If the context does not contain the answer, say you don't know and suggest a better query.
Be concise, accurate, and never fabricate citations."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[{c['arxiv_id']}] {c['title']}: {c['text']}" for c in chunks
    )
    return ("Answer the question using only the context below. "
            "Include the arxiv id of the source in brackets.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")


class LLM:
    def __init__(self) -> None:
        self.mode = "none"
        self._client = None
        self._local = None
        if cfg.GROQ_API_KEY:
            try:
                from groq import Groq
                self._client = Groq(api_key=cfg.GROQ_API_KEY)
                self.mode = "groq"
                return
            except Exception:
                pass
        if cfg.HF_TOKEN:
            self.mode = "hf"
            return
        # local fallback
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(cfg.LOCAL_LLM)
            self._local = AutoModelForSeq2SeqLM.from_pretrained(cfg.LOCAL_LLM)
            self.mode = "local"
        except Exception:
            self.mode = "retrieval-only"

    def _ensure_citations(self, answer: str, chunks: list[dict]) -> str:
        ids = [c["arxiv_id"] for c in chunks]
        if any(i in answer for i in ids):
            return answer
        # append a sources line so citations are always present
        src = ", ".join(f"[{i}]" for i in ids[:3])
        return answer.rstrip() + f"\n\nSources: {src}"

    def generate(self, question: str, chunks: list[dict]) -> str:
        prompt = build_prompt(question, chunks)
        if self.mode == "groq":
            resp = self._client.chat.completions.create(
                model=cfg.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            return self._ensure_citations(resp.choices[0].message.content.strip(), chunks)
        if self.mode == "hf":
            import requests
            url = f"https://api-inference.huggingface.co/models/{cfg.HF_MODEL}"
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {cfg.HF_TOKEN}"},
                json={"inputs": SYSTEM_PROMPT + "\n" + prompt, "parameters": {"max_new_tokens": 400}},
                timeout=60,
            )
            data = r.json()
            if isinstance(data, list) and data:
                return self._ensure_citations(data[0].get("generated_text", "").strip(), chunks)
            return str(data)
        if self.mode == "local":
            # flan-t5 works best with a short, focused prompt on the top chunk
            top = chunks[:1]
            ctx = " ".join(f"{c['title']}. {c['text']}" for c in top)
            flan_prompt = f"answer the question based on the context. question: {question} context: {ctx}"
            inputs = self._tok(flan_prompt, return_tensors="pt", truncation=True, max_length=512)
            out_ids = self._local.generate(**inputs, max_new_tokens=128, do_sample=False,
                                          num_beams=2)
            ans = self._tok.decode(out_ids[0], skip_special_tokens=True).strip()
            if not ans or len(ans) < 8 or ans.lower().startswith("title:"):
                # fall back to an extractive snippet from the top chunk
                snippet = top[0]["text"] if top else ""
                # strip the "Title: ... Abstract:" prefix for readability
                if "Abstract:" in snippet:
                    snippet = snippet.split("Abstract:", 1)[-1].strip()
                sentences = [x.strip() for x in snippet.replace("\n", " ").split(".") if x.strip()]
                ans = ". ".join(sentences[:2]) + "." if sentences else snippet[:280]
            return self._ensure_citations(ans, chunks)
        # retrieval-only: return a stitched answer from top chunks
        lines = ["(No LLM API key configured — showing retrieved evidence.)\n"]
        for c in chunks[:3]:
            lines.append(f"- [{c['arxiv_id']}] {c['title']}: {c['text'][:280]}...")
        return "\n".join(lines)

    def stream(self, question: str, chunks: list[dict]) -> Iterator[str]:
        # Stream only supported for groq; otherwise yield whole
        text = self.generate(question, chunks)
        # fake stream by words for UI
        for w in text.split(" "):
            yield w + " "
