"""PaperMind — Streamlit RAG chatbot over recent ArXiv AI/ML papers."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import config as cfg
from llm import LLM
from retriever import Retriever


st.set_page_config(page_title="PaperMind — ArXiv RAG", page_icon="📄", layout="wide")

st.title("📄 PaperMind")
st.caption("RAG chatbot over recent ArXiv AI/ML papers (cs.AI, cs.LG, cs.CL, cs.CV). "
           "Hybrid retrieval (BM25 + dense) with cross-encoder reranking and citations.")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource
def get_llm() -> LLM:
    return LLM()


retriever = get_retriever()
llm = get_llm()

with st.sidebar:
    st.subheader("Settings")
    st.write(f"LLM mode: **{llm.mode}**")
    st.write(f"Index size: **{len(retriever.chunks)} chunks**")
    top_k = st.slider("Top-K retrieval", 4, 16, cfg.TOP_K)
    rerank = st.toggle("Cross-encoder rerank", value=True)
    show_sources = st.toggle("Show sources", value=True)
    st.divider()
    st.markdown("**Example questions**")
    examples = [
        "What is retrieval augmented generation?",
        "How do vision-language models work?",
        "What are recent advances in efficient LLM training?",
        "Explain diffusion models for image generation.",
        "What is chain-of-thought prompting?",
    ]
    for e in examples:
        if st.button(e, use_container_width=True):
            st.session_state["prefill"] = e

if "messages" not in st.session_state:
    st.session_state.messages = []

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources") and show_sources:
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"- [{s['arxiv_id']}] [{s['title']}]({s['pdf_url']})")

prefill = st.session_state.pop("prefill", None)
prompt = st.chat_input("Ask about recent AI/ML papers...") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            chunks = retriever.retrieve(prompt, top_k=top_k, rerank=rerank)
            answer = llm.generate(prompt, chunks)
        st.markdown(answer)
        sources = [
            {"arxiv_id": c["arxiv_id"], "title": c["title"], "pdf_url": c["pdf_url"]}
            for c in chunks
        ]
        if show_sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- [{s['arxiv_id']}] [{s['title']}]({s['pdf_url']})")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
