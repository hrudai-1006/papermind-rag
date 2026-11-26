"""PaperMind — Streamlit RAG chatbot over recent ArXiv AI/ML papers (with CRAG)."""
from __future__ import annotations

import streamlit as st

import config as cfg
from llm import LLM
from rag import RagPipeline
from retriever import Retriever


st.set_page_config(page_title="PaperMind — ArXiv RAG", page_icon="📄", layout="wide")

st.title("📄 PaperMind")
st.caption("CRAG chatbot over recent ArXiv AI/ML papers (cs.AI, cs.LG, cs.CL, cs.CV). "
           "Hybrid retrieval (BM25 + dense) + cross-encoder rerank + corrective "
           "retrieval + grounded citations.")


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline(Retriever(), LLM())


pipeline = get_pipeline()
llm = pipeline.llm

with st.sidebar:
    st.subheader("Settings")
    st.write(f"LLM mode: **{llm.mode}**")
    st.write(f"Index size: **{len(pipeline.retriever.chunks)} chunks**")
    top_k = st.slider("Top-K retrieval", 4, 16, cfg.TOP_K)
    rerank = st.toggle("Cross-encoder rerank", value=True)
    show_sources = st.toggle("Show sources", value=True)
    show_debug = st.toggle("Show retrieval debug", value=False)
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

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources") and show_sources:
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"- [{s['arxiv_id']}] [{s['title']}]({s['pdf_url']})")
        if m.get("debug") and show_debug:
            st.code(m["debug"], language="text")

prefill = st.session_state.pop("prefill", None)
prompt = st.chat_input("Ask about recent AI/ML papers...") or prefill

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + (corrective) generating..."):
            res = pipeline.run(prompt, history=history, top_k=top_k, rerank=rerank)
        st.markdown(res.answer)
        sources = [
            {"arxiv_id": c["arxiv_id"], "title": c["title"], "pdf_url": c["pdf_url"]}
            for c in res.chunks
        ]
        if show_sources and not res.refused:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- [{s['arxiv_id']}] [{s['title']}]({s['pdf_url']})")
        debug = (f"query_used: {res.query_used}\n"
                 f"rewritten: {res.rewritten}\n"
                 f"refused: {res.refused}\n"
                 f"scores: {[round(s,3) for s in res.scores]}")
        if show_debug:
            st.code(debug, language="text")
    st.session_state.messages.append({
        "role": "assistant", "content": res.answer, "sources": sources, "debug": debug,
    })
