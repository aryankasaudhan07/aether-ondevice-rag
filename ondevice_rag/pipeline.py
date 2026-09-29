"""The RAG pipeline: retrieve relevant chunks, ground the LLM, answer with cites.

This ties the pieces together:
    question -> embed -> retrieve top-k chunks -> build a grounded prompt
             -> LLM generates an answer that must stick to the sources.

The prompt instructs the model to answer ONLY from the provided context and to
say so when the answer isn't there — the core discipline that makes RAG
trustworthy instead of hallucinated.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from ondevice_rag import config
from ondevice_rag.embeddings import Embedder
from ondevice_rag.llm import get_llm
from ondevice_rag.vectorstore import Hit, VectorStore


def detect_mentioned_sources(question: str, doc_names: list[str]) -> set[str] | None:
    """If the question clearly names one or more documents, return them so retrieval
    can be scoped to just those files. Returns None when nothing is clearly named.

    Matching is conservative to avoid false positives: a document counts as
    mentioned when the question contains a distinctive word from its filename
    (length >= 7, e.g. 'mailtrace', 'practice'), or at least two of its filename
    words covering half the name.
    """
    q = question.lower()
    mentioned: set[str] = set()
    for name in doc_names:
        stem = re.sub(r"\.(pdf|txt|md)$", "", name.lower())
        tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if len(t) >= 3 and t != "pdf"]
        if not tokens:
            continue
        matched = [t for t in tokens if re.search(rf"\b{re.escape(t)}\b", q)]
        distinctive = any(len(t) >= 7 for t in matched)
        if distinctive or (len(matched) >= 2 and len(matched) / len(tokens) >= 0.5):
            mentioned.add(name)
    return mentioned or None

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's documents. "
    "Use the context excerpts provided below to answer — summarize, explain, or "
    "extract from them as needed, even if the excerpts are brief notes, cue sheets, "
    "or fragments. Work with whatever relevant information is present and give a "
    "clear, useful answer. Do not invent facts that the context does not support. "
    "Only say you couldn't find it if the context is clearly unrelated to the "
    "question.\n\n"
    "Format your answer in Markdown for readability:\n"
    "- Use a Markdown table when presenting comparisons or structured data.\n"
    "- Use bullet or numbered lists for steps or multiple points.\n"
    "- Write mathematics in LaTeX: inline as $...$ and display equations as $$...$$ "
    "(e.g. $f(x,y) = 2x + 2y - 16xy$).\n"
    "Cite sources inline as [source] where helpful. Be concise."
)


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]


def _build_user_message(question: str, hits: list[Hit]) -> str:
    """The user turn: the retrieved context followed by the question."""
    context_blocks = []
    for h in hits:
        tag = f"{h.chunk.source}#{h.chunk.chunk_index}"
        context_blocks.append(f"[{tag}]\n{h.chunk.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no documents found)"
    return f"Context:\n{context}\n\nQuestion: {question}"


class RagPipeline:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store = VectorStore.load()
        self.llm = get_llm()

    def retrieve(
        self, question: str, top_k: int = config.TOP_K, sources: set[str] | None = None
    ) -> list[Hit]:
        return self.store.search(
            self.embedder.encode_one(question), top_k=top_k, allowed_sources=sources
        )

    def answer(
        self, question: str, top_k: int = config.TOP_K, sources: set[str] | None = None
    ) -> Answer:
        hits = self.retrieve(question, top_k, sources)
        text = self.llm.generate(SYSTEM_PROMPT, _build_user_message(question, hits))
        return Answer(question=question, text=text.strip(), hits=hits)

    def stream_answer(
        self, question: str, top_k: int = config.TOP_K, sources: set[str] | None = None
    ) -> tuple[list[Hit], Iterator[str]]:
        """Return the sources immediately and a token stream for the answer."""
        hits = self.retrieve(question, top_k, sources)
        return hits, self.llm.stream(SYSTEM_PROMPT, _build_user_message(question, hits))
