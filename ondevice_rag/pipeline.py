"""The RAG pipeline: retrieve relevant chunks, ground the LLM, answer with cites.

This ties the pieces together:
    question -> embed -> retrieve top-k chunks -> build a grounded prompt
             -> LLM generates an answer that must stick to the sources.

The prompt instructs the model to answer ONLY from the provided context and to
say so when the answer isn't there — the core discipline that makes RAG
trustworthy instead of hallucinated.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ondevice_rag import config
from ondevice_rag.embeddings import Embedder
from ondevice_rag.llm import LLM
from ondevice_rag.vectorstore import Hit, VectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided below. If the answer is not contained in the context, say you don't "
    "know based on the documents. Cite sources inline as [source]. Be concise."
)


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]


def _build_prompt(question: str, hits: list[Hit]) -> str:
    context_blocks = []
    for h in hits:
        tag = f"{h.chunk.source}#{h.chunk.chunk_index}"
        context_blocks.append(f"[{tag}]\n{h.chunk.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no documents found)"

    # Phi-3 chat format.
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )


class RagPipeline:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store = VectorStore.load()
        self.llm = LLM()

    def retrieve(self, question: str, top_k: int = config.TOP_K) -> list[Hit]:
        return self.store.search(self.embedder.encode_one(question), top_k=top_k)

    def answer(self, question: str, top_k: int = config.TOP_K) -> Answer:
        hits = self.retrieve(question, top_k)
        text = self.llm.generate(_build_prompt(question, hits))
        return Answer(question=question, text=text.strip(), hits=hits)

    def stream_answer(
        self, question: str, top_k: int = config.TOP_K
    ) -> tuple[list[Hit], Iterator[str]]:
        """Return the sources immediately and a token stream for the answer."""
        hits = self.retrieve(question, top_k)
        return hits, self.llm.stream(_build_prompt(question, hits))
