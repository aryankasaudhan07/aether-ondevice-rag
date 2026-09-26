"""Ingest ./data, embed every chunk, and persist the vector index to ./index.

Run:  uv run python scripts/build_index.py
"""
from __future__ import annotations

import time

from ondevice_rag import config
from ondevice_rag.embeddings import Embedder
from ondevice_rag.ingest import ingest_dir
from ondevice_rag.vectorstore import VectorStore


def main() -> None:
    print(config.backend_summary())
    print(f"Ingesting documents from {config.DATA_DIR} ...")
    chunks = ingest_dir()
    if not chunks:
        print("No documents found. Drop .pdf/.txt/.md files into ./data and retry.")
        return

    print(f"Embedding {len(chunks)} chunks ...")
    embedder = Embedder()
    print(f"  active provider: {embedder.active_provider}")
    t0 = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks])
    dt = time.perf_counter() - t0
    print(f"  embedded in {dt:.2f}s ({len(chunks) / dt:.1f} chunks/s)")

    store = VectorStore()
    store.add(chunks, vectors)
    store.save()
    print(f"Saved index with {len(store)} chunks to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
