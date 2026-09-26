"""A small, dependency-free vector index.

Embeddings are L2-normalized, so cosine similarity is a plain dot product and the
whole search is one matrix multiply in NumPy. For demo-scale corpora (thousands
of chunks) this is instant and keeps the project portable — no native vector-DB
build to fight with on Windows-on-Arm. The interface is deliberately swappable if
you later want FAISS/hnswlib.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ondevice_rag import config
from ondevice_rag.ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None  # (N, dim), normalized
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors count mismatch")
        self._chunks.extend(chunks)
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )

    def search(self, query_vec: np.ndarray, top_k: int = config.TOP_K) -> list[Hit]:
        if self._vectors is None or not self._chunks:
            return []
        scores = self._vectors @ query_vec  # cosine (vectors are normalized)
        k = min(top_k, len(self._chunks))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [Hit(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._chunks)

    # --- persistence ------------------------------------------------------
    def save(self, index_dir: Path | None = None) -> None:
        index_dir = index_dir or config.INDEX_DIR
        index_dir.mkdir(parents=True, exist_ok=True)
        vec_path = index_dir / "vectors.npy"
        if self._vectors is not None:
            np.save(vec_path, self._vectors)
        elif vec_path.exists():
            vec_path.unlink()  # clear stale vectors when the index is emptied
        meta = [
            {"text": c.text, "source": c.source, "chunk_index": c.chunk_index}
            for c in self._chunks
        ]
        (index_dir / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "VectorStore":
        index_dir = index_dir or config.INDEX_DIR
        store = cls()
        vec_path = index_dir / "vectors.npy"
        meta_path = index_dir / "chunks.json"
        if not vec_path.exists() or not meta_path.exists():
            return store  # empty store
        store._vectors = np.load(vec_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store._chunks = [
            Chunk(text=m["text"], source=m["source"], chunk_index=m["chunk_index"])
            for m in meta
        ]
        return store
