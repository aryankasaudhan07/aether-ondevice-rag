"""Sentence embeddings via ONNX Runtime.

Runs the all-MiniLM-L6-v2 embedding model directly through ONNX Runtime — no
PyTorch at runtime — so the exact same code path runs on your Mac (CPU) and on
the Snapdragon NPU (QNN). The execution provider comes from config.ort_providers(),
which is the single switch between the two.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ondevice_rag import config


class Embedder:
    def __init__(self, model_dir: Path | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = model_dir or config.EMBED_MODEL_DIR
        model_path = model_dir / "model.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"embedding model not found in {model_dir}. "
                "Run: uv run python scripts/download_models.py"
            )

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=config.EMBED_MAX_TOKENS)
        self.tokenizer.enable_padding()

        providers = config.ort_providers()
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        # The provider ORT actually selected (QNN may fall back to CPU).
        self.active_provider = self.session.get_providers()[0]
        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encs = self.tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in encs], dtype=np.int64)
            attention = np.array([e.attention_mask for e in encs], dtype=np.int64)

            feeds = {"input_ids": input_ids, "attention_mask": attention}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            token_embeddings = self.session.run(None, feeds)[0]  # (B, T, H)
            emb = _mean_pool(token_embeddings, attention)
            out.append(emb)

        embeddings = np.vstack(out).astype(np.float32)
        return _l2_normalize(embeddings)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)  # (B, T, 1)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=1e-12, a_max=None)
