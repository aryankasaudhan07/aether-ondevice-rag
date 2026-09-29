"""On-device text generation with two interchangeable backends:

- LlamaCppLLM   — a small GGUF model via llama.cpp. Fast and memory-light on CPU
                  (ideal for Mac/dev and low-RAM machines). Default when a GGUF
                  file is present.
- OnnxGenaiLLM  — a quantized model via onnxruntime-genai. This is the path that
                  targets the Snapdragon NPU (QNN) — see docs/SNAPDRAGON_PORT.md.

Both expose the same interface: stream(system, user) -> Iterator[str]. The RAG
pipeline builds a system prompt and a user message (context + question) and does
not care which backend runs. get_llm() picks the backend from what's installed.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator

from ondevice_rag import config

# Generation is serialized: running two generations at once on one model can hang
# or corrupt state. A second request waits for the lock instead.
_GEN_LOCK = threading.Lock()


class LlamaCppLLM:
    """Fast, low-memory CPU generation via a small GGUF model (llama.cpp)."""

    def __init__(self, model_path=None) -> None:
        from llama_cpp import Llama

        model_path = model_path or config.GGUF_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {model_path}. "
                "Run: uv run python scripts/download_models.py llm"
            )
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=8192,  # room for many retrieved chunks + the answer (was 4096: could overflow)
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )

    def stream(
        self, system: str, user: str, max_new_tokens: int = 512, temperature: float = 0.2
    ) -> Iterator[str]:
        with _GEN_LOCK:
            for chunk in self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,  # small models can loop; this curbs repetition
                stream=True,
            ):
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

    def generate(self, system: str, user: str, **kw) -> str:
        return "".join(self.stream(system, user, **kw))


class OnnxGenaiLLM:
    """Quantized generation via onnxruntime-genai (the Snapdragon NPU path)."""

    def __init__(self, model_dir=None) -> None:
        import onnxruntime_genai as og

        model_dir = model_dir or config.LLM_MODEL_DIR
        if not (model_dir / "genai_config.json").exists():
            raise FileNotFoundError(f"ONNX-GenAI model not found in {model_dir}.")
        self._og = og
        self.model = og.Model(str(model_dir))
        self.tokenizer = og.Tokenizer(self.model)

    def _prompt(self, system: str, user: str) -> str:
        return (
            f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
        )

    def stream(
        self, system: str, user: str, max_new_tokens: int = 512, temperature: float = 0.2
    ) -> Iterator[str]:
        og = self._og
        input_tokens = self.tokenizer.encode(self._prompt(system, user))
        params = og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=len(input_tokens) + max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )
        with _GEN_LOCK:
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            stream = self.tokenizer.create_stream()
            while not generator.is_done():
                generator.generate_next_token()
                piece = stream.decode(generator.get_next_tokens()[0])
                if piece:
                    yield piece

    def generate(self, system: str, user: str, **kw) -> str:
        return "".join(self.stream(system, user, **kw))


def get_llm():
    """Build the LLM for the user's currently-selected model (see models.py)."""
    from ondevice_rag import models

    m = models.selected()
    if m.backend == "gguf":
        return LlamaCppLLM(models.gguf_path(m))
    return OnnxGenaiLLM()
