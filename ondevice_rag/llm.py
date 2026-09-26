"""On-device text generation via onnxruntime-genai.

Wraps a quantized instruct model (Phi-3.5-mini) that runs locally. The same code
runs on your Mac (CPU build of onnxruntime-genai) and on the Snapdragon NPU (QNN
build), because the model directory's genai_config.json carries the execution
target — nothing here hardcodes a backend.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

from ondevice_rag import config

# onnxruntime-genai is not safe to run two generations at once on one model.
# Serialize generation so a second request waits instead of corrupting/hanging.
_GEN_LOCK = threading.Lock()


class LLM:
    def __init__(self, model_dir: Path | None = None) -> None:
        import onnxruntime_genai as og

        model_dir = model_dir or config.LLM_MODEL_DIR
        if not (model_dir / "genai_config.json").exists():
            raise FileNotFoundError(
                f"LLM not found in {model_dir}. "
                "Run: uv run python scripts/download_models.py llm"
            )
        self._og = og
        self.model = og.Model(str(model_dir))
        self.tokenizer = og.Tokenizer(self.model)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        return "".join(self.stream(prompt, max_new_tokens, temperature))

    def stream(
        self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2
    ) -> Iterator[str]:
        """Yield generated text token-by-token (good for a live UI)."""
        og = self._og
        input_tokens = self.tokenizer.encode(prompt)

        params = og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=len(input_tokens) + max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )
        # Hold the lock for the whole generation. If the consumer (the web stream)
        # is closed early — e.g. the user hits Stop — the generator is closed and
        # the lock is released in the finally, so the next request can run.
        with _GEN_LOCK:
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            stream = self.tokenizer.create_stream()
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                piece = stream.decode(new_token)
                if piece:
                    yield piece
