"""Ask a question against the indexed documents, from the terminal.

Run:  uv run python scripts/ask.py "your question here"
"""
from __future__ import annotations

import sys

from ondevice_rag import config
from ondevice_rag.pipeline import RagPipeline


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: uv run python scripts/ask.py "your question"')
        return

    print(config.backend_summary())
    print("Loading pipeline (embedder + index + LLM) ...")
    rag = RagPipeline()

    hits, stream = rag.stream_answer(question)
    print(f"\nQ: {question}\n")
    print("Sources:")
    for h in hits:
        print(f"  - {h.chunk.source}#{h.chunk.chunk_index}  (score {h.score:.3f})")
    print("\nAnswer:")
    for piece in stream:
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
