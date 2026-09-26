"""Download the ONNX models this project needs into ./models.

Embedding model: all-MiniLM-L6-v2 (ONNX + tokenizer.json), 384-dim sentence
embeddings, ~90 MB. Small enough to run comfortably on the NPU later.

Run:  uv run python scripts/download_models.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from ondevice_rag import config

EMBED_REPO = "Xenova/all-MiniLM-L6-v2"  # ships onnx/model.onnx + tokenizer.json

# Phi-3.5-mini instruct, ONNX for onnxruntime-genai. The CPU int4 variant runs on
# your Mac now; a QNN/NPU variant swaps in on the Snapdragon (same code).
LLM_REPO = "microsoft/Phi-3.5-mini-instruct-onnx"
LLM_VARIANT = "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"


def _fetch(repo: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  = {dest.relative_to(config.ROOT)} (already present)")
        return
    print(f"  ↓ {repo}/{filename}")
    cached = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(cached, dest)
    print(f"  + {dest.relative_to(config.ROOT)}")


def download_embedder() -> None:
    print("Embedding model: all-MiniLM-L6-v2")
    _fetch(EMBED_REPO, "onnx/model.onnx", config.EMBED_MODEL_DIR / "model.onnx")
    _fetch(EMBED_REPO, "tokenizer.json", config.EMBED_MODEL_DIR / "tokenizer.json")


def download_llm() -> None:
    print(f"LLM: {LLM_REPO} ({LLM_VARIANT}) — this is the ~2 GB download")
    dest = config.LLM_MODEL_DIR
    if (dest / "genai_config.json").exists():
        print(f"  = {dest.relative_to(config.ROOT)} (already present)")
        return

    staging = config.MODELS_DIR / "_llm_staging"
    snapshot_download(
        repo_id=LLM_REPO,
        allow_patterns=[f"{LLM_VARIANT}/*"],
        local_dir=str(staging),
    )
    # Find the folder that actually holds genai_config.json and flatten it.
    cfg = next(staging.rglob("genai_config.json"))
    src = cfg.parent
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, dest / f.name)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  + LLM ready in {dest.relative_to(config.ROOT)}")


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in {"all", "embed"}:
        download_embedder()
    if which in {"all", "llm"}:
        download_llm()
    print("\nDone.")
