"""Central configuration and — most importantly — the single place where the
compute backend (CPU on your Mac now, Snapdragon NPU later) is chosen.

The whole project is written so that moving from Mac development to the
Snapdragon NPU is *one* decision made here, not a rewrite. Everything else in the
codebase asks this module which ONNX Runtime execution providers to use and never
hardcodes a backend.

Override auto-detection with the RAG_BACKEND env var: "cpu", "coreml", or "qnn".
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

# --- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"        # user documents to index
MODELS_DIR = ROOT / "models"    # downloaded ONNX models
INDEX_DIR = ROOT / "index"      # persisted vector index

EMBED_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"  # embedding model + tokenizer
LLM_MODEL_DIR = MODELS_DIR / "llm"                 # onnxruntime-genai model dir

# --- retrieval knobs ------------------------------------------------------
CHUNK_CHARS = 900          # target characters per chunk
CHUNK_OVERLAP = 150        # overlap between consecutive chunks
TOP_K = 4                  # chunks retrieved per question
EMBED_MAX_TOKENS = 256     # truncate long chunks for the embedder

# --- backend selection ----------------------------------------------------
def detect_backend() -> str:
    """Return 'cpu', 'coreml', or 'qnn'. Honors RAG_BACKEND override."""
    forced = os.environ.get("RAG_BACKEND", "").strip().lower()
    if forced in {"cpu", "coreml", "qnn"}:
        return forced

    system = platform.system().lower()
    machine = platform.machine().lower()

    # Snapdragon X laptops run Windows on Arm -> use the Hexagon NPU via QNN.
    if system == "windows" and machine in {"arm64", "aarch64"}:
        return "qnn"

    # Everything else (your Apple Silicon Mac) develops on CPU.
    return "cpu"


def ort_providers() -> list:
    """ONNX Runtime execution providers, best-first, for the current backend.

    This is the ONE function the rest of the code relies on. On the Snapdragon
    laptop it returns the QNN (NPU) provider with a CPU fallback so the app never
    hard-fails; on the Mac it returns CPU.
    """
    backend = detect_backend()

    if backend == "qnn":
        # QnnHtp.dll targets the Hexagon Tensor Processor (the NPU) on Snapdragon.
        return [
            ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),
            "CPUExecutionProvider",
        ]
    if backend == "coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def backend_summary() -> str:
    return f"backend={detect_backend()} providers={ort_providers()}"
