"""Registry of selectable LLMs + download/selection helpers.

Users pick a model that fits their machine: a tiny fast one for low-RAM laptops,
a bigger higher-quality one for machines with more memory, or the ONNX model for
the Snapdragon NPU. The selection is persisted so it survives restarts.

Backends:
- "gguf": a llama.cpp model (one .gguf file in ./models) — fast/light on CPU.
- "onnx": the onnxruntime-genai model in ./models/llm — the Snapdragon NPU path.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass

from ondevice_rag import config


@dataclass
class ModelInfo:
    id: str
    label: str
    backend: str      # "gguf" | "onnx"
    tagline: str      # short quality note
    ram: str          # memory guidance for the user
    size_gb: float
    repo: str
    file: str = ""    # gguf filename (backend == "gguf")
    variant: str = "" # onnx subfolder (backend == "onnx")


REGISTRY: list[ModelInfo] = [
    ModelInfo(
        id="llama-1b", label="Llama 3.2 · 1B", backend="gguf",
        tagline="Fastest, lightest", ram="Great for 8 GB laptops", size_gb=0.8,
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        file="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    ),
    ModelInfo(
        id="llama-3b", label="Llama 3.2 · 3B", backend="gguf",
        tagline="Balanced quality", ram="Needs ~6 GB free RAM", size_gb=2.0,
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        file="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    ModelInfo(
        id="qwen-7b", label="Qwen 2.5 · 7B", backend="gguf",
        tagline="Best quality", ram="Needs 16 GB+ RAM", size_gb=4.7,
        repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        file="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ),
    ModelInfo(
        id="phi35-onnx", label="Phi-3.5 mini · ONNX", backend="onnx",
        tagline="Snapdragon NPU path", ram="Heavy on 8 GB CPU", size_gb=2.7,
        repo="microsoft/Phi-3.5-mini-instruct-onnx",
        variant="cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4",
    ),
]

DEFAULT_ID = "llama-1b"
_SEL_FILE = config.INDEX_DIR / "selected_model.json"


def by_id(model_id: str) -> ModelInfo | None:
    return next((m for m in REGISTRY if m.id == model_id), None)


def gguf_path(m: ModelInfo):
    return config.MODELS_DIR / m.file


def is_downloaded(m: ModelInfo) -> bool:
    if m.backend == "gguf":
        return gguf_path(m).exists()
    return (config.LLM_MODEL_DIR / "genai_config.json").exists()


def selected_id() -> str:
    try:
        return json.loads(_SEL_FILE.read_text(encoding="utf-8"))["id"]
    except Exception:  # noqa: BLE001
        return DEFAULT_ID


def set_selected(model_id: str) -> None:
    _SEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SEL_FILE.write_text(json.dumps({"id": model_id}), encoding="utf-8")


def selected() -> ModelInfo:
    """The chosen model, falling back to any downloaded one if it's missing."""
    m = by_id(selected_id()) or by_id(DEFAULT_ID)
    if m and is_downloaded(m):
        return m
    for candidate in REGISTRY:
        if is_downloaded(candidate):
            return candidate
    return m or REGISTRY[0]


def list_status() -> list[dict]:
    cur = selected().id
    out = []
    for m in REGISTRY:
        d = asdict(m)
        d["downloaded"] = is_downloaded(m)
        d["current"] = m.id == cur
        out.append(d)
    return out


def download(m: ModelInfo) -> None:
    """Fetch a model's weights if not already present (can be several GB)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if is_downloaded(m):
        return
    if m.backend == "gguf":
        cached = hf_hub_download(repo_id=m.repo, filename=m.file)
        gguf_path(m).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, gguf_path(m))
        return
    # onnx (Snapdragon NPU path)
    staging = config.MODELS_DIR / "_llm_staging"
    snapshot_download(repo_id=m.repo, allow_patterns=[f"{m.variant}/*"], local_dir=str(staging))
    cfg = next(staging.rglob("genai_config.json"))
    dest = config.LLM_MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    for f in cfg.parent.iterdir():
        if f.is_file():
            shutil.copy(f, dest / f.name)
    shutil.rmtree(staging, ignore_errors=True)
