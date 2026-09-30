# Aether — On-Device RAG Assistant

A fully offline, privacy-preserving **document question-answering** app. Drop in
your PDFs/notes, ask questions in natural language, and get grounded, cited
answers — with **every step (OCR, embeddings, retrieval, and the language model)
running locally on the device**. No cloud, no internet, no data leaving the
machine.

Built for the **Qualcomm Snapdragon AI Lab Build & Present Challenge**: the app is
written to run on the **Snapdragon NPU (Hexagon)**, with the compute backend
switchable in a single config file. It runs on any CPU for development and is
NPU-ready by design.

![status](https://img.shields.io/badge/runs-100%25%20on--device-3fb950)

---

## Why on-device matters

| | Cloud AI | Aether (on-device) |
|---|---|---|
| Privacy | Data sent to a server | **Never leaves the device** |
| Offline | Needs internet | **Works with no connection** |
| Latency | Network round-trip | **Local, immediate** |
| Cost | Per-request API fees | **Zero** |

## Architecture

```
  your .pdf/.txt/.md ─► chunk (+ OCR for scans) ─► embed (ONNX) ─► vector index
                                                                        │ (./index)
                                                                        ▼
  your question ─► embed ─► retrieve top-k (diversity-aware) ─► grounded prompt
                                                                        │
                                                                        ▼
                                        on-device LLM ─► answer + cited sources
```

- **Embeddings & OCR** run through **ONNX Runtime** (`all-MiniLM-L6-v2`, RapidOCR)
  — no PyTorch at runtime, and portable to the NPU.
- **Language model** runs through **llama.cpp** (fast, low-memory GGUF models on
  CPU) or **onnxruntime-genai** (the ONNX model for the Snapdragon NPU path).
- **Backend:** FastAPI with server-sent events; a single-page web UI.

### Choose a model for your machine

The in-app **model picker** lets you switch models without touching code:

| Model | Backend | Good for |
|---|---|---|
| **Llama 3.2 · 1B** (default) | llama.cpp (GGUF) | Fastest; great on 8 GB laptops |
| Llama 3.2 · 3B | llama.cpp (GGUF) | Balanced quality (~6 GB RAM) |
| Qwen 2.5 · 7B | llama.cpp (GGUF) | Best quality (16 GB+ RAM) |
| Phi-3.5 mini · ONNX | onnxruntime-genai | The Snapdragon **NPU** path |

### The one switch: CPU ↔ NPU

The codebase asks [`config.ort_providers()`](ondevice_rag/config.py) which backend
to use and **never hardcodes one**. It auto-detects:

- **CPU** (development) → `CPUExecutionProvider`
- **Snapdragon (Windows on Arm)** → `QNNExecutionProvider` — the Hexagon **NPU**

Force it with `RAG_BACKEND=cpu|coreml|qnn`. Moving to the NPU is configuration,
not a rewrite. See [docs/SNAPDRAGON_PORT.md](docs/SNAPDRAGON_PORT.md).

---

## Features

- **On-device OCR** — reads scanned / image PDFs locally (RapidOCR + PyMuPDF).
- **Model picker** — pick a model that fits your machine; download-on-demand.
- **Per-question document scoping** — search only ticked PDFs, or name a document
  in the question and it scopes automatically.
- **Cited answers** — sources grouped by document.
- **Markdown + math rendering** — tables and LaTeX render in the answer.
- **Streaming** responses with a stop/interrupt control.
- **Extraction cache** — re-indexing stays fast, even with OCR.

## Quickstart

```bash
# 1. install uv (one time)  ─ https://astral.sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. install deps + Python 3.12
uv sync

# 3. download the models (embedder ~90 MB, default LLM ~0.8 GB GGUF)
uv run python scripts/download_models.py
#    optional: the ONNX model for the NPU path (~2.7 GB)
#    uv run python scripts/download_models.py llm-onnx

# 4a. web app  →  http://127.0.0.1:8000
uv run uvicorn ondevice_rag.server:app --port 8000

# 4b. terminal usage
uv run python scripts/build_index.py
uv run python scripts/ask.py "your question here"
```

Put documents in `./data`, click **Re-index** in the web UI (or upload from the
sidebar — new files auto-index), then ask.

## Project layout

```
ondevice_rag/
  config.py        # paths + the CPU/NPU backend switch  ← the key file
  models.py        # the selectable-model registry + download/selection
  ingest.py        # load & chunk PDF/TXT/MD, with on-device OCR + a cache
  embeddings.py    # sentence embeddings via ONNX Runtime
  vectorstore.py   # NumPy cosine-similarity index (diversity-aware) + persistence
  llm.py           # generation via llama.cpp (GGUF) or onnxruntime-genai (ONNX/NPU)
  pipeline.py      # RAG: retrieve → grounded prompt → generate
  server.py        # FastAPI backend (SSE streaming, model + document APIs)
web/index.html     # single-page chat UI  (web/vendor/ = offline Markdown + math)
scripts/           # download_models, build_index, ask
```

## What this project demonstrates

- Retrieval-Augmented Generation (RAG), end to end, fully on-device.
- On-device **OCR**, embeddings, and LLM inference — with a swappable model picker.
- **Backend-agnostic design**: CPU for development, the **Snapdragon NPU** via one
  config switch, same code.
- Practical engineering for constrained hardware (right-sizing the model for RAM,
  an extraction cache, diversity-aware retrieval, streaming over SSE).

## License

MIT (see project owner).
