# On-Device RAG Assistant

A fully offline, privacy-preserving **document question-answering** app. Drop in
your PDFs/notes, ask questions in natural language, and get grounded, cited
answers — with **every step (embeddings, retrieval, and the language model)
running locally on the device**. No cloud, no internet, no data leaving the
machine.

Built for the **Qualcomm Snapdragon AI Lab Build & Present Challenge**: the app
is written to run on the **Snapdragon NPU (Hexagon)**, and is developed on a Mac
with the compute backend switchable in a single config file.

![status](https://img.shields.io/badge/runs-100%25%20on--device-3fb950)

---

## Why on-device matters

| | Cloud AI | This app (on-device) |
|---|---|---|
| Privacy | Data sent to a server | **Never leaves the device** |
| Offline | Needs internet | **Works on a plane** |
| Latency | Network round-trip | **No round-trip** |
| Cost | Per-request API fees | **Zero** |

## Architecture

```
                        ┌──────────────────────────────────────────┐
  your .pdf/.txt/.md ──► │ ingest → chunk → embed (ONNX) → vector DB │
                        └──────────────────────────────────────────┘
                                              │  (persisted to ./index)
                                              ▼
      your question ──► embed ──► retrieve top-k chunks ──► build grounded prompt
                                                                   │
                                                                   ▼
                                     on-device LLM (onnxruntime-genai)
                                                                   │
                                                                   ▼
                                       answer + inline source citations
```

Two AI models, both run through **ONNX Runtime**, both portable to the NPU:

- **Embeddings:** `all-MiniLM-L6-v2` (384-dim) — no PyTorch at runtime.
- **LLM:** `Phi-3.5-mini-instruct` (int4 quantized) via `onnxruntime-genai`.

### The one switch: CPU ↔ NPU

The entire codebase asks [`config.ort_providers()`](ondevice_rag/config.py) which
backend to use and **never hardcodes one**. It auto-detects:

- **Your Mac** → `CPUExecutionProvider` (development)
- **Snapdragon (Windows on Arm)** → `QNNExecutionProvider` — the Hexagon **NPU**

Force it with `RAG_BACKEND=cpu|coreml|qnn`. Moving to the NPU is configuration,
not a rewrite. See [docs/SNAPDRAGON_PORT.md](docs/SNAPDRAGON_PORT.md).

---

## Quickstart (Mac / development)

```bash
# 1. install uv (one time)  ─ https://astral.sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. install deps + Python 3.12
uv sync

# 3. download the models (embedder ~90MB, LLM ~2GB)
uv run python scripts/download_models.py

# 4a. terminal usage
uv run python scripts/build_index.py
uv run python scripts/ask.py "your question here"

# 4b. web app  →  http://127.0.0.1:8000
uv run uvicorn ondevice_rag.server:app --port 8000
```

Put documents in `./data`, click **Re-index** in the web UI (or run
`build_index.py`), then ask.

## Project layout

```
ondevice_rag/
  config.py        # paths + the CPU/NPU backend switch  ← the key file
  ingest.py        # load & chunk PDF/TXT/MD
  embeddings.py    # sentence embeddings via ONNX Runtime
  vectorstore.py   # NumPy cosine-similarity index + persistence
  llm.py           # on-device generation via onnxruntime-genai
  pipeline.py      # RAG: retrieve → grounded prompt → generate
  server.py        # FastAPI backend (SSE streaming)
web/index.html     # single-page chat UI
scripts/           # download_models, build_index, ask
```

## What this project demonstrates

- Retrieval-Augmented Generation (RAG) end to end
- Running quantized LLMs and embedding models with **ONNX Runtime**
- **On-device / edge inference** targeting the Qualcomm Hexagon NPU (QNN)
- Backend-agnostic design (CPU dev → NPU deploy with one switch)
- Streaming inference over Server-Sent Events, FastAPI

## License

MIT (see project owner).
