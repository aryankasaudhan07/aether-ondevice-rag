"""FastAPI backend for the on-device RAG assistant.

Endpoints:
    GET  /                -> the web UI
    GET  /api/status      -> backend + document/index stats
    POST /api/upload      -> upload documents into ./data
    POST /api/reindex     -> (re)build the vector index from ./data
    GET  /api/ask?q=...   -> Server-Sent Events: sources first, then answer tokens

The pipeline (embedder + index + LLM) is loaded lazily and cached, because loading
the LLM takes a few seconds. Everything runs locally — no request ever leaves the
machine.

Run:  uv run uvicorn ondevice_rag.server:app --reload
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ondevice_rag import config
from ondevice_rag.embeddings import Embedder
from ondevice_rag.ingest import ingest_dir
from ondevice_rag.pipeline import RagPipeline
from ondevice_rag.vectorstore import VectorStore

WEB_DIR = config.ROOT / "web"

app = FastAPI(title="On-Device RAG")

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline so the next query reloads the fresh index."""
    global _pipeline
    _pipeline = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> JSONResponse:
    import platform as _pf

    docs = [p.name for p in config.DATA_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    store = VectorStore.load()
    backend = config.detect_backend()

    # A friendly accelerator label + whether this run is actually on the NPU.
    accel_labels = {
        "qnn": "Snapdragon NPU (Hexagon)",
        "coreml": "Apple Neural Engine",
        "cpu": "CPU",
    }
    return JSONResponse(
        {
            "backend": backend,
            "accelerator": accel_labels.get(backend, backend.upper()),
            "on_npu": backend == "qnn",
            "providers": [str(p) for p in config.ort_providers()],
            "platform": f"{_pf.system()} {_pf.machine()}",
            "documents": docs,
            "chunks_indexed": len(store),
        }
    )


@app.post("/api/upload")
async def upload(files: list[UploadFile]) -> JSONResponse:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if Path(f.filename).suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        dest = config.DATA_DIR / Path(f.filename).name
        dest.write_bytes(await f.read())
        saved.append(dest.name)
    return JSONResponse({"saved": saved})


def _rebuild_index() -> dict:
    """Re-ingest ./data and rebuild the vector index. Handles the empty case by
    clearing the index so deleted documents don't linger in search results."""
    chunks = ingest_dir()
    store = VectorStore()
    provider = None
    if chunks:
        embedder = Embedder()
        vectors = embedder.encode([c.text for c in chunks])
        store.add(chunks, vectors)
        provider = embedder.active_provider
    store.save()  # writes empty index (and clears stale vectors) when no chunks
    reset_pipeline()
    have_text = {c.source for c in chunks}
    files = [
        p.name for p in config.DATA_DIR.rglob("*")
        if p.suffix.lower() in {".pdf", ".txt", ".md"} and p.is_file()
    ]
    empty = sorted(f for f in files if f not in have_text)  # no readable text found
    return {
        "chunks_indexed": len(store),
        "documents": len(have_text),
        "empty_documents": empty,
        "provider": provider,
    }


@app.post("/api/reindex")
def reindex() -> JSONResponse:
    return JSONResponse(_rebuild_index())


@app.delete("/api/document")
def delete_document(name: str) -> JSONResponse:
    """Delete one document from ./data (by name) and rebuild the index."""
    target = config.DATA_DIR / Path(name).name  # basename only — no path traversal
    deleted = False
    if target.exists() and target.is_file():
        target.unlink()
        deleted = True
    result = _rebuild_index()
    result["deleted"] = deleted
    return JSONResponse(result)


@app.get("/api/ask")
def ask(q: str, request: Request) -> StreamingResponse:
    def event_stream():
        try:
            rag = get_pipeline()
            hits, token_stream = rag.stream_answer(q)
            sources = [
                {"source": h.chunk.source, "chunk_index": h.chunk.chunk_index, "score": round(h.score, 3)}
                for h in hits
            ]
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            if not hits:
                yield f"event: token\ndata: {json.dumps('No documents are indexed yet. Add a file and re-index, then ask again.')}\n\n"
            for piece in token_stream:
                yield f"event: token\ndata: {json.dumps(piece)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:  # surface errors to the UI instead of a silent close
            import traceback
            traceback.print_exc()
            yield f"event: failure\ndata: {json.dumps(f'{type(exc).__name__}: {exc}')}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
