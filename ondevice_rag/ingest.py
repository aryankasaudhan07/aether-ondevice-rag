"""Load documents from disk and split them into overlapping text chunks.

Supports .pdf, .txt, and .md. Chunking is character-based with overlap — simple,
predictable, and dependency-light, which keeps it identical across Mac and
Snapdragon. Each chunk keeps a reference back to its source file so answers can
cite where they came from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ondevice_rag import config


@dataclass
class Chunk:
    text: str
    source: str          # file name
    chunk_index: int     # position within the source document
    meta: dict = field(default_factory=dict)


_MIN_PAGE_TEXT = 20  # a page with less text than this is treated as scanned → OCR
_ocr_engine = None


def _get_ocr():
    """Lazily create the on-device OCR engine (ONNX Runtime under the hood)."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_page(page) -> str:
    """Render one PDF page to an image and OCR it — for scanned documents."""
    import numpy as np
    import pymupdf as fitz

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)  # 2x = sharper OCR
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    result, _ = _get_ocr()(img)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF. Uses the embedded text layer when present, and
    falls back to on-device OCR for scanned/image pages that have no text."""
    import pymupdf  # (formerly "fitz")

    doc = pymupdf.open(str(path))
    parts: list[str] = []
    ocr_pages = 0
    try:
        for page in doc:
            text = page.get_text().strip()
            if len(text) < _MIN_PAGE_TEXT:  # looks scanned → OCR it
                ocr_text = _ocr_page(page).strip()
                if ocr_text:
                    text = ocr_text
                    ocr_pages += 1
            if text:
                parts.append(text)
    finally:
        doc.close()
    if ocr_pages:
        print(f"    (OCR used on {ocr_pages} scanned page(s))")
    return "\n".join(parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_READERS = {".pdf": _read_pdf, ".txt": _read_text, ".md": _read_text}


def load_document(path: Path) -> str:
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"unsupported file type: {path.suffix} ({path.name})")
    return reader(path)


def chunk_text(
    text: str,
    source: str,
    chunk_chars: int = config.CHUNK_CHARS,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split text into overlapping windows, trying to break on whitespace."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(1, chunk_chars - overlap)
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        # prefer to break on whitespace so we don't cut words in half
        if end < len(text):
            space = text.rfind(" ", start + step, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_index=idx))
            idx += 1
        if end >= len(text):
            break
        start = end - overlap  # slide window forward, keeping overlap
    return chunks


def _cache_path() -> Path:
    return config.INDEX_DIR / "extract_cache.json"


def _load_cache() -> dict[str, str]:
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt cache is not fatal
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache), encoding="utf-8")


def ingest_dir(data_dir: Path | None = None) -> list[Chunk]:
    """Load and chunk every supported document under data_dir.

    Extracted text is cached per (name, mtime, size), so re-indexing, deleting,
    or uploading only does the expensive work (especially OCR) for files that are
    new or changed. Unchanged files are served from cache instantly. The rebuilt
    cache contains only current files, so deleted files are pruned automatically.
    """
    data_dir = data_dir or config.DATA_DIR
    old_cache = _load_cache()
    new_cache: dict[str, str] = {}
    all_chunks: list[Chunk] = []

    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in _READERS or not path.is_file():
            continue
        st = path.stat()
        key = f"{path.name}:{st.st_mtime_ns}:{st.st_size}"
        raw = old_cache.get(key)
        if raw is None:  # new or changed file → extract (may run OCR)
            try:
                raw = load_document(path)
            except Exception as exc:  # noqa: BLE001 - a bad file shouldn't kill ingest
                print(f"  ! skipped {path.name}: {exc}")
                continue
        new_cache[key] = raw
        chunks = chunk_text(raw, source=path.name)
        print(f"  + {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    _save_cache(new_cache)
    return all_chunks
