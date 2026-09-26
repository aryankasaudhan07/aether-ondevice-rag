# Porting to the Snapdragon NPU

This guide moves the app from your Mac (CPU) to your Snapdragon laptop and onto
the **Hexagon NPU**. Do it in three stages — each stage leaves you with a working
app, so you always have something to demo.

> **Golden rule:** the application logic never changes. Only *which ONNX Runtime
> execution provider runs the models* changes, and that is decided in one place —
> [`ondevice_rag/config.py`](../ondevice_rag/config.py).

---

## Stage A — run on the Snapdragon at all (CPU) ✅ guaranteed

Proves the whole app runs on Windows-on-Arm before touching the NPU.

```powershell
# On the Snapdragon laptop (Windows on Arm), in PowerShell:

# 1. install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. copy/clone this project, then from its folder:
uv sync
uv run python scripts/download_models.py

# 3. run it
uv run uvicorn ondevice_rag.server:app --port 8000
```

At this point `config.detect_backend()` returns `qnn` on this machine, but if the
QNN provider or NPU models aren't installed yet, ONNX Runtime **automatically
falls back to CPU** (that fallback is built into `ort_providers()`). Force pure
CPU explicitly if you want a clean baseline:

```powershell
$env:RAG_BACKEND = "cpu"
```

**Deliverable:** the app running on the Snapdragon, CPU. Benchmark it (tokens/sec,
query latency) — this is your "before" number.

---

## Stage B — embeddings on the NPU 🎯 the reliable NPU win

The embedding model is small and quantizes cleanly, so it's the dependable way to
get real NPU acceleration.

1. **Install the QNN build of ONNX Runtime** (replaces the stock CPU wheel):
   ```powershell
   uv pip uninstall onnxruntime
   uv pip install onnxruntime-qnn
   ```
2. **Get an NPU-ready embedding model.** Float32 ONNX will mostly fall back to CPU
   on the NPU. Compile/quantize `all-MiniLM-L6-v2` for the Hexagon target using
   **Qualcomm AI Hub** (free): upload the model, target a Snapdragon X device,
   download the produced `.onnx`, and drop it in `models/all-MiniLM-L6-v2/`.
   - AI Hub: https://aihub.qualcomm.com
3. **Run with QNN:**
   ```powershell
   $env:RAG_BACKEND = "qnn"
   uv run python scripts/build_index.py
   ```
   In the output, `active provider:` should now read `QNNExecutionProvider`. The
   web UI badge will switch to **"Snapdragon NPU (QNN)."**

**Deliverable:** embeddings measurably faster / lower-power on the NPU vs Stage A.

---

## Stage C — the LLM on the NPU 🚀 the frontier / stretch goal

This is the ambitious part (and the best story for your write-up). Two paths:

- **Path 1 — onnxruntime-genai QNN:** use a Snapdragon/QNN-targeted build of the
  Phi-3.5 (or Llama-3.2-3B) ONNX model. Qualcomm AI Hub and the ONNX Runtime
  GenAI docs publish NPU model variants and the matching `onnxruntime-genai` QNN
  wheel. Point `config.LLM_MODEL_DIR` at that model directory — `llm.py` needs no
  changes.
- **Path 2 — Qualcomm Genie / QAIRT:** Qualcomm's native LLM runtime for
  Snapdragon. More setup, best NPU performance. Wrap it behind the same `LLM`
  interface in `llm.py` so `pipeline.py` is untouched.

**If the NPU LLM fights you, ship Stage B + a CPU LLM.** The app is still fully
on-device and fully working; the NPU-LLM attempt is documented as future work.
That is an honest, strong result — not a failure.

---

## Demo checklist

- [ ] App launches on the Snapdragon laptop
- [ ] Backend badge shows **Snapdragon NPU (QNN)**
- [ ] Turn WiFi **off** live — it still answers (the on-device money shot)
- [ ] Show a before/after benchmark (CPU vs NPU) for embeddings
- [ ] Answers cite their source chunks

## Troubleshooting

| Symptom | Fix |
|---|---|
| `active provider` stays `CPUExecutionProvider` under `qnn` | Model isn't NPU-compiled (Stage B step 2), or `onnxruntime-qnn` not installed |
| `QnnHtp.dll not found` | Install `onnxruntime-qnn`; ensure QNN libs are on `PATH` |
| App runs but slow | Confirm you're on the NPU, not the CPU fallback; check provider in logs |
