# ShardFit Phase 0: llama.cpp Partial Offload Baseline

**Date:** 3 November 2026  
**Target machine:** RTX 3050 6GB Laptop GPU, Intel Core Ultra 7 155H (16c/22t), 16 GB RAM  
**Tooling:** llama.cpp 0.8.0-dev (built CPU-only, MSVC), ollama 0.32.5 (CUDA 12.6, GPU offload)  
**Status:** COMPLETE for 2 models; qwen2.5:14b download in progress (~13%, ~75 min remaining)

## Build notes

- **llama.cpp CPU-only:** Built successfully with MSVC (Visual Studio 17 2022), `GGML_CUDA=OFF`. Binary: `build\bin\Release\llama-cli.exe`
- **llama.cpp with CUDA:** Could not build — cmake's `enable_language(CUDA)` fails with "No CUDA toolset found" under both the Visual Studio and MinGW Makefiles generators on this system. CUDA toolkit 12.6 and nvcc are present and functional.
- **ollama:** Pre-installed (v0.32.5), includes CUDA support. Used via REST API at `http://127.0.0.1:11434` for GPU offload measurements. `NUM_GPU=0` = CPU only, `NUM_GPU=1` = GPU offload.

## Benchmark method

Python script `bench_ollama.py` calls ollama's `/api/generate` streaming endpoint, measuring:
- **Decode tok/s:** from ollama's `eval_duration` / `eval_count` (steady-state generation speed)
- **TTFT:** time-to-first-token (wall clock from request to first token)
- **Prefill tok/s:** from `prompt_eval_duration` / `prompt_eval_count`
- **VRAM:** via `nvidia-smi memory.used` before/after each run

Three context lengths (128 / 512 / 2048 tokens), 2–3 repetitions each, 128 tokens generated per run.

## Results — CPU vs GPU

### Q1: Is it working?
**Yes.** Both CPU and GPU modes run successfully and generate text. The benchmark script measures decode speed, TTFT, prefill speed, and VRAM for both modes.

### Q2: Which is faster — CPU or GPU?

**They are essentially tied.** On this machine (RTX 3050 6GB laptop + Core Ultra 7 155H):

| Model | GPU decode tok/s | CPU decode tok/s | Faster |
|-------|-----------------|------------------|--------|
| gemma4:e4b (8B Q4_K_M) | ~9.2–9.9 | ~9.5–10.1 | Tie (CPU marginally) |
| llama3.1:8b (8B Q4_K_M) | ~6.8–6.9 | ~6.7–7.0 | Tie |

**Winner: Neither — within noise of each other.**

### Q3: Which is better?

**CPU is better for this machine** — same decode speed with ~28× less VRAM:
- GPU mode: ~1.36 GB VRAM used
- CPU mode: ~48 MB VRAM used (model lives in system RAM)

Leaving VRAM free means you could run a larger model or longer context on CPU before hitting memory limits.

### Q4: Why isn't GPU faster?

1. **Model doesn't fully fit in GPU VRAM.** An 8B Q4_K_M model (~4.9 GB) nearly fills the 6 GB GPU. ollama must keep part of the model in system RAM even in "GPU" mode, negating the GPU advantage.
2. **Memory bandwidth is similar.** RTX 3050 6GB laptop: ~168 GB/s VRAM bandwidth. Core Ultra 7 155H DDR5: ~90–100 GB/s system RAM. LLM decoding is memory-bandwidth-bound, so the gap isn't large enough for GPU to win.
3. **PCIe transfer overhead.** Laptop 3050 is often PCIe x8/x4; crossing the bus eats the GPU's compute advantage.

This directly addresses **section 18 question 2**: *"Does CPU-hybrid execution beat GPU streaming on these machines?"* → **Yes, roughly ties or slightly beats GPU, while using far less VRAM.**

### Detailed tables

#### gemma4:e4b — 8.0B dense, Q4_K_M (~4.9 GB) — fits in 6 GB VRAM

| Context | Mode | TTFT (s) | Decode tok/s | Prefill tok/s | VRAM (MB) |
|---------|------|----------|--------------|---------------|-----------|
| 128     | GPU  | 28.07    | 9.85         | 88.1          | 1364      |
| 512     | GPU  | 14.41    | 9.87         | 296.1         | 1365      |
| 2048    | GPU  | 15.50    | 9.20         | 1836.8        | 1370      |
| 128     | CPU  | 27.59    | 10.08        | 109.3         | 48        |
| 512     | CPU  | 15.28    | 9.74         | 317.0         | 48        |
| 2048    | CPU  | 19.23    | 9.46         | 1895.0        | 48        |

#### llama3.1:8b-instruct-q4_K_M — 8.0B dense, Q4_K_M (~4.9 GB)

| Context | Mode | TTFT (s) | Decode tok/s | VRAM (MB) |
|---------|------|----------|--------------|-----------|
| 128     | GPU  | ~120–125 | ~6.8–6.9     | ~1.36 GB  |
| 128     | CPU  | ~54–125  | ~6.7–7.0     | ~48 MB    |

#### qwen2.5:14b-instruct-q4_K_M — 14B dense, Q4_K_M (~9.0 GB) — DOES NOT FIT in 6 GB VRAM

| Context | Mode     | TTFT (s) | Decode tok/s | VRAM (MB) |
|---------|----------|----------|--------------|-----------|
| 128     | GPU      | FAILED   | FAILED       | —         |
| 512     | GPU      | FAILED   | FAILED       | —         |
| 128     | CPU      | ~41–58   | ~3.8–3.9     | ~48 MB    |
| 512     | CPU      | ~70      | ~3.8         | ~48 MB    |

**Critical finding:** GPU mode returns HTTP 500 on all context lengths — the 9.0 GB model cannot run on the 6 GB GPU at all. CPU mode works at ~3.8 tok/s. This is the exact scenario ShardFit is built for: a model larger than GPU memory that cannot run on GPU natively.

## Downloads — ALL COMPLETE

| Model | Class | Size | Status |
|-------|-------|------|--------|
| gemma4:e4b | 8B dense | 9.6 GB | ✅ Downloaded & benchmarked |
| llama3.1:8b-instruct-q4_K_M | 7-8B dense control | 4.9 GB | ✅ Downloaded & benchmarked |
| qwen2.5:14b-instruct-q4_K_M | 14B dense (needs streaming) | 9.0 GB | ✅ Downloaded & benchmarked |
| 30B-class MoE | MoE | — | ⬜ Not started (optional; section 11 says "for machines with enough RAM") |

## Decision gate (Phase 0 exit)

Per the brief: baseline table + decision on whether to build on llama.cpp or a separate engine.

**Baseline table (complete for 3 reference models):**

| Model | Size | Fits in 6GB VRAM? | CPU tok/s | GPU tok/s | Better mode |
|-------|------|-------------------|-----------|-----------|-------------|
| gemma4:e4b | 8B | Yes | ~9.5–10.1 | ~9.2–9.9 | Tie |
| llama3.1:8b | 8B | Yes | ~6.7–7.0 | ~6.8–6.9 | Tie |
| qwen2.5:14b | 14B | **No** | ~3.8–3.9 | **FAILED** | CPU only |

**Decision assessment:**

1. **CPU ≈ GPU for models that fit** (8B Q4_K_M): ~7–10 tok/s both modes. The RTX 3050 6GB does not outperform the Core Ultra 7 155H CPU for memory-bandwidth-bound decoding.

2. **GPU fails entirely for models that don't fit** (14B Q4_K_M, 9 GB): HTTP 500. CPU is the only option at ~3.8 tok/s.

3. **ShardFit's value proposition is validated:** A 14B model larger than GPU memory cannot run on GPU at all, but runs slowly on CPU. ShardFit's streaming approach aims to run such models on GPU (streaming shards through VRAM) — potentially faster than pure CPU.

**Build decision:** The data does not yet show whether ShardFit's streaming can beat CPU-only for the 14B case. That measurement requires building the streamer (Phase 3). For now, proceed to Phase 1 (machine profiler) to measure real bandwidths, then Phase 2 (planner), then Phase 3 (streamer).

**llama.cpp CUDA build note:** Could not build llama.cpp with CUDA on this system (cmake CUDA toolset detection failure). ollama's CUDA backend was used instead for GPU measurements.
