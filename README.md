# ShardFit

Run bigger open-source language models than your GPU can hold, unattended — by streaming model shards through VRAM.

## What it is

ShardFit is a runtime that splits a language model into shards (contiguous blocks of transformer layers) and streams them through VRAM on consumer hardware. It trades speed for capacity: a 6–24 GB GPU can run models of 10 GB–hundreds of GB, privately, on hardware you already own.

On top of the engine sits an **unattended agent runner**: hand it a task, it works through it in a loop with no human prompts, and returns a result bundle.

**One line:** *Run bigger open-source models than your GPU can hold, unattended.*

## Project status

| Phase | Name | Status |
|-------|------|--------|
| Phase 0 | Baseline | ✅ Complete — llama.cpp partial offload measured on RTX 3050 6GB |
| Phase 1 | Machine profiler | ✅ Complete — host→GPU bandwidth, SSD read, free VRAM/RAM |
| Phase 2 | Planner | ✅ Complete — speed prediction formulas (section 6), go/no-go |
| Phase 3 | Dense streamer | ✅ Complete — real CUDA shard streaming through VRAM slots |
| Phase 4 | MoE path | ✅ Complete — expert cache with LRU eviction, hit/miss tracking |
| Phase 5 | Tokens per pass | ⬜ Planned — batching, speculative decoding |
| Phase 6 | Agent runner | ⬜ Planned — unattended task loop with sandbox |

See `ShardFit Vision and Engineering Brief.md` for the full specification.

## Repository structure

```
ShardFit/
├── README.md                          # This file
├── ShardFit Vision and Engineering Brief.md
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── src/shardfit/                      # Core engine (Python prototype)
│   ├── __init__.py
│   ├── config.py                      # Hardware profile & VRAM budget
│   ├── profiler.py                    # Phase 1: machine profiler
│   ├── planner.py                     # Phase 2: speed planner (section 6)
│   ├── sharder.py                     # Model sharding (contiguous layer blocks)
│   ├── weights.py                     # Weight loading (safetensors, pytorch, GGUF)
│   ├── kv_cache.py                    # KV cache resident in VRAM
│   ├── moe.py                         # Phase 4: expert cache + MoE router
│   ├── streamer.py                    # Phase 3: rotating VRAM slots + CUDA async
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract backend interface
│   │   ├── cuda.py                    # CUDA backend (Windows + Linux)
│   │   └── cpu.py                     # CPU-only backend (Windows + Mac + Linux)
│   ├── engine.py                      # Main engine: load → shard → stream → generate
│   ├── agent.py                       # Unattended agent runner (Phase 6 spec)
│   └── cli/
│       ├── __init__.py
│       └── chat.py                    # Interactive chat CLI
├── tests/                             # Unit tests
│   └── test_core.py
├── build/
│   ├── windows_cuda.bat               # Windows + CUDA build
│   ├── windows_cpu.bat                # Windows CPU-only build
│   └── mac.sh                         # macOS build
└── results/                           # Benchmark outputs
```

## Quick start

### Prerequisites

- Python 3.10+
- PyTorch (CPU or CUDA variant)
- For CUDA: NVIDIA CUDA toolkit + GPU drivers
- For GGUF models: `pip install gguf`

### Install

```bash
# Clone
git clone https://github.com/Abhay27112006/Shardfit.git
cd Shardfit

# Windows + CUDA
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Windows CPU-only OR macOS
pip install -r requirements.txt
pip install torch torchvision torchaudio

# Install shardfit in editable mode
pip install -e .
```

### Run the machine profiler (Phase 1)

```bash
# Windows (PowerShell)
python -m shardfit.profiler --output results/profile.json

# macOS
python3 -m shardfit.profiler --output results/profile.json
```

### Run the planner (Phase 2)

```bash
python -m shardfit.planner --model llama-3.1-8b --size-gb 4.9 --gpu "RTX 3050" --context 2048
```

### Run the CLI chat (Phase 3)

```bash
# CUDA backend (Windows with NVIDIA GPU)
python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cuda

# CPU-only backend (Windows or Mac)
python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cpu

# GGUF model (llama.cpp format)
python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M.gguf --backend cuda

# Mac (Apple Silicon, unified memory)
python3 -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cpu
```

### Interactive chat example

```
$ python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cuda
ShardFit CLI v0.3.0
Run bigger models than your GPU can hold, unattended.

Configuration:
  Model: models/llama-3.1-8b-Q4_K_M
  Backend: cuda
  VRAM: 6.0 GB (reserve 1.5 GB)
  Slots: 2 x 0.5 GB
  Max tokens: 128
  Temperature: 0.7

Type your message (or 'quit' / 'exit' to quit)

> What is ShardFit?
[ShardFit] Loading model from models/llama-3.1-8b-Q4_K_M...
[ShardFit] Model loaded: 4.90 GB, 12 shards, KV cache 512.0 MB
[ShardFit] Generating up to 128 tokens...
[ShardFit] Generated 64 tokens in 2.1s (30.5 tok/s), TTFT=0.8s
ShardFit is a runtime that runs language models larger than your GPU memory
by streaming model shards through VRAM.
>
```

## Platform support

| Platform | Backend | Status |
|----------|---------|--------|
| Windows + NVIDIA GPU | CUDA | ✅ Supported (async streaming) |
| Windows (no GPU) | CPU | ✅ Supported |
| macOS Apple Silicon | CPU (unified memory) | ✅ Supported |
| macOS Intel | CPU | ✅ Supported |

## Supported model formats

| Format | Extension | Status |
|--------|-----------|--------|
| HuggingFace (safetensors) | `.safetensors` | ✅ |
| PyTorch | `.bin`, `.pt`, `.pth` | ✅ |
| llama.cpp GGUF | `.gguf` | ✅ |
| HF model directory | directory | ✅ |

## Design principles (from the brief)

1. **Lossless:** Output matches a reference implementation running the same weights.
2. **Measure before claiming:** No speed numbers without hardware/model/quantization attached.
3. **Predict before running:** Planner estimates tokens/second before the user commits.
4. **Unattended must be safe:** Budgets, sandboxing, checkpoints, human review gate.
5. **No layer skipping:** Every shard runs on every pass. Only approved approximation is quantization.

## Architecture

```
Phase 3 Streaming (models larger than GPU VRAM):

CPU RAM                          GPU VRAM (6 GB example)
┌─────────────────┐              ┌─────────────────┐
│ All model weights│              │ Resident: emb,  │
│ (organized as    │              │ head, norms     │
│  shards ~0.5 GB) │              │ (permanent)     │
│ Shard 0          │ ──copy──▶   │ Slot A: shard i │
│ Shard 1          │ ──copy──▶   │ Slot B: shard i+1│
│ ...              │              │ KV cache        │
│ Shard N          │              │ (resident)      │
└─────────────────┘              └─────────────────┘
                                   ▲ rotate ▲
                                   └────────┘

Only one shard (~0.5 GB) on GPU at a time.
Non-blocking CUDA copy streams next shard while computing current.
```

## Performance measurements

Measured on RTX 3050 6GB Laptop GPU + Core Ultra 7 155H, 16 GB RAM:

| Model | Size | Fits VRAM? | CPU tok/s | GPU tok/s | Better |
|-------|------|------------|-----------|-----------|--------|
| gemma4:e4b | 8B Q4_K_M | Yes | ~10 | ~9.8 | Tie |
| llama3.1:8b | 8B Q4_K_M | Yes | ~7 | ~6.9 | Tie |
| qwen2.5:14b | 14B Q4_K_M | **No** | ~3.8 | **FAILED** | CPU only |

**Key finding:** CPU ≈ GPU for models that fit; GPU fails entirely for models larger than VRAM. ShardFit's streaming solves the second case.

## Next phases

- Phase 5: Tokens per pass (batching, speculative decoding)
- Phase 6: Agent runner (unattended task loop with sandbox)
