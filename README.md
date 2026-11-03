# ShardFit

Run bigger open-source language models than your GPU can hold, unattended — by streaming model shards through VRAM.

## What it is

ShardFit is a runtime that splits a language model into shards (contiguous blocks of transformer layers) and streams them through VRAM on consumer hardware. It trades speed for capacity: a 6–24 GB GPU can run models of 10 GB–hundreds of GB, privately, on hardware you already own.

On top of the engine sits an **unattended agent runner**: hand it a task, it works through it in a loop with no human prompts, and returns a result bundle.

**One line:** *Run bigger open-source models than your GPU can hold, unattended.*

## Project status

- **Phase 0 (baseline):** Complete. Measured llama.cpp partial offload on RTX 3050 6GB laptop. CPU ≈ GPU for fitting models; GPU fails entirely for 14B models that don't fit in VRAM.
- **Phase 1 (machine profiler):** In progress — this implementation.
- **Phase 2 (planner):** Planned.
- **Phase 3 (dense streamer):** Planned.

See `ShardFit Vision and Engineering Brief.md` for the full specification.

## Repository structure

```
ShardFit/
├── src/shardfit/           # Core engine (Python prototype)
│   ├── config.py           # Hardware profile & VRAM budget
│   ├── profiler.py         # Phase 1: machine profiler
│   ├── planner.py          # Phase 2: speed planner (section 6 formulas)
│   ├── sharder.py          # Model sharding (contiguous layer blocks)
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── cuda.py         # CUDA backend (Windows + Linux)
│   │   └── cpu.py          # CPU-only backend (Windows + Mac + Linux)
│   ├── streamer.py         # Streaming execution loop (rotate slots)
│   ├── engine.py           # Main engine: load → shard → stream → generate
│   ├── agent.py            # Unattended agent runner (Phase 6 spec)
│   └── __init__.py
├── src/cli/                # Command-line interface
│   └── chat.py             # Interactive chat CLI
├── tests/                  # Unit tests
├── build/                  # Platform build scripts
│   ├── windows_cuda.bat    # Windows + CUDA build
│   ├── windows_cpu.bat     # Windows CPU-only build
│   └── mac.sh              # macOS build (Apple Silicon + Intel)
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
└── ShardFit Vision and Engineering Brief.md
```

## Quick start

### Prerequisites

- Python 3.10+
- PyTorch (CPU or CUDA variant)
- For CUDA: NVIDIA CUDA toolkit + GPU drivers

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
python -m shardfit.planner --model llama3.1-8b --gpu rtx3050 --context 2048
```

### Run the CLI chat

```bash
# CUDA backend (Windows)
python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cuda

# CPU-only backend (Windows or Mac)
python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cpu

# Mac (Apple Silicon, unified memory)
python3 -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cpu
```

### Interactive chat example

```
$ python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cpu
ShardFit CLI v0.1.0
Backend: cpu | Model: llama-3.1-8b-Q4_K_M | VRAM budget: 4.5 GB
Type your message (or 'quit' to exit)

> What is ShardFit?
Loading shard 0/5 into slot 0...
Loading shard 1/5 into slot 1...
Streaming pass 1/5...
ShardFit is a runtime that runs language models larger than your GPU memory
by streaming model shards through VRAM.
>
```

## Platform support

| Platform | Backend | Status |
|----------|---------|--------|
| Windows + NVIDIA GPU | CUDA | ✅ Supported |
| Windows (no GPU) | CPU | ✅ Supported |
| macOS Apple Silicon | CPU (unified memory) | ✅ Supported |
| macOS Intel | CPU | ✅ Supported |

## Design principles (from the brief)

1. **Lossless:** Output matches a reference implementation running the same weights.
2. **Measure before claiming:** No speed numbers without hardware/model/quantization attached.
3. **Predict before running:** Planner estimates tokens/second before the user commits.
4. **Unattended must be safe:** Budgets, sandboxing, checkpoints, human review gate.
5. **No layer skipping:** Every shard runs on every pass. Only approved approximation is quantization.

## Next phases

- Phase 1: Machine profiler (real bandwidth measurements) — implemented here
- Phase 2: Planner (speed prediction formulas) — implemented here
- Phase 3: Dense streamer (rotating VRAM slots, copy stream)
- Phase 4: MoE path (expert caching)
- Phase 5: Tokens per pass (batching, speculative decoding)
- Phase 6: Agent runner (unattended task loop)
