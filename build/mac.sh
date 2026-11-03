#!/bin/bash
# ShardFit macOS build script
# Supports Apple Silicon (M1/M2/M3/M4) and Intel Macs

set -e

echo "============================================"
echo "ShardFit macOS Setup"
echo "============================================"

echo ""
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install from https://python.org"
    exit 1
fi
python3 --version

echo ""
echo "[2/4] Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "[3/4] Installing PyTorch (CPU - unified memory on Mac)..."
pip3 install torch torchvision torchaudio

echo ""
echo "[4/4] Installing ShardFit..."
pip3 install -e .

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Run the machine profiler:"
echo "  python3 -m shardfit.profiler"
echo ""
echo "Run the planner:"
echo "  python3 -m shardfit.planner --model llama-3.1-8b --size-gb 4.9 --gpu "Apple Silicon""
echo ""
echo "Run the CLI chat (CPU backend, unified memory):"
echo "  python3 -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cpu"
echo ""
