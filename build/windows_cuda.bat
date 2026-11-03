@echo off
REM ShardFit Windows + CUDA build script
REM Builds and runs ShardFit with NVIDIA GPU support

echo ============================================
echo ShardFit Windows + CUDA Setup
echo ============================================

echo.
echo [1/4] Checking Python...
python --version || (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

echo.
echo [2/4] Installing Python dependencies...
pip install -r requirements.txt

echo.
echo [3/4] Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/4] Installing ShardFit...
pip install -e .

echo.
echo ============================================
echo Setup complete!
echo ============================================
echo.
echo Run the machine profiler:
echo   python -m shardfit.profiler
echo.
echo Run the planner:
echo   python -m shardfit.planner --model llama-3.1-8b --size-gb 4.9 --gpu "RTX 3050"
echo.
echo Run the CLI chat:
echo   python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cuda
echo.
pause
