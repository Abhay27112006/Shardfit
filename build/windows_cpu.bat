@echo off
REM ShardFit Windows CPU-only build script
REM Builds and runs ShardFit without GPU (CPU inference)

echo ============================================
echo ShardFit Windows CPU Setup
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
echo [3/4] Installing PyTorch (CPU-only)...
pip install torch torchvision torchaudio

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
echo Run the CLI chat (CPU backend):
echo   python -m shardfit.cli.chat --model models/llama-3.1-8b-Q4_K_M --backend cpu
echo.
pause
