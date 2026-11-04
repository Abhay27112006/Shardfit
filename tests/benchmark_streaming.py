"""Performance benchmarking for ShardFit Phase 3-4.

Compares streaming tok/s vs CPU-only for models larger than VRAM,
and validates planner predictions against measured results.

Usage:
    python tests/benchmark_streaming.py --model sshleifer/tiny-gpt2 --backend cuda
    python tests/benchmark_streaming.py --model sshleifer/tiny-gpt2 --backend cpu
    python tests/benchmark_streaming.py --model models/llama-3.1-8b-Q4_K_M.gguf --backend cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shardfit.engine import ShardFitEngine, GenerationConfig
from shardfit.planner import Planner, ModelSpec
from shardfit.config import HardwareProfile, VRAMBudget


def run_benchmark(
    model_path: str,
    backend: str = "auto",
    vram_gb: float = 6.0,
    reserve_gb: float = 1.5,
    num_slots: int = 2,
    slot_size_gb: float = 0.5,
    max_context_len: int = 2048,
    max_new_tokens: int = 128,
    num_runs: int = 3,
    temperature: float = 0.7,
    use_disk_backed: bool = True,
    output_file: Optional[str] = None,
) -> dict:
    """Run streaming benchmark and compare with planner predictions."""

    print("=" * 70)
    print("ShardFit Streaming Benchmark")
    print("=" * 70)
    print(f"Model:        {model_path}")
    print(f"Backend:      {backend}")
    print(f"VRAM:         {vram_gb} GB (reserve {reserve_gb} GB)")
    print(f"Slots:        {num_slots} x {slot_size_gb} GB")
    print(f"Disk-backed:  {use_disk_backed}")
    print(f"Max tokens:   {max_new_tokens}")
    print(f"Runs:         {num_runs}")
    print("=" * 70)

    # Initialize engine
    engine = ShardFitEngine(
        model_path=model_path,
        backend_type=backend,
        vram_gb=vram_gb,
        reserve_gb=reserve_gb,
        num_slots=num_slots,
        slot_size_gb=slot_size_gb,
        max_context_len=max_context_len,
        use_disk_backed=use_disk_backed,
    )

    # Load model
    engine.load_model()

    # Get model size from manifest or engine
    if engine.disk_manifest is not None:
        model_size_gb = engine.disk_manifest.total_size_gb
        source_format = engine.disk_manifest.source_format
    elif engine.sharded_weights is not None:
        model_size_gb = engine.sharded_weights.total_size_gb
        source_format = engine.sharded_weights.source_format
    else:
        model_size_gb = 0.0
        source_format = "unknown"

    print(f"\nModel size:   {model_size_gb:.2f} GB ({source_format})")
    print(f"Fits in VRAM: {'Yes' if model_size_gb <= vram_gb - reserve_gb else 'No'}")

    # Run planner prediction
    from shardfit.config import detect_hardware
    hardware = detect_hardware()
    budget = VRAMBudget.calculate(
        vram_total_gb=vram_gb,
        reserve_gb=reserve_gb,
        num_slots=num_slots,
        slot_size_gb=slot_size_gb,
    )
    planner = Planner(hardware=hardware, budget=budget)

    model_spec = ModelSpec(name=model_path, size_gb=model_size_gb)
    plan = planner.plan_dense(
        model=model_spec,
        context_len=max_context_len,
    )

    print(f"\nPlanner prediction:")
    print(f"  Predicted tok/s: {plan.tok_s:.3f}")
    print(f"  Predicted TTFT:  {plan.ttft_estimate_s:.2f}s")
    print(f"  Verdict:         {plan.go_no_go}")
    print(f"  Streamed/pass:   {plan.streamed_per_pass_gb:.2f} GB")

    # Run generation benchmarks
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )

    results = []
    for i in range(num_runs):
        print(f"\n  Run {i + 1}/{num_runs}...")
        result = engine.generate("The quick brown fox jumps over the lazy dog", config)
        results.append(result)
        print(f"    tok/s: {result.tok_s:.3f}, TTFT: {result.ttft_s:.2f}s")

    # Aggregate
    avg_tok_s = sum(r.tok_s for r in results) / len(results)
    avg_ttft = sum(r.ttft_s for r in results) / len(results)
    avg_time = sum(r.total_time_s for r in results) / len(results)

    print(f"\n{'=' * 70}")
    print("Results Summary")
    print(f"{'=' * 70}")
    print(f"Measured tok/s:   {avg_tok_s:.3f}")
    print(f"Predicted tok/s:  {plan.tok_s:.3f}")
    print(f"Accuracy:         {100 * avg_tok_s / plan.tok_s:.1f}% "
          f"(planner vs measured)")
    print(f"Measured TTFT:    {avg_ttft:.2f}s")
    print(f"Predicted TTFT:   {plan.ttft_estimate_s:.2f}s")
    print(f"Backend:          {results[0].backend}")
    print(f"Model size:       {model_size_gb:.2f} GB")
    print(f"Fits VRAM:        {'Yes' if model_size_gb <= vram_gb - reserve_gb else 'No'}")
    print(f"{'=' * 70}")

    # Memory stats
    mem = engine.get_memory_usage()
    print(f"\nMemory usage:")
    print(f"  Backend: {mem['backend_mb']} MB")
    print(f"  KV cache: {mem['kv_cache_mb']:.1f} MB")
    if "moe_cache" in mem:
        print(f"  MoE hit rate: {mem['moe_cache']['hit_rate']:.2%}")

    benchmark_result = {
        "model": model_path,
        "model_size_gb": model_size_gb,
        "source_format": source_format,
        "backend": backend,
        "vram_gb": vram_gb,
        "reserve_gb": reserve_gb,
        "num_slots": num_slots,
        "slot_size_gb": slot_size_gb,
        "use_disk_backed": use_disk_backed,
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "measured_tok_s": avg_tok_s,
        "measured_ttft_s": avg_ttft,
        "measured_total_time_s": avg_time,
        "predicted_tok_s": plan.tok_s,
        "predicted_ttft_s": plan.ttft_estimate_s,
        "planner_accuracy_percent": 100 * avg_tok_s / plan.tok_s if plan.tok_s > 0 else 0,
        "fits_vram": model_size_gb <= vram_gb - reserve_gb,
        "planner_verdict": plan.go_no_go,
        "streamed_per_pass_gb": plan.streamed_per_pass_gb,
        "memory_backend_mb": mem["backend_mb"],
        "memory_kv_cache_mb": mem["kv_cache_mb"],
        "individual_results": [r.tok_s for r in results],
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(benchmark_result, f, indent=2)
        print(f"\nBenchmark results written to {output_file}")

    return benchmark_result


def compare_backends(
    model_path: str,
    vram_gb: float = 6.0,
    max_new_tokens: int = 128,
    num_runs: int = 3,
) -> None:
    """Compare CUDA vs CPU backend for the same model."""
    print("\n" + "=" * 70)
    print("Backend Comparison: CUDA vs CPU")
    print("=" * 70)

    results = {}
    for backend in ["cpu", "cuda"]:
        print(f"\n--- Testing {backend.upper()} backend ---")
        try:
            result = run_benchmark(
                model_path=model_path,
                backend=backend,
                vram_gb=vram_gb,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
            )
            results[backend] = result
        except Exception as e:
            print(f"  {backend.upper()} failed: {e}")
            results[backend] = None

    if results.get("cpu") and results.get("cuda"):
        cpu_tok_s = results["cpu"]["measured_tok_s"]
        gpu_tok_s = results["cuda"]["measured_tok_s"]
        speedup = gpu_tok_s / cpu_tok_s if cpu_tok_s > 0 else 0
        print(f"\n{'=' * 70}")
        print("Comparison Summary")
        print(f"{'=' * 70}")
        print(f"CPU tok/s:  {cpu_tok_s:.3f}")
        print(f"GPU tok/s:  {gpu_tok_s:.3f}")
        print(f"Speedup:    {speedup:.2f}x")
        print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="ShardFit streaming performance benchmark"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model path or HF name (e.g., sshleifer/tiny-gpt2, models/xxx.gguf)",
    )
    parser.add_argument(
        "--backend", choices=["cuda", "cpu", "auto"],
        default="auto", help="Compute backend (default: auto)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare CPU vs GPU backends",
    )
    parser.add_argument(
        "--vram-gb", type=float, default=6.0,
        help="Total VRAM in GB (default: 6.0)",
    )
    parser.add_argument(
        "--reserve-gb", type=float, default=1.5,
        help="Reserved VRAM in GB (default: 1.5)",
    )
    parser.add_argument(
        "--slots", type=int, default=2,
        help="Number of streaming slots (default: 2)",
    )
    parser.add_argument(
        "--slot-gb", type=float, default=0.5,
        help="Slot size in GB (default: 0.5)",
    )
    parser.add_argument(
        "--context", type=int, default=2048,
        help="Context length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max new tokens per run (default: 128)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--no-disk-backed", action="store_true",
        help="Disable disk-backed loading (load full model to CPU RAM)",
    )
    parser.add_argument(
        "--output", help="Write results as JSON to file",
    )
    args = parser.parse_args()

    if args.compare:
        compare_backends(
            model_path=args.model,
            vram_gb=args.vram_gb,
            max_new_tokens=args.max_tokens,
            num_runs=args.runs,
        )
    else:
        run_benchmark(
            model_path=args.model,
            backend=args.backend,
            vram_gb=args.vram_gb,
            reserve_gb=args.reserve_gb,
            num_slots=args.slots,
            slot_size_gb=args.slot_gb,
            max_context_len=args.context,
            max_new_tokens=args.max_tokens,
            num_runs=args.runs,
            temperature=args.temperature,
            use_disk_backed=not args.no_disk_backed,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()
