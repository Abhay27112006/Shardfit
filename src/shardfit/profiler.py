"""Phase 1: Machine profiler.

Reports real host-to-GPU bandwidth for shard-sized pinned copies, SSD sequential
read bandwidth, free VRAM and RAM available for pinning.

Exit criterion (from brief): repeatable results across runs.
"""

from __future__ import annotations

import argparse
import json
import time
import os
import sys
from typing import Optional

import psutil


def measure_host_to_gpu_bandwidth(
    size_gb: float = 0.5, repetitions: int = 5
) -> dict:
    """Measure effective host-to-GPU bandwidth for shard-sized pinned copies.

    Copies a buffer of size_gb from system RAM to GPU and back, measuring
    the effective bandwidth. Uses PyTorch for GPU transfers.
    """
    try:
        import torch
    except ImportError:
        return {
            "error": "PyTorch not installed",
            "bandwidth_gb_s": None,
        }

    if not torch.cuda.is_available():
        return {
            "error": "CUDA not available",
            "bandwidth_gb_s": None,
        }

    device = torch.device("cuda")
    size_bytes = int(size_gb * 1024**3)
    results = []

    for i in range(repetitions):
        # Host -> GPU
        host_tensor = torch.randn(size_bytes // 4, dtype=torch.float32, pin_memory=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_tensor = host_tensor.to(device, non_blocking=True)
        torch.cuda.synchronize()
        h2d_time = time.perf_counter() - t0

        # GPU -> Host
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = gpu_tensor.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        d2h_time = time.perf_counter() - t0

        total_time = h2d_time + d2h_time
        total_bytes = size_bytes * 2  # round trip
        bw_gb_s = (total_bytes / (1024**3)) / total_time
        results.append(
            {
                "repetition": i + 1,
                "size_gb": size_gb,
                "h2d_s": h2d_time,
                "d2h_s": d2h_time,
                "total_s": total_time,
                "bandwidth_gb_s": bw_gb_s,
            }
        )

    bw_values = [r["bandwidth_gb_s"] for r in results]
    return {
        "size_gb": size_gb,
        "repetitions": repetitions,
        "results": results,
        "avg_bandwidth_gb_s": sum(bw_values) / len(bw_values),
        "min_bandwidth_gb_s": min(bw_values),
        "max_bandwidth_gb_s": max(bw_values),
    }


def measure_ssd_read_bandwidth(
    size_gb: float = 1.0, repetitions: int = 3
) -> dict:
    """Measure SSD sequential read bandwidth with read-ahead."""
    import tempfile

    size_bytes = int(size_gb * 1024**3)
    block_size = 4 * 1024**2  # 4 MB blocks
    results = []

    for i in range(repetitions):
        with tempfile.NamedTemporaryFile(delete=False, dir="/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:\\")) as f:
            fname = f.name
            # Write test data first
            remaining = size_bytes
            while remaining > 0:
                chunk = min(block_size, remaining)
                f.write(os.urandom(chunk))
                remaining -= chunk
            f.flush()
            os.fsync(f.fileno())

        # Read sequentially
        t0 = time.perf_counter()
        with open(fname, "rb") as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
        elapsed = time.perf_counter() - t0

        bw_gb_s = (size_bytes / (1024**3)) / elapsed if elapsed > 0 else 0
        results.append(
            {
                "repetition": i + 1,
                "size_gb": size_gb,
                "elapsed_s": elapsed,
                "bandwidth_gb_s": bw_gb_s,
            }
        )

        try:
            os.unlink(fname)
        except OSError:
            pass

    bw_values = [r["bandwidth_gb_s"] for r in results]
    return {
        "size_gb": size_gb,
        "repetitions": repetitions,
        "results": results,
        "avg_bandwidth_gb_s": sum(bw_values) / len(bw_values) if bw_values else 0,
        "min_bandwidth_gb_s": min(bw_values) if bw_values else 0,
        "max_bandwidth_gb_s": max(bw_values) if bw_values else 0,
    }


def measure_free_vram() -> Optional[float]:
    """Return free VRAM in GB, or None if CUDA unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024**3)
    except ImportError:
        pass
    except Exception:
        pass
    return None


def measure_system_ram() -> dict:
    """Return system RAM statistics."""
    ram = psutil.virtual_memory()
    return {
        "total_gb": ram.total / (1024**3),
        "available_gb": ram.available / (1024**3),
        "used_gb": ram.used / (1024**3),
        "percent": ram.percent,
    }


def run_profiler(output: Optional[str] = None) -> dict:
    """Run the full machine profiler and return results."""
    print("=" * 60)
    print("ShardFit Phase 1: Machine Profiler")
    print("=" * 60)

    import platform

    profile = {
        "platform": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        },
        "system_ram": measure_system_ram(),
        "cuda_available": False,
        "host_to_gpu_bandwidth": None,
        "ssd_read_bandwidth": None,
        "free_vram_gb": None,
    }

    try:
        import torch

        profile["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            profile["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            profile["gpu_vram_total_gb"] = props.total_memory / (1024**3)
            profile["free_vram_gb"] = measure_free_vram()
    except ImportError:
        profile["cuda_available"] = False

    print(f"\nPlatform: {profile['platform']['os']} {profile['platform']['architecture']}")
    print(f"System RAM: {profile['system_ram']['total_gb']:.1f} GB total, "
          f"{profile['system_ram']['available_gb']:.1f} GB available")

    if profile["cuda_available"]:
        print(f"\nGPU: {profile['gpu_name']}")
        print(f"VRAM: {profile['gpu_vram_total_gb']:.1f} GB total, "
              f"{profile['free_vram_gb']:.1f} GB free")
    else:
        print("\nGPU: CUDA not available — CPU-only mode")

    # Host-to-GPU bandwidth
    if profile["cuda_available"]:
        print("\n--- Host-to-GPU bandwidth (shard-sized pinned copies) ---")
        h2b = measure_host_to_gpu_bandwidth()
        if h2b.get("error"):
            print(f"  ERROR: {h2b['error']}")
        else:
            print(f"  Size: {h2b['size_gb']} GB, {h2b['repetitions']} repetitions")
            print(f"  Avg bandwidth: {h2b['avg_bandwidth_gb_s']:.2f} GB/s")
            print(f"  Range: {h2b['min_bandwidth_gb_s']:.2f} - {h2b['max_bandwidth_gb_s']:.2f} GB/s")
        profile["host_to_gpu_bandwidth"] = h2b
    else:
        print("\n--- Host-to-GPU bandwidth: skipped (no CUDA) ---")

    # SSD read bandwidth
    print("\n--- SSD sequential read bandwidth ---")
    ssd = measure_ssd_read_bandwidth()
    print(f"  Size: {ssd['size_gb']} GB, {ssd['repetitions']} repetitions")
    print(f"  Avg bandwidth: {ssd['avg_bandwidth_gb_s']:.2f} GB/s")
    print(f"  Range: {ssd['min_bandwidth_gb_s']:.2f} - {ssd['max_bandwidth_gb_s']:.2f} GB/s")
    profile["ssd_read_bandwidth"] = ssd

    print("\n" + "=" * 60)
    print("Profile complete")
    print("=" * 60)

    if output:
        import json as _json

        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
        with open(output, "w") as f:
            _json.dump(profile, f, indent=2)
        print(f"\nResults written to {output}")

    return profile


def main():
    parser = argparse.ArgumentParser(description="ShardFit Phase 1: Machine Profiler")
    parser.add_argument(
        "--output",
        default="results/profile.json",
        help="Output JSON file (default: results/profile.json)",
    )
    args = parser.parse_args()

    run_profiler(output=args.output)


if __name__ == "__main__":
    main()
