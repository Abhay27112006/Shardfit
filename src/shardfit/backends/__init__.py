"""Compute backends for ShardFit.

Supports CUDA (NVIDIA GPU) and CPU-only backends.
"""

from shardfit.backends.base import Backend
from shardfit.backends.cuda import CudaBackend
from shardfit.backends.cpu import CpuBackend

__all__ = ["Backend", "CudaBackend", "CpuBackend", "create_backend"]


def create_backend(backend_type: str = "auto", **kwargs) -> Backend:
    """Factory function to create the appropriate backend.

    Args:
        backend_type: "cuda", "cpu", or "auto"
    """
    import torch

    # num_threads only applies to CPU backend
    num_threads = kwargs.pop("num_threads", None)

    if backend_type == "auto":
        if torch.cuda.is_available():
            return CudaBackend()
        return CpuBackend(num_threads=num_threads)

    if backend_type == "cuda":
        return CudaBackend()

    if backend_type == "cpu":
        return CpuBackend(num_threads=num_threads)

    raise ValueError(f"Unknown backend type: {backend_type}")
