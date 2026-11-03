"""CUDA backend for NVIDIA GPUs.

Handles GPU memory allocation, shard loading, and forward passes on CUDA.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from shardfit.backends.base import Backend, BackendStats


class CudaBackend(Backend):
    """CUDA compute backend."""

    def __init__(self, device_id: int = 0):
        super().__init__(device_name=f"cuda:{device_id}")
        self.device_id = device_id
        self._dtype = torch.float16  # quantized weights run in fp16/bf16

    def initialize(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")

        torch.cuda.set_device(self.device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self._initialized = True

        # Warm up
        _ = torch.zeros(1, device=self.device)
        torch.cuda.synchronize()

    def load_shard(self, shard_state_dict: dict) -> None:
        """Move shard weights to GPU memory.

        In a full implementation, this would use pinned memory and async copies
        with CUDA events for double-buffering. Here we do a synchronous copy
        for clarity.
        """
        if not self._initialized:
            self.initialize()

        t0 = time.perf_counter()
        bytes_moved = 0

        for key, tensor in shard_state_dict.items():
            if tensor is None:
                continue
            bytes_moved += tensor.element_size() * tensor.numel()
            # Move to GPU
            _ = tensor.to(device=self.device, dtype=self._dtype, non_blocking=False)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self.record_transfer(bytes_moved, elapsed)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run a forward pass on CUDA.

        This is a placeholder — the actual model forward is implemented in
        shardfit.engine.ShardFitEngine which orchestrates shard-by-shard
        computation. This backend method provides the device primitive.
        """
        if not self._initialized:
            self.initialize()

        input_ids = input_ids.to(device=self.device)
        t0 = time.perf_counter()

        # Placeholder: identity forward (real model logic in engine)
        logits = input_ids.float()
        elapsed = time.perf_counter() - t0

        self.record_forward(elapsed)

        return logits, kv_cache or {}

    def get_memory_used_mb(self) -> int:
        if not torch.cuda.is_available() or self.device is None:
            return 0
        return int(torch.cuda.memory_allocated(self.device) / (1024**2))

    def clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def supports_cuda(self) -> bool:
        return True
